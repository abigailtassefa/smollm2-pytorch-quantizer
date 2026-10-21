import argparse
import gc
import json
import math
import platform
import statistics
import threading
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM

from checkpoints import load_quantized

ROOT = Path("/content/model_quantizer")
PROCESS = psutil.Process()


class MemorySampler:
    """Sample this process's physical RAM every 5 milliseconds."""
    def __init__(self):
        self.peak = PROCESS.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            self.peak = max(self.peak, PROCESS.memory_info().rss)
            self.stop_event.wait(0.005)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, PROCESS.memory_info().rss)
        self.stop_event.set()
        self.thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("mode", choices=["quality_ram", "speed"])
    args = parser.parse_args()

    environment = json.loads((ROOT / "environment.json").read_text())
    torch.set_num_threads(environment["cpu_threads"])
    torch.manual_seed(42)

    workload = torch.load(
        ROOT / "workload.pt",
        map_location="cpu",
        weights_only=True,
    )
    framework_ram = PROCESS.memory_info().rss
    folder = ROOT / args.name

    with MemorySampler() as loading_memory:
        if args.name == "baseline_fp32":
            model = AutoModelForCausalLM.from_pretrained(
                folder,
                dtype=torch.float32,
                device_map="cpu",
                local_files_only=True,
            )
        else:
            model = load_quantized(folder)

        model.eval()
        model.requires_grad_(False)

    gc.collect()
    loaded_ram = PROCESS.memory_info().rss

    input_ids = workload["timing_input_ids"]
    attention_mask = workload["timing_attention_mask"]
    generation_settings = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": 50,
        "min_new_tokens": 50,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": model.config.eos_token_id,
    }

    result = {
        "configuration": args.name,
        "mode": args.mode,
        "cpu_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "cpu_description": platform.processor(),
        "total_system_ram_mb": psutil.virtual_memory().total / 1e6,
        "checkpoint_mb": sum(
            path.stat().st_size
            for path in folder.glob("*.safetensors")
        ) / 1e6,
    }

    with torch.inference_mode():
        if args.mode == "quality_ram":
            total_nll = 0.0
            predicted_tokens = 0

            # Include first-use allocations: no inference warmup here.
            with MemorySampler() as inference_memory:
                for block in workload["eval_blocks"]:
                    tokens = block.unsqueeze(0)
                    output = model(
                        input_ids=tokens,
                        labels=tokens,
                        use_cache=False,
                    )
                    loss = output.loss.item()
                    if not math.isfinite(loss):
                        raise RuntimeError("Non-finite evaluation loss")

                    count = tokens.shape[1] - 1
                    total_nll += loss * count
                    predicted_tokens += count
                    del output

                generated = model.generate(**generation_settings)
                assert generated.shape[1] - input_ids.shape[1] == 50
                del generated

            mean_loss = total_nll / predicted_tokens
            result.update({
                "framework_ram_mb": framework_ram / 1e6,
                "loaded_ram_mb": loaded_ram / 1e6,
                "peak_loading_ram_mb": loading_memory.peak / 1e6,
                "peak_inference_ram_mb": inference_memory.peak / 1e6,
                "memory_sampling_interval_ms": 5,
                "meets_1gb_budget": inference_memory.peak < 1_000_000_000,
                "evaluated_tokens": predicted_tokens,
                "mean_nll": mean_loss,
                "perplexity": math.exp(mean_loss),
            })

        else:
            # Warmup is excluded from timing.
            warmup = model.generate(**generation_settings)
            del warmup

            durations = []
            for _ in range(3):
                start = time.perf_counter()
                generated = model.generate(**generation_settings)
                duration = time.perf_counter() - start
                assert generated.shape[1] - input_ids.shape[1] == 50
                durations.append(duration)
                del generated

            median_seconds = statistics.median(durations)
            result.update({
                "generation_tokens": 50,
                "timing_trials_seconds": durations,
                "median_generation_seconds": median_seconds,
                "tokens_per_second": 50 / median_seconds,
            })

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    destination = results_dir / f"{args.name}_{args.mode}.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
