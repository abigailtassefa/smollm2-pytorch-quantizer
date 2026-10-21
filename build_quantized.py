
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from quantizer import CONFIGS, convert_linear_layers
from checkpoints import save_quantized, load_quantized

root = Path("/content/model_quantizer")
name = sys.argv[1]
settings = CONFIGS[name]
folder = root / name

environment = json.loads(
    (root / "environment.json").read_text()
)
torch.set_num_threads(environment["cpu_threads"])

print(f"Loading FP32 baseline for {name}...", flush=True)
candidate = AutoModelForCausalLM.from_pretrained(
    root / "baseline_fp32",
    dtype=torch.float32,
    device_map="cpu",
    local_files_only=True,
)
candidate.eval()
candidate.requires_grad_(False)

print("Quantizing internal linear layers...", flush=True)
statistics = convert_linear_layers(candidate, settings)

assert statistics["layers_quantized"] == 210
assert statistics["weights_quantized"] == 106_168_320

workload = torch.load(
    root / "workload.pt",
    map_location="cpu",
    weights_only=True,
)
probe = workload["eval_blocks"][:1, :16]

with torch.inference_mode():
    output = candidate(input_ids=probe, use_cache=False)
    reference_logits = output.logits[:, -1, :].clone()
    assert torch.isfinite(reference_logits).all()
    del output

print("Saving quantized checkpoint...", flush=True)
save_quantized(candidate, folder, settings, statistics)

# Release the converted model before testing the loader.
del candidate
gc.collect()

print("Reloading compressed checkpoint...", flush=True)
restored = load_quantized(folder)

with torch.inference_mode():
    output = restored(input_ids=probe, use_cache=False)
    restored_logits = output.logits[:, -1, :]
    torch.testing.assert_close(
        restored_logits,
        reference_logits,
        rtol=1e-5,
        atol=1e-5,
    )
    maximum_difference = (
        restored_logits - reference_logits
    ).abs().max().item()

checkpoint_mb = (
    folder / "model.safetensors"
).stat().st_size / 1e6

print(f"\nConfiguration: {name}")
print(f"Layers quantized: {statistics['layers_quantized']}")
print(f"Weights quantized: {statistics['weights_quantized']:,}")
print(f"Weight MSE: {statistics['weight_mse']:.8g}")
print(f"Checkpoint size: {checkpoint_mb:.2f} MB")
print(f"Reload maximum logit difference: {maximum_difference:.8g}")
print("PASS: quantization, inference, saving, and reloading.")
print("Peak inference RAM and perplexity are not measured yet.")
