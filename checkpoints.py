import json
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoConfig, AutoModelForCausalLM

from quantizer import QuantizedLinear


def save_quantized(model, folder, settings, statistics):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(folder)

    state = model.state_dict()
    if not model.config.tie_word_embeddings:
        raise ValueError("Expected tied embeddings")

    embedding_key = "model.embed_tokens.weight"
    output_key = "lm_head.weight"

    if state[embedding_key].data_ptr() != state[output_key].data_ptr():
        raise ValueError("Embedding/output weights are not shared")

    stored_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if name != output_key
    }
    save_file(stored_state, str(folder / "model.safetensors"))

    manifest = {
        "format_version": 1,
        "settings": settings,
        "statistics": statistics,
        "layers": {
            name: {"weight_shape": list(layer.weight_shape)}
            for name, layer in model.named_modules()
            if isinstance(layer, QuantizedLinear)
        },
    }
    (folder / "quantization.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def load_quantized(folder):
    folder = Path(folder)
    manifest = json.loads(
        (folder / "quantization.json").read_text(encoding="utf-8")
    )
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported checkpoint version")

    state = load_file(str(folder / "model.safetensors"), device="cpu")
    config = AutoConfig.from_pretrained(folder, local_files_only=True)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, dtype=torch.float32
        )

    # Rebuild non-checkpoint rotary-position buffers on CPU.
    for name, module in list(model.named_modules()):
        if name.endswith("rotary_emb"):
            with torch.device("cpu"):                 
              replacement = type(module)(config=config)
            parent_name, _, child_name = name.rpartition(".")
            parent = (
                model.get_submodule(parent_name)
                if parent_name else model
            )
            setattr(parent, child_name, replacement)

    settings = manifest["settings"]

    for name, information in manifest["layers"].items():
        layer = QuantizedLinear.__new__(QuantizedLinear)
        torch.nn.Module.__init__(layer)

        layer.bits = settings["bits"]
        layer.granularity = settings["granularity"]
        layer.symmetric = settings["symmetric"]
        layer.weight_shape = tuple(information["weight_shape"])
        layer.out_features, layer.in_features = layer.weight_shape

        for buffer_name in ["weight_codes", "scale", "zero_point", "bias"]:
            key = f"{name}.{buffer_name}"
            buffer = (
                torch.empty_like(state[key], device="meta")
                if key in state else None
            )
            layer.register_buffer(buffer_name, buffer)

        parent_name, _, child_name = name.rpartition(".")
        parent = (
            model.get_submodule(parent_name)
            if parent_name else model
        )
        setattr(parent, child_name, layer)

    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.eval()
    model.requires_grad_(False)

    meta_names = [
        name
        for name, tensor in list(model.named_parameters())
        + list(model.named_buffers())
        if tensor.is_meta
    ]
    if meta_names:
        raise RuntimeError(f"Unloaded meta tensors: {meta_names}")

    if model.get_input_embeddings().weight is not (
        model.get_output_embeddings().weight
    ):
        raise RuntimeError("Weight sharing was not restored")

    return model
