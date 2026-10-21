import torch
import torch.nn.functional as F


def quantize_symmetric(weight, bits=8, granularity="channel"):
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if granularity not in ("tensor", "channel"):
        raise ValueError("Unsupported granularity")
    if weight.ndim != 2:
        raise ValueError("Expected a 2D weight matrix")

    weight = weight.detach().float()
    qmax = 2 ** (bits - 1) - 1

    if granularity == "channel":
        max_abs = weight.abs().amax(dim=1, keepdim=True)
    else:
        max_abs = weight.abs().amax()

    scale = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs / qmax,
    )
    codes = torch.round(weight / scale).clamp(-qmax, qmax)
    return codes.to(torch.int8), scale


def quantize_asymmetric(weight, bits=4, granularity="channel"):
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if granularity not in ("tensor", "channel"):
        raise ValueError("Unsupported granularity")
    if weight.ndim != 2:
        raise ValueError("Expected a 2D weight matrix")

    weight = weight.detach().float()
    qmax = 2 ** bits - 1

    if granularity == "channel":
        minimum = weight.amin(dim=1, keepdim=True)
        maximum = weight.amax(dim=1, keepdim=True)
    else:
        minimum = weight.amin()
        maximum = weight.amax()

    minimum = torch.minimum(minimum, torch.zeros_like(minimum))
    maximum = torch.maximum(maximum, torch.zeros_like(maximum))
    span = maximum - minimum

    scale = torch.where(
        span == 0, torch.ones_like(span), span / qmax
    )
    zero_point = torch.round(-minimum / scale).clamp(0, qmax)
    codes = (torch.round(weight / scale) + zero_point).clamp(0, qmax)
    return codes.to(torch.uint8), scale, zero_point.to(torch.uint8)


def pack_int4(codes, symmetric=True):
    flat = codes.detach().reshape(-1).to(torch.int16)
    lower, upper = (-7, 7) if symmetric else (0, 15)

    if not ((flat >= lower) & (flat <= upper)).all().item():
        raise ValueError("Codes outside the allowed INT4 range")

    if symmetric:
        flat = flat + 7
    flat = flat.to(torch.uint8)

    if flat.numel() % 2:
        flat = torch.cat([flat, flat.new_zeros(1)])

    return flat[0::2] | (flat[1::2] << 4)


def unpack_int4(packed, shape, symmetric=True):
    count = 1
    for dimension in shape:
        count *= dimension

    if packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("Expected a one-dimensional uint8 tensor")
    if packed.numel() != (count + 1) // 2:
        raise ValueError("Packed length does not match shape")

    flat = torch.empty(
        packed.numel() * 2,
        dtype=torch.uint8,
        device=packed.device,
    )
    flat[0::2] = packed & 15
    flat[1::2] = (packed >> 4) & 15
    flat = flat[:count]

    if symmetric:
        flat = (flat.to(torch.int16) - 7).to(torch.int8)
    return flat.reshape(shape)


class QuantizedLinear(torch.nn.Module):
    def __init__(
        self, weight, bias=None, bits=4,
        granularity="channel", symmetric=True, collect_stats=False,
    ):
        super().__init__()
        self.bits = bits
        self.granularity = granularity
        self.symmetric = symmetric
        self.out_features, self.in_features = weight.shape
        self.weight_shape = tuple(weight.shape)

        with torch.no_grad():
            if symmetric:
                q, scale = quantize_symmetric(weight, bits, granularity)
                zero_point = None
            else:
                q, scale, zero_point = quantize_asymmetric(
                    weight, bits, granularity
                )

            self.conversion_stats = None
            if collect_stats:
                original = weight.detach().float()
                reconstructed = q.float()
                if not symmetric:
                    reconstructed.sub_(zero_point.float())
                reconstructed.mul_(scale)
                difference = reconstructed - original

                self.conversion_stats = {
                    "squared_error": difference.square().sum(
                        dtype=torch.float64
                    ).item(),
                    "squared_weight": original.square().sum(
                        dtype=torch.float64
                    ).item(),
                    "weight_count": original.numel(),
                }
                del reconstructed, difference, original

            stored = pack_int4(q, symmetric) if bits == 4 else q
            self.register_buffer("weight_codes", stored)
            self.register_buffer("scale", scale)
            self.register_buffer("zero_point", zero_point)
            self.register_buffer(
                "bias",
                None if bias is None else bias.detach().float().clone(),
            )

    def reconstruct_weight(self):
        q = (
            unpack_int4(
                self.weight_codes, self.weight_shape, self.symmetric
            )
            if self.bits == 4 else self.weight_codes
        )
        weight = q.float()
        if not self.symmetric:
            weight.sub_(self.zero_point.float())
        return weight.mul_(self.scale)

    def forward(self, inputs):
        return F.linear(inputs, self.reconstruct_weight(), self.bias)


CONFIGS = {
    "int8_channel_sym": dict(
        bits=8, granularity="channel", symmetric=True
    ),
    "int4_channel_sym": dict(
        bits=4, granularity="channel", symmetric=True
    ),
    "int4_tensor_sym": dict(
        bits=4, granularity="tensor", symmetric=True
    ),
    "int4_channel_asym": dict(
        bits=4, granularity="channel", symmetric=False
    ),
}


def convert_linear_layers(model_to_convert, settings):
    output_layer = model_to_convert.get_output_embeddings()
    names = [
        name for name, layer in model_to_convert.named_modules()
        if isinstance(layer, torch.nn.Linear)
        and layer is not output_layer
    ]

    total_error = 0.0
    total_weight = 0.0
    total_count = 0
    layer_results = []

    with torch.inference_mode():
        for name in names:
            original = model_to_convert.get_submodule(name)
            replacement = QuantizedLinear(
                original.weight, original.bias,
                collect_stats=True, **settings,
            )
            stats = replacement.conversion_stats
            count = stats["weight_count"]
            total_error += stats["squared_error"]
            total_weight += stats["squared_weight"]
            total_count += count
            layer_results.append({
                "layer": name,
                "weight_count": count,
                "mse": stats["squared_error"] / count,
            })
            del replacement.conversion_stats

            parent_name, _, child_name = name.rpartition(".")
            parent = (
                model_to_convert.get_submodule(parent_name)
                if parent_name else model_to_convert
            )
            setattr(parent, child_name, replacement)
            del original, replacement

    return {
        "layers_quantized": len(names),
        "weights_quantized": total_count,
        "weight_mse": total_error / total_count if total_count else 0.0,
        "relative_squared_error": (
            total_error / total_weight if total_weight else 0.0
        ),
        "per_layer": layer_results,
    }
