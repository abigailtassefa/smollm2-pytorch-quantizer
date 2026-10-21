# SmolLM2 custom PyTorch quantizer

Raw-PyTorch quantization of HuggingFaceTB/SmolLM2-135M. No pre-built quantization library is used.

## Objective

Keep absolute peak inference-process physical RAM below 1,000 MB on the tested Colab CPU runtime.

## Three comparisons

- INT8 versus INT4: per-channel symmetric.
- Per-tensor versus per-channel: INT4 symmetric.
- Symmetric versus asymmetric: INT4 per-channel.

All configurations quantize 210 internal linear layers. Embeddings, normalization, and the shared output projection remain FP32. INT4 codes are packed two per byte; computation reconstructs weights per layer.

## Results

| Configuration | Checkpoint MB | Peak RAM MB | MSE | Perplexity | 50-token seconds |
|---|---:|---:|---:|---:|---:|
| baseline_fp32 | 538.09 | 1105.4 | 0 | 36.203 | 3.700 |
| int8_channel_sym | 220.23 | 788.7 | 2.698376e-06 | 36.585 | 8.704 |
| int4_channel_sym | 167.15 | 730.7 | 0.0008829628 | 81.945 | 23.265 |
| int4_tensor_sym | 166.52 | 729.6 | 0.01255635 | 4976543.760 | 20.564 |
| int4_channel_asym | 167.33 | 730.6 | 0.000653875 | 65.510 | 20.419 |

## Decision

INT8 per-channel symmetric met the RAM target with the lowest perplexity among quantized versions. All quantized versions were slower than FP32 with this custom implementation.

## Evaluation

Identical WikiText-2 inputs: 16 blocks of 128 tokens and 2,032 next-token predictions. Two CPU threads. Fresh processes per configuration and measurement mode. RAM sampled every 5 ms during evaluation and generation. Separate speed runs used 50 generated tokens, one warmup, and three timed trials; median time is reported.

## Reproduction

Use the accompanying notebook in Colab. Scripts expect /content/model_quantizer. Save the FP32 baseline there under baseline_fp32, place scripts in /content, and place workload.pt and environment.json in /content/model_quantizer.

Example commands:
```bash
python /content/build_quantized.py int8_channel_sym
python /content/benchmark.py int8_channel_sym quality_ram
python /content/benchmark.py int8_channel_sym speed
```

See SmolLM2_Custom_Quantizer.ipynb for the notebook workflow. requirements.txt records package versions; workload.pt preserves exact evaluated token IDs.

## Limitations

Small evaluation sample, one timing prompt, sampled RAM may miss brief peaks, and cloud measurements do not verify laptop performance. Loading and conversion peaks differ from inference peaks.

## Model

https://huggingface.co/Abbyxx/smollm2-135m-custom-int8

## Attribution

Source model: HuggingFaceTB/SmolLM2-135M, Apache 2.0. This project changes its weight representation; source-model authorship is not claimed.
