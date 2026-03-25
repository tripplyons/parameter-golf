# Int5 Blockwise + FP8 LoRA Winner Variant

This variant keeps the full `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` training and evaluation stack, but swaps the export codec:

- Large 2D weight matrices are quantized to packed 5-bit weights with per-row, per-block scaling.
- Each quantized matrix also stores a blockwise-scaled FP8 low-rank residual adapter (`LoRA`) to recover part of the quantization loss.
- Small tensors still pass through in fp16 or fp32, matching the original script's control-tensor handling.
- The wrapper also rewrites the training-time QAT path so fake quantization uses the same blockwise int5 scheme as export.

The implementation is intentionally minimal:

- `train_gpt.py` loads the winning record script and replaces only its quantization section.
- `int5_lora_fp8_quant.py` contains the new codec and blockwise int5 QAT helpers.

## Backbone shrinking

This variant can also deliberately shrink the dense model to leave more room for FP8 LoRA residuals. The wrapper applies a `LORA_SPACE_PROFILE` before handing off to the original winner script:

- `LORA_SPACE_PROFILE=auto` (default): choose the least-shrunk profile that still leaves contest-safe headroom for LoRA under the true code+model budget
- `LORA_SPACE_PROFILE=off`: keep the original winner defaults
- `LORA_SPACE_PROFILE=expand_effective`: `NUM_LAYERS=12`, `MLP_MULT=3.5`, `BIGRAM_VOCAB_SIZE=5120`, `BIGRAM_DIM=256`, `VE_DIM=288`
- `LORA_SPACE_PROFILE=light`: `MODEL_DIM=480`, `MLP_MULT=2.75`, `BIGRAM_VOCAB_SIZE=1536`, `BIGRAM_DIM=96`, `VE_DIM=96`
- `LORA_SPACE_PROFILE=medium`: `NUM_LAYERS=10`, `MODEL_DIM=480`, `MLP_MULT=2.5`, `BIGRAM_VOCAB_SIZE=1024`, `BIGRAM_DIM=96`, `VE_DIM=96`
- `LORA_SPACE_PROFILE=heavy`: `NUM_LAYERS=10`, `MODEL_DIM=448`, `MLP_MULT=2.5`, `BIGRAM_VOCAB_SIZE=1024`, `BIGRAM_DIM=64`, `VE_DIM=64`

Any explicitly provided environment variable still wins over the profile.
The auto selector uses `INT5_LORA_AUTO_MIN_HEADROOM_BYTES=524288` by default.

## Default knobs

The codec is controlled through environment variables:

- `INT5_LORA_RANK=128`
- `INT5_LORA_MIN_NUMEL=65536`
- `INT5_LORA_TARGET_CATEGORIES=attn,mlp,embed,other`
- `INT5_TRAIN_QAT=0`
- `INT5_BLOCK_SIZE=64`
- `FP8_LORA_BLOCK_SIZE=64`
- `FP8_LORA_DTYPE=float8_e4m3fn`
- `INT5_LORA_MAX_TOTAL_BYTES=16000000`
- `INT5_LORA_SAFETY_BYTES=16384`

The export path now enforces a global size budget. It starts from pure blockwise-int5 base weights, scores candidate LoRA rank increments by captured residual energy per added byte, and only keeps the highest-value increments that still fit under the 16MB cap after accounting for the wrapper, the helper, and the referenced base winner script. The default rank cap is intentionally generous so the allocator can spend most of the available budget instead of getting bottlenecked by a low per-matrix LoRA limit.

## Notes

- The LoRA factors are derived post-training from the quantization residual via low-rank factorization.
- By default the wrapper preserves the previous-best dense training path; set `INT5_TRAIN_QAT=1` to enable late-stage int5 fake quant during training.
- LoRA rows are stored as blockwise-scaled FP8 tensors, with separate scales for `A` and `B`.
- Artifact names and metric labels are rewritten from the base script to use the `int5_block_fp8_lora` suffix.
