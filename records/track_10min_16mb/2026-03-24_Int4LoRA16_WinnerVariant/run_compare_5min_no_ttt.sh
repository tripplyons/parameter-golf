#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WINNER_DIR="${SCRIPT_DIR%/2026-03-24_Int4LoRA16_WinnerVariant}/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon"
NEW_DIR="${SCRIPT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SEED="${SEED:-1337}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
ITERATIONS="${ITERATIONS:-9000}"
EVAL_STRIDE="${EVAL_STRIDE:-64}"

run_model() {
    local label="$1"
    local workdir="$2"
    local script_path="$3"
    local run_id="$4"
    local bigram_vocab_size="$5"
    local ve_dim="$6"
    local lora_space_profile="$7"

    echo "===================================================================="
    echo "Running ${label}"
    echo "  workdir: ${workdir}"
    echo "  script : ${script_path}"
    echo "  run_id : ${run_id}"
    echo "===================================================================="

    (
        cd "${workdir}"
        env \
            RUN_ID="${run_id}" \
            SEED="${SEED}" \
            NUM_LAYERS="${NUM_LAYERS:-11}" \
            BIGRAM_VOCAB_SIZE="${bigram_vocab_size}" \
            XSA_LAST_N="${XSA_LAST_N:-4}" \
            EMA_ENABLED="${EMA_ENABLED:-1}" \
            EMA_DECAY="${EMA_DECAY:-0.997}" \
            SWA_ENABLED="${SWA_ENABLED:-1}" \
            SWA_EVERY="${SWA_EVERY:-50}" \
            ROPE_DIMS="${ROPE_DIMS:-16}" \
            LN_SCALE="${LN_SCALE:-1}" \
            LATE_QAT="${LATE_QAT:-1}" \
            LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.15}" \
            VE_ENABLED="${VE_ENABLED:-1}" \
            VE_DIM="${ve_dim}" \
            VE_LAYERS="${VE_LAYERS:-9,10}" \
            TTT_ENABLED=0 \
            MUON_WD="${MUON_WD:-0.04}" \
            ADAM_WD="${ADAM_WD:-0.04}" \
            MATRIX_LR="${MATRIX_LR:-0.025}" \
            SCALAR_LR="${SCALAR_LR:-0.025}" \
            TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}" \
            MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}" \
            MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}" \
            MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}" \
            WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}" \
            ITERATIONS="${ITERATIONS}" \
            MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
            EVAL_STRIDE="${EVAL_STRIDE}" \
            LORA_SPACE_PROFILE="${lora_space_profile}" \
            torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${script_path}"
    )
}

run_model \
    "Previous best (5 min, no TTT)" \
    "${WINNER_DIR}" \
    "${WINNER_DIR}/train_gpt.py" \
    "prevbest_5min_no_ttt_seed${SEED}" \
    "${BASE_BIGRAM_VOCAB_SIZE:-1536}" \
    "${BASE_VE_DIM:-128}" \
    "${BASE_LORA_SPACE_PROFILE:-off}"

run_model \
    "Int5 blockwise + FP8 LoRA variant (5 min, no TTT)" \
    "${NEW_DIR}" \
    "${NEW_DIR}/train_gpt.py" \
    "int5_block_fp8_lora_5min_no_ttt_seed${SEED}" \
    "${NEW_BIGRAM_VOCAB_SIZE:-5120}" \
    "${NEW_VE_DIM:-288}" \
    "${NEW_LORA_SPACE_PROFILE:-expand_effective}"
