from __future__ import annotations

import os
from pathlib import Path


BASE_RECORD = (
    Path(__file__).resolve().parent.parent
    / "2026-03-23_LeakyReLU_LegalTTT_ParallelMuon"
    / "train_gpt.py"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
QUANT_HELPER = Path(__file__).resolve().with_name("int5_lora_fp8_quant.py")

QUANT_SECTION_HEADER = "# --- GPTQ-lite int6 quantization ---"
TRAIN_SECTION_HEADER = "# --- Training ---"
QUANT_REPLACEMENT = """# --- Int5 blockwise + fp8 LoRA quantization ---
from int5_lora_fp8_quant import (
    _classify_param,
    _qat_embedding,
    _qat_weight,
    _rebank_state_dict,
    _unbank_state_dict,
    dequantize_int5_lora as dequantize_mixed_int6,
    mixed_quantize_int5_lora as mixed_quantize_int6,
)

"""

STRING_REPLACEMENTS = (
    ("final_model.int6.ptz", "final_model.int5_block_fp8_lora.ptz"),
    ("Serialized model int6+lzma", "Serialized model int5+fp8lora+lzma"),
    ("Total submission size int6+lzma", "Total submission size int5+fp8lora+lzma"),
    ("final_int6_roundtrip", "final_int5_block_fp8_lora_roundtrip"),
    ("final_int6_sliding_window", "final_int5_block_fp8_lora_sliding_window"),
    ("final_int8_zlib_roundtrip_exact", "final_int5_block_fp8_lora_roundtrip_exact"),
)
CODE_REPLACEMENT_OLD = '    code = Path(__file__).read_text(encoding="utf-8")'
CODE_REPLACEMENT_NEW = f"""    code = Path(__file__).read_text(encoding="utf-8")
    base_record_path = Path(r"{BASE_RECORD}")
    if base_record_path.exists():
        code += "\\n\\n# --- base_record_train_gpt.py ---\\n\\n" + base_record_path.read_text(encoding="utf-8")
    quant_path = Path(r"{QUANT_HELPER}")
    if quant_path.exists():
        code += "\\n\\n# --- int5_lora_fp8_quant.py ---\\n\\n" + quant_path.read_text(encoding="utf-8")"""

CASTED_LINEAR_OLD = """    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)"""

CASTED_LINEAR_NEW = """    def forward(self, x: Tensor) -> Tensor:
        w = _qat_weight(
            self.weight,
            self.training and CastedLinear._qat_enabled and bool(int(os.environ.get("INT5_TRAIN_QAT", "0"))),
        ).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)"""

QUANT_EXPORT_OLD = """    # Unbank 3D tensors into individual 2D tensors for quantization
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"})
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=6)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+lzma: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+lzma: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()"""

QUANT_EXPORT_NEW = """    # Unbank 3D tensors into individual 2D tensors for quantization
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    if master_process:
        t_quantize = time.perf_counter()
        quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"})
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = lzma.compress(quant_raw, preset=6)
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+lzma: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+lzma: {quant_file_bytes + code_bytes} bytes")
        log0(f"quantize_serialize_time:{1000.0 * (time.perf_counter() - t_quantize):.0f}ms")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()"""

BANK_QAT_REPLACEMENTS = (
    ("self.qo_bank[i]", "_qat_weight(self.qo_bank[i], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.kv_bank[i]", "_qat_weight(self.kv_bank[i], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.kv_bank[n + i]", "_qat_weight(self.kv_bank[n + i], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.qo_bank[n + i]", "_qat_weight(self.qo_bank[n + i], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.mlp_up_bank[i]", "_qat_weight(self.mlp_up_bank[i], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.mlp_down_bank[i]", "_qat_weight(self.mlp_down_bank[i], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.qo_bank[bi]", "_qat_weight(self.qo_bank[bi], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.kv_bank[bi]", "_qat_weight(self.kv_bank[bi], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.kv_bank[n + bi]", "_qat_weight(self.kv_bank[n + bi], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.qo_bank[n + bi]", "_qat_weight(self.qo_bank[n + bi], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.mlp_up_bank[bi]", "_qat_weight(self.mlp_up_bank[bi], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
    ("self.mlp_down_bank[bi]", "_qat_weight(self.mlp_down_bank[bi], self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))"),
)

SPACE_PROFILES = {
    "auto": {},
    "off": {},
    # Contest-budget growth: target roughly a 15.7 MB compressed artifact based on
    # the current measured estimate->compressed ratio from local runs.
    "expand_effective": {
        "NUM_LAYERS": "12",
        "MLP_MULT": "3.5",
        "BIGRAM_VOCAB_SIZE": "5120",
        "BIGRAM_DIM": "256",
        "VE_DIM": "288",
    },
    # Small trim: keep 11 layers and head layout, but trim the widest/least-compressible pieces.
    "light": {
        "MODEL_DIM": "480",
        "MLP_MULT": "2.75",
        "BIGRAM_VOCAB_SIZE": "1536",
        "BIGRAM_DIM": "96",
        "VE_DIM": "96",
    },
    # Medium trim: one fewer layer and slightly smaller side modules.
    "medium": {
        "NUM_LAYERS": "10",
        "MODEL_DIM": "480",
        "MLP_MULT": "2.5",
        "BIGRAM_VOCAB_SIZE": "1024",
        "BIGRAM_DIM": "96",
        "VE_DIM": "96",
    },
    # Heavy trim: leave more headroom for higher LoRA rank.
    "heavy": {
        "NUM_LAYERS": "10",
        "MODEL_DIM": "448",
        "MLP_MULT": "2.5",
        "BIGRAM_VOCAB_SIZE": "1024",
        "BIGRAM_DIM": "64",
        "VE_DIM": "64",
    },
}

PROFILE_ORDER = ["expand_effective", "off", "light", "medium", "heavy"]
PROFILE_KEYS = {
    "NUM_LAYERS",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "MODEL_DIM",
    "MLP_MULT",
    "BIGRAM_VOCAB_SIZE",
    "BIGRAM_DIM",
    "VE_DIM",
}
WINNER_DEFAULTS = {
    "VOCAB_SIZE": 1024,
    "NUM_LAYERS": 11,
    "NUM_HEADS": 8,
    "NUM_KV_HEADS": 4,
    "MODEL_DIM": 512,
    "MLP_MULT": 3.0,
    "BIGRAM_VOCAB_SIZE": 2048,
    "BIGRAM_DIM": 128,
    "VE_ENABLED": 1,
    "VE_DIM": 128,
    "VE_LAYERS": "9,10",
    "TIE_EMBEDDINGS": 1,
}

PREV_BEST_DEFAULT_ENV = {
    "DATA_PATH": str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"),
    "TOKENIZER_PATH": str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"),
    "NUM_LAYERS": "11",
    "BIGRAM_VOCAB_SIZE": "1536",
    "XSA_LAST_N": "4",
    "EMA_ENABLED": "1",
    "EMA_DECAY": "0.997",
    "SWA_ENABLED": "1",
    "SWA_EVERY": "50",
    "ROPE_DIMS": "16",
    "LN_SCALE": "1",
    "LATE_QAT_THRESHOLD": "0.15",
    "VE_ENABLED": "1",
    "VE_DIM": "128",
    "VE_LAYERS": "9,10",
    "TTT_ENABLED": "1",
    "TTT_LR": "0.002",
    "TTT_EPOCHS": "3",
    "TTT_CHUNK_TOKENS": "32768",
    "TTT_FREEZE_BLOCKS": "0",
    "TTT_MOMENTUM": "0.9",
    "TTT_BATCH_SEQS": "32",
    "TTT_GRAD_CLIP": "1.0",
    "MUON_WD": "0.04",
    "ADAM_WD": "0.04",
    "MATRIX_LR": "0.025",
    "SCALAR_LR": "0.025",
    "TIED_EMBED_LR": "0.035",
    "MUON_MOMENTUM": "0.99",
    "MUON_MOMENTUM_WARMUP_START": "0.92",
    "MUON_MOMENTUM_WARMUP_STEPS": "1500",
    "WARMDOWN_ITERS": "3500",
    "ITERATIONS": "9000",
    "MAX_WALLCLOCK_SECONDS": "600",
    "EVAL_STRIDE": "64",
    "LORA_SPACE_PROFILE": "off",
}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _code_budget_paths() -> list[Path]:
    return [Path(__file__).resolve(), BASE_RECORD, QUANT_HELPER]


def _submission_budget_bytes() -> int:
    total_budget = _env_int("INT5_LORA_MAX_TOTAL_BYTES", 16_000_000)
    safety = _env_int("INT5_LORA_SAFETY_BYTES", 16_384)
    code_bytes = sum(path.stat().st_size for path in _code_budget_paths() if path.exists())
    return max(total_budget - code_bytes - safety, 1)


def _profile_value(profile: str, key: str) -> str:
    if key in os.environ:
        return os.environ[key]
    if key in SPACE_PROFILES[profile]:
        return SPACE_PROFILES[profile][key]
    return str(WINNER_DEFAULTS[key])


def _int5_matrix_bytes(rows: int, cols: int) -> int:
    block_size = _env_int("INT5_BLOCK_SIZE", 64)
    blocks = max((cols + block_size - 1) // block_size, 1)
    padded_cols = blocks * block_size
    return ((rows * padded_cols * 5) + 7) // 8 + 2 * rows * blocks


def _int8_matrix_bytes(rows: int, cols: int) -> int:
    return rows * cols + 2 * rows


def _estimate_profile_base_model_bytes(profile: str) -> int:
    target_categories = {
        piece.strip()
        for piece in os.environ.get("INT5_LORA_TARGET_CATEGORIES", "attn,mlp,embed,other").split(",")
        if piece.strip()
    }
    min_numel = _env_int("INT5_LORA_MIN_NUMEL", 65536)
    vocab_size = _env_int("VOCAB_SIZE", WINNER_DEFAULTS["VOCAB_SIZE"])
    num_layers = int(_profile_value(profile, "NUM_LAYERS"))
    num_heads = int(_profile_value(profile, "NUM_HEADS"))
    num_kv_heads = int(_profile_value(profile, "NUM_KV_HEADS"))
    model_dim = int(_profile_value(profile, "MODEL_DIM"))
    mlp_mult = float(_profile_value(profile, "MLP_MULT"))
    bigram_vocab = int(_profile_value(profile, "BIGRAM_VOCAB_SIZE"))
    bigram_dim = int(_profile_value(profile, "BIGRAM_DIM"))
    ve_enabled = _env_int("VE_ENABLED", WINNER_DEFAULTS["VE_ENABLED"]) != 0
    ve_dim = int(_profile_value(profile, "VE_DIM"))
    ve_layers = os.environ.get("VE_LAYERS", WINNER_DEFAULTS["VE_LAYERS"])
    tie_embeddings = _env_int("TIE_EMBEDDINGS", WINNER_DEFAULTS["TIE_EMBEDDINGS"]) != 0
    kv_dim = num_kv_heads * (model_dim // num_heads)
    mlp_dim = int(mlp_mult * model_dim)
    num_encoder_layers = num_layers // 2
    num_decoder_layers = num_layers - num_encoder_layers
    num_skip_weights = min(num_encoder_layers, num_decoder_layers)
    ve_layer_count = len([x for x in ve_layers.split(",") if x.strip()]) if ve_enabled else 0
    estimate = 0

    def add_matrix(rows: int, cols: int, category: str) -> None:
        nonlocal estimate
        numel = rows * cols
        if numel <= min_numel:
            estimate += 2 * numel
        elif category in target_categories:
            estimate += _int5_matrix_bytes(rows, cols)
        else:
            estimate += _int8_matrix_bytes(rows, cols)

    add_matrix(vocab_size, model_dim, "embed")
    if not tie_embeddings:
        add_matrix(vocab_size, model_dim, "embed")
    for _ in range(2 * num_layers):
        add_matrix(model_dim, model_dim, "attn")
    for _ in range(2 * num_layers):
        add_matrix(kv_dim, model_dim, "attn")
    for _ in range(num_layers):
        add_matrix(mlp_dim, model_dim, "mlp")
        add_matrix(model_dim, mlp_dim, "mlp")
    if bigram_vocab > 0:
        add_matrix(bigram_vocab, bigram_dim, "other")
        if bigram_dim != model_dim:
            add_matrix(model_dim, bigram_dim, "attn")
        estimate += 2
    if ve_layer_count > 0:
        add_matrix(vocab_size, ve_dim, "other")
        if ve_dim != kv_dim:
            add_matrix(kv_dim, ve_dim, "attn")
        estimate += 2 * ve_layer_count
        estimate += 2

    estimate += 2 * model_dim  # smear.gate fp16
    estimate += 4 * num_skip_weights * model_dim  # skip_weights control fp32
    estimate += num_layers * (
        4 * num_heads +  # q_gain
        4 * model_dim +  # attn_scale
        4 * model_dim +  # mlp_scale
        8 * model_dim    # resid_mix (2 x dim)
    )
    estimate += 4096  # small metadata cushion
    return estimate


def _select_auto_space_profile() -> str:
    budget = _submission_budget_bytes()
    min_headroom = _env_int("INT5_LORA_AUTO_MIN_HEADROOM_BYTES", 524_288)
    estimates = {profile: _estimate_profile_base_model_bytes(profile) for profile in PROFILE_ORDER}
    roomy = [p for p in PROFILE_ORDER if estimates[p] <= budget - min_headroom]
    if roomy:
        return roomy[0]
    fitting = [p for p in PROFILE_ORDER if estimates[p] <= budget]
    if fitting:
        return fitting[0]
    raise ValueError(
        "no auto space profile fits the contest budget; "
        f"budget={budget} estimates={estimates}"
    )


def _apply_space_profile() -> None:
    profile = os.environ.get("LORA_SPACE_PROFILE", "auto").strip().lower()
    if profile not in SPACE_PROFILES:
        raise ValueError(
            f"unknown LORA_SPACE_PROFILE={profile!r}; expected one of {sorted(SPACE_PROFILES)}"
        )
    if profile == "auto":
        profile = _select_auto_space_profile()
        if os.environ.get("RANK", "0") == "0":
            budget = _submission_budget_bytes()
            min_headroom = _env_int("INT5_LORA_AUTO_MIN_HEADROOM_BYTES", 524_288)
            estimate = _estimate_profile_base_model_bytes(profile)
            print(
                f"auto_space_profile:{profile} "
                f"estimated_base_model_bytes:{estimate} "
                f"model_budget_bytes:{budget} "
                f"target_headroom_bytes:{min_headroom}"
            )
    for key, value in SPACE_PROFILES[profile].items():
        os.environ.setdefault(key, value)


def _patched_source() -> str:
    source = BASE_RECORD.read_text(encoding="utf-8")
    start = source.index(QUANT_SECTION_HEADER)
    end = source.index(TRAIN_SECTION_HEADER)
    patched = source[:start] + QUANT_REPLACEMENT + source[end:]
    patched = patched.replace(CODE_REPLACEMENT_OLD, CODE_REPLACEMENT_NEW)
    patched = patched.replace(CASTED_LINEAR_OLD, CASTED_LINEAR_NEW)
    patched = patched.replace(QUANT_EXPORT_OLD, QUANT_EXPORT_NEW)
    patched = patched.replace("x = self.tok_emb(input_ids)", "x = _qat_embedding(self.tok_emb, input_ids, self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))")
    patched = patched.replace("h = self.embed(self.bigram_hash(token_ids))", "h = _qat_embedding(self.embed, self.bigram_hash(token_ids), self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))")
    patched = patched.replace("h = self.embed(token_ids)", "h = _qat_embedding(self.embed, token_ids, self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\"))))")
    patched = patched.replace(
        "F.linear(x_flat, self.tok_emb.weight)",
        "F.linear(x_flat, _qat_weight(self.tok_emb.weight, self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\")))).to(x_flat.dtype))",
    )
    patched = patched.replace(
        "F.linear(x, self.tok_emb.weight)",
        "F.linear(x, _qat_weight(self.tok_emb.weight, self.training and CastedLinear._qat_enabled and bool(int(os.environ.get(\"INT5_TRAIN_QAT\", \"0\")))).to(x.dtype))",
    )
    for old, new in BANK_QAT_REPLACEMENTS:
        patched = patched.replace(old, new)
    for old, new in STRING_REPLACEMENTS:
        patched = patched.replace(old, new)
    return patched


def main() -> None:
    for key, value in PREV_BEST_DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
    _apply_space_profile()
    os.environ["INT5_LORA_CODE_PATHS"] = os.pathsep.join(
        [
            str(Path(__file__).resolve()),
            str(BASE_RECORD),
        ]
    )
    source = _patched_source()
    namespace = {
        "__file__": __file__,
        "__name__": "__main__",
    }
    exec(compile(source, __file__, "exec"), namespace)


if __name__ == "__main__":
    main()
