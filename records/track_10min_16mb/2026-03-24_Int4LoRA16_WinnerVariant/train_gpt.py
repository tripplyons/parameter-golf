from __future__ import annotations

import os
from pathlib import Path


BASE_RECORD = (
    Path(__file__).resolve().parent.parent
    / "2026-03-23_LeakyReLU_LegalTTT_ParallelMuon"
    / "train_gpt.py"
)
QUANT_HELPER = Path(__file__).resolve().with_name("int4_lora_quant.py")

QUANT_SECTION_HEADER = "# --- GPTQ-lite int6 quantization ---"
TRAIN_SECTION_HEADER = "# --- Training ---"
QUANT_REPLACEMENT = """# --- Int4 + fp16 LoRA quantization ---
from int4_lora_quant import (
    _classify_param,
    _rebank_state_dict,
    _unbank_state_dict,
    dequantize_int4_lora as dequantize_mixed_int6,
    mixed_quantize_int4_lora as mixed_quantize_int6,
)

"""

STRING_REPLACEMENTS = (
    ("final_model.int6.ptz", "final_model.int4_lora16.ptz"),
    ("Serialized model int6+lzma", "Serialized model int4+lora16+lzma"),
    ("Total submission size int6+lzma", "Total submission size int4+lora16+lzma"),
    ("final_int6_roundtrip", "final_int4_lora16_roundtrip"),
    ("final_int6_sliding_window", "final_int4_lora16_sliding_window"),
    ("final_int8_zlib_roundtrip_exact", "final_int4_lora16_roundtrip_exact"),
)
CODE_REPLACEMENT_OLD = '    code = Path(__file__).read_text(encoding="utf-8")'
CODE_REPLACEMENT_NEW = f"""    code = Path(__file__).read_text(encoding="utf-8")
    base_record_path = Path(r"{BASE_RECORD}")
    if base_record_path.exists():
        code += "\\n\\n# --- base_record_train_gpt.py ---\\n\\n" + base_record_path.read_text(encoding="utf-8")
    quant_path = Path(r"{QUANT_HELPER}")
    if quant_path.exists():
        code += "\\n\\n# --- int4_lora_quant.py ---\\n\\n" + quant_path.read_text(encoding="utf-8")"""

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


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _code_budget_paths() -> list[Path]:
    return [Path(__file__).resolve(), BASE_RECORD, QUANT_HELPER]


def _submission_budget_bytes() -> int:
    total_budget = _env_int("INT4_LORA_MAX_TOTAL_BYTES", 16_000_000)
    safety = _env_int("INT4_LORA_SAFETY_BYTES", 16_384)
    code_bytes = sum(path.stat().st_size for path in _code_budget_paths() if path.exists())
    return max(total_budget - code_bytes - safety, 1)


def _profile_value(profile: str, key: str) -> str:
    if key in os.environ:
        return os.environ[key]
    if key in SPACE_PROFILES[profile]:
        return SPACE_PROFILES[profile][key]
    return str(WINNER_DEFAULTS[key])


def _int4_matrix_bytes(rows: int, cols: int) -> int:
    return rows * ((cols + 1) // 2) + 2 * rows


def _int8_matrix_bytes(rows: int, cols: int) -> int:
    return rows * cols + 2 * rows


def _estimate_profile_base_model_bytes(profile: str) -> int:
    target_categories = {
        piece.strip()
        for piece in os.environ.get("INT4_LORA_TARGET_CATEGORIES", "attn,mlp,embed,other").split(",")
        if piece.strip()
    }
    min_numel = _env_int("INT4_LORA_MIN_NUMEL", 65536)
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
            estimate += _int4_matrix_bytes(rows, cols)
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
    min_headroom = _env_int("INT4_LORA_AUTO_MIN_HEADROOM_BYTES", 524_288)
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
            min_headroom = _env_int("INT4_LORA_AUTO_MIN_HEADROOM_BYTES", 524_288)
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
    for old, new in STRING_REPLACEMENTS:
        patched = patched.replace(old, new)
    return patched


def main() -> None:
    _apply_space_profile()
    os.environ["INT4_LORA_CODE_PATHS"] = os.pathsep.join(
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
