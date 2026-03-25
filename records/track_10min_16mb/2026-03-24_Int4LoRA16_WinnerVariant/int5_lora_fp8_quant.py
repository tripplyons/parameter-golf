from __future__ import annotations

import io
import lzma
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

_CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def _unbank_state_dict(sd: dict[str, Tensor], num_layers: int) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out


def _rebank_state_dict(sd: dict[str, Tensor], num_layers: int, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    n = num_layers
    qo_slices = [None] * (2 * n)
    kv_slices = [None] * (2 * n)
    up_slices = [None] * n
    down_slices = [None] * n
    consumed = set()
    for i in range(n):
        qk = f"blocks.{i}.attn.c_q.weight"
        if qk in sd:
            qo_slices[i] = sd[qk]
            consumed.add(qk)
        ok = f"blocks.{i}.attn.proj.weight"
        if ok in sd:
            qo_slices[n + i] = sd[ok]
            consumed.add(ok)
        kk = f"blocks.{i}.attn.c_k.weight"
        if kk in sd:
            kv_slices[i] = sd[kk]
            consumed.add(kk)
        vk = f"blocks.{i}.attn.c_v.weight"
        if vk in sd:
            kv_slices[n + i] = sd[vk]
            consumed.add(vk)
        fk = f"blocks.{i}.mlp.fc.weight"
        if fk in sd:
            up_slices[i] = sd[fk]
            consumed.add(fk)
        dk = f"blocks.{i}.mlp.proj.weight"
        if dk in sd:
            down_slices[i] = sd[dk]
            consumed.add(dk)
    out["qo_bank"] = torch.stack(qo_slices).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv_slices).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up_slices).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down_slices).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out


def _int5_block_size() -> int:
    return _int_env("INT5_BLOCK_SIZE", 64)


def _fp8_block_size() -> int:
    return _int_env("FP8_LORA_BLOCK_SIZE", _int5_block_size())


def _lora_factor_max_matrices() -> int:
    return _int_env("INT5_LORA_MAX_FACTORIZED_MATRICES", 12)


def _lora_svd_rank() -> int:
    return _int_env("INT5_LORA_SVD_MAX_RANK", 48)


def _lora_svd_niter() -> int:
    return _int_env("INT5_LORA_SVD_NITER", 1)


def _lora_svd_device() -> torch.device:
    device_name = os.environ.get("INT5_LORA_SVD_DEVICE", "auto").strip().lower()
    if device_name in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _pad_last_dim(t: Tensor, block_size: int) -> tuple[Tensor, int]:
    cols = t.shape[-1]
    blocks = max((cols + block_size - 1) // block_size, 1)
    padded_cols = blocks * block_size
    if padded_cols == cols:
        return t.contiguous(), padded_cols
    pad = torch.zeros(*t.shape[:-1], padded_cols - cols, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=-1).contiguous(), padded_cols


def _quantize_int5_blockwise(t: Tensor, block_size: int | None = None) -> tuple[Tensor, Tensor]:
    if t.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(t.shape)}")
    bs = block_size or _int5_block_size()
    t32 = t.float().contiguous()
    padded, padded_cols = _pad_last_dim(t32, bs)
    blocks = padded_cols // bs
    view = padded.view(t32.shape[0], blocks, bs)
    clip = view.abs().amax(dim=-1)
    scale = (clip / 15.0).clamp_min(1.0 / 15.0).to(torch.float16).contiguous()
    q = torch.clamp(torch.round(view / scale.float().unsqueeze(-1)), -16, 15).to(torch.int8)
    return q.view(t32.shape[0], padded_cols).contiguous(), scale


def _dequantize_int5_blockwise(q: Tensor, scale: Tensor, cols: int, block_size: int | None = None) -> Tensor:
    bs = block_size or _int5_block_size()
    rows = q.shape[0]
    padded_cols = q.shape[1]
    blocks = padded_cols // bs
    deq = q.float().view(rows, blocks, bs) * scale.float().unsqueeze(-1)
    return deq.view(rows, padded_cols)[:, :cols].contiguous()


def _pack_int5(q: Tensor) -> Tensor:
    flat = (q.to(torch.int16).reshape(-1) + 16).to(torch.uint8)
    n = flat.numel()
    groups = (n + 7) // 8
    if groups * 8 != n:
        flat = torch.cat([flat, torch.zeros(groups * 8 - n, dtype=torch.uint8)], dim=0)
    vals = flat.view(groups, 8).to(torch.int16)
    packed = torch.empty(groups * 5, dtype=torch.uint8)
    packed[0::5] = (vals[:, 0] | ((vals[:, 1] & 0x07) << 5)).to(torch.uint8)
    packed[1::5] = (((vals[:, 1] >> 3) & 0x03) | ((vals[:, 2] & 0x1F) << 2) | ((vals[:, 3] & 0x01) << 7)).to(torch.uint8)
    packed[2::5] = (((vals[:, 3] >> 1) & 0x0F) | ((vals[:, 4] & 0x0F) << 4)).to(torch.uint8)
    packed[3::5] = (((vals[:, 4] >> 4) & 0x01) | ((vals[:, 5] & 0x1F) << 1) | ((vals[:, 6] & 0x03) << 6)).to(torch.uint8)
    packed[4::5] = (((vals[:, 6] >> 2) & 0x07) | ((vals[:, 7] & 0x1F) << 3)).to(torch.uint8)
    return packed.contiguous()


def _unpack_int5(packed: Tensor, rows: int, padded_cols: int) -> Tensor:
    count = rows * padded_cols
    groups = (count + 7) // 8
    data = packed[: groups * 5].view(groups, 5).to(torch.int16)
    vals = torch.empty(groups, 8, dtype=torch.int16)
    vals[:, 0] = data[:, 0] & 0x1F
    vals[:, 1] = ((data[:, 0] >> 5) & 0x07) | ((data[:, 1] & 0x03) << 3)
    vals[:, 2] = (data[:, 1] >> 2) & 0x1F
    vals[:, 3] = ((data[:, 1] >> 7) & 0x01) | ((data[:, 2] & 0x0F) << 1)
    vals[:, 4] = ((data[:, 2] >> 4) & 0x0F) | ((data[:, 3] & 0x01) << 4)
    vals[:, 5] = (data[:, 3] >> 1) & 0x1F
    vals[:, 6] = ((data[:, 3] >> 6) & 0x03) | ((data[:, 4] & 0x07) << 2)
    vals[:, 7] = (data[:, 4] >> 3) & 0x1F
    return (vals.view(-1)[:count] - 16).to(torch.int8).view(rows, padded_cols).contiguous()


def _fp8_dtype() -> torch.dtype:
    dtype_name = os.environ.get("FP8_LORA_DTYPE", "float8_e4m3fn")
    if not hasattr(torch, dtype_name):
        raise RuntimeError(f"requested FP8 dtype {dtype_name!r} is unavailable in this PyTorch build")
    return getattr(torch, dtype_name)


def _fp8_max_abs() -> float:
    return float(torch.finfo(_fp8_dtype()).max)


def _quantize_rows_fp8_blockwise(t: Tensor, block_size: int | None = None) -> tuple[Tensor, Tensor]:
    if t.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(t.shape)}")
    bs = block_size or _fp8_block_size()
    t32 = t.float().contiguous()
    padded, padded_cols = _pad_last_dim(t32, bs)
    blocks = padded_cols // bs
    view = padded.view(t32.shape[0], blocks, bs)
    max_abs = _fp8_max_abs()
    clip = view.abs().amax(dim=-1)
    scale = (clip / max_abs).clamp_min(1.0 / max_abs).to(torch.float16).contiguous()
    fp8 = torch.clamp(view / scale.float().unsqueeze(-1), -max_abs, max_abs).to(_fp8_dtype())
    return fp8.view(t32.shape[0], padded_cols).contiguous(), scale


def _dequantize_rows_fp8_blockwise(q: Tensor, scale: Tensor, cols: int, block_size: int | None = None) -> Tensor:
    bs = block_size or _fp8_block_size()
    padded_cols = q.shape[1]
    blocks = padded_cols // bs
    deq = q.float().view(q.shape[0], blocks, bs) * scale.float().unsqueeze(-1)
    return deq.view(q.shape[0], padded_cols)[:, :cols].contiguous()


def _code_paths() -> list[Path]:
    paths = [Path(__file__).resolve()]
    extra = os.environ.get("INT5_LORA_CODE_PATHS", "")
    for piece in extra.split(os.pathsep):
        piece = piece.strip()
        if piece:
            paths.append(Path(piece).resolve())
    deduped: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen and path.exists():
            deduped.append(path)
            seen.add(key)
    return deduped


def _model_budget_bytes() -> int:
    total_budget = _int_env("INT5_LORA_MAX_TOTAL_BYTES", 16_000_000)
    safety = _int_env("INT5_LORA_SAFETY_BYTES", 16_384)
    code_bytes = sum(path.stat().st_size for path in _code_paths())
    return max(total_budget - code_bytes - safety, 1)


def _payload_size_bytes(result: dict[str, Tensor], meta: dict[str, object]) -> int:
    buf = io.BytesIO()
    torch.save({"w": result, "m": meta}, buf)
    return len(lzma.compress(buf.getvalue(), preset=6))


def _rowwise_fp8_rank_bytes(length: int, block_size: int) -> int:
    blocks = max((length + block_size - 1) // block_size, 1)
    padded = blocks * block_size
    return padded + 2 * blocks


def _build_result_from_plans(
    passthrough: dict[str, Tensor],
    passthrough_meta: dict[str, object],
    matrix_plans: list[dict[str, object]],
) -> tuple[dict[str, Tensor], dict[str, object]]:
    result = dict(passthrough)
    meta = dict(passthrough_meta)
    for plan in matrix_plans:
        name = plan["name"]
        result[name + ".q5"] = plan["packed_q"]
        result[name + ".scale"] = plan["scale"]
        rank = int(plan["selected_rank"])
        if rank > 0:
            result[name + ".lora_b_fp8"] = plan["lora_b_fp8"][:rank].contiguous()
            result[name + ".lora_b_scale"] = plan["lora_b_scale"][:rank].contiguous()
            result[name + ".lora_a_fp8"] = plan["lora_a_fp8"][:rank].contiguous()
            result[name + ".lora_a_scale"] = plan["lora_a_scale"][:rank].contiguous()
        meta[name] = {"type": "int5_block_lora_fp8", "rank": rank}
    return result, meta


def _selected_component_density(plan: dict[str, object], rank_idx: int) -> float:
    gain = float(plan["gains"][rank_idx])
    cost = float(plan["bytes_per_rank"])
    return gain / max(cost, 1.0)


def _quantize_int8_per_row(t: Tensor, clip_range: int = 127) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_clip = t32.abs().amax(dim=1)
        scale = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -clip_range, clip_range).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def mixed_quantize_int5_lora(state_dict: dict[str, Tensor], _unused_categories: set[str]):
    rank = _int_env("INT5_LORA_RANK", 128)
    min_numel = _int_env("INT5_LORA_MIN_NUMEL", 65_536)
    int5_block_size = _int5_block_size()
    fp8_block_size = _fp8_block_size()
    factor_max_matrices = _lora_factor_max_matrices()
    svd_rank_cap = _lora_svd_rank()
    svd_niter = _lora_svd_niter()
    svd_device = _lora_svd_device()
    target_categories = {
        piece.strip()
        for piece in os.environ.get("INT5_LORA_TARGET_CATEGORIES", "attn,mlp,embed,other").split(",")
        if piece.strip()
    }
    passthrough: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    matrix_plans: list[dict[str, object]] = []
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point():
            passthrough[name] = t
            meta[name] = "passthrough"
            continue
        if any(pattern in name for pattern in _CONTROL_TENSOR_NAME_PATTERNS):
            passthrough[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        cat = _classify_param(name)
        if t.ndim == 2 and t.numel() > min_numel and cat in target_categories:
            q_int5, scale = _quantize_int5_blockwise(t, block_size=int5_block_size)
            base = _dequantize_int5_blockwise(q_int5, scale, t.shape[1], block_size=int5_block_size)
            residual_energy = float((t.float() - base).pow(2).sum().item())
            plan = {
                "name": name,
                "packed_q": _pack_int5(q_int5),
                "scale": scale,
                "selected_rank": 0,
                "bytes_per_rank": _rowwise_fp8_rank_bytes(t.shape[0], fp8_block_size) + _rowwise_fp8_rank_bytes(t.shape[1], fp8_block_size),
                "gains": [],
                "factor_density_hint": residual_energy / max(
                    _rowwise_fp8_rank_bytes(t.shape[0], fp8_block_size) + _rowwise_fp8_rank_bytes(t.shape[1], fp8_block_size),
                    1,
                ),
                "tensor": t,
                "base": base,
            }
            plan["lora_b_fp8"] = torch.empty(0, 0, dtype=_fp8_dtype())
            plan["lora_b_scale"] = torch.empty(0, 0, dtype=torch.float16)
            plan["lora_a_fp8"] = torch.empty(0, 0, dtype=_fp8_dtype())
            plan["lora_a_scale"] = torch.empty(0, 0, dtype=torch.float16)
            matrix_plans.append(plan)
            continue
        if t.numel() <= min_numel:
            passthrough[name] = t.to(torch.float16)
            meta[name] = "passthrough_fp16"
            continue
        q, s = _quantize_int8_per_row(t)
        passthrough[name + ".q"] = q
        passthrough[name + ".scale"] = s
        meta[name] = {"type": "int8"}
    if matrix_plans and factor_max_matrices > 0 and rank > 0:
        selected_for_factor = sorted(
            range(len(matrix_plans)),
            key=lambda idx: float(matrix_plans[idx]["factor_density_hint"]),
            reverse=True,
        )[: min(factor_max_matrices, len(matrix_plans))]
        for plan_idx in selected_for_factor:
            plan = matrix_plans[plan_idx]
            t = plan["tensor"]
            base = plan["base"]
            max_rank = min(rank, svd_rank_cap, t.shape[0], t.shape[1])
            if max_rank <= 0:
                continue
            residual = (t.float() - base).to(device=svd_device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad():
                u, s, v = torch.svd_lowrank(residual, q=max_rank, niter=svd_niter)
                lora_b_rows = (u * s.unsqueeze(0)).transpose(0, 1).contiguous()
                lora_a_rows = v.transpose(0, 1).contiguous()
                lora_b_fp8, lora_b_scale = _quantize_rows_fp8_blockwise(lora_b_rows, block_size=fp8_block_size)
                lora_a_fp8, lora_a_scale = _quantize_rows_fp8_blockwise(lora_a_rows, block_size=fp8_block_size)
            plan["lora_b_fp8"] = lora_b_fp8.cpu()
            plan["lora_b_scale"] = lora_b_scale.cpu()
            plan["lora_a_fp8"] = lora_a_fp8.cpu()
            plan["lora_a_scale"] = lora_a_scale.cpu()
            plan["gains"] = s.to(torch.float32).square().cpu().tolist()
            del residual
        if svd_device.type == "cuda":
            torch.cuda.synchronize(svd_device)
            torch.cuda.empty_cache()
    for plan in matrix_plans:
        plan.pop("tensor", None)
        plan.pop("base", None)
    result, final_meta = _build_result_from_plans(passthrough, meta, matrix_plans)
    budget_bytes = _model_budget_bytes()
    if matrix_plans:
        base_size = _payload_size_bytes(result, final_meta)
        if base_size > budget_bytes:
            raise RuntimeError(f"int5 base payload already exceeds budget: {base_size} > {budget_bytes}")
        remaining = budget_bytes - base_size
        candidates: list[tuple[float, int, int]] = []
        for plan_idx, plan in enumerate(matrix_plans):
            for rank_idx in range(len(plan["gains"])):
                density = _selected_component_density(plan, rank_idx)
                candidates.append((density, plan_idx, rank_idx))
        candidates.sort(reverse=True)
        for _, plan_idx, rank_idx in candidates:
            plan = matrix_plans[plan_idx]
            if int(plan["selected_rank"]) != rank_idx:
                continue
            cost = int(plan["bytes_per_rank"])
            if cost <= remaining:
                plan["selected_rank"] = rank_idx + 1
                remaining -= cost
        result, final_meta = _build_result_from_plans(passthrough, meta, matrix_plans)
        exact_size = _payload_size_bytes(result, final_meta)
        while exact_size > budget_bytes:
            selected: list[tuple[float, int, int]] = []
            for plan_idx, plan in enumerate(matrix_plans):
                rank_now = int(plan["selected_rank"])
                if rank_now > 0:
                    rank_idx = rank_now - 1
                    density = _selected_component_density(plan, rank_idx)
                    selected.append((density, plan_idx, rank_idx))
            if not selected:
                raise RuntimeError(f"unable to fit int5 payload within budget: {exact_size} > {budget_bytes}")
            selected.sort()
            _, plan_idx, rank_idx = selected[0]
            matrix_plans[plan_idx]["selected_rank"] = rank_idx
            result, final_meta = _build_result_from_plans(passthrough, meta, matrix_plans)
            exact_size = _payload_size_bytes(result, final_meta)
        if os.environ.get("RANK", "0") == "0":
            selected_rank_sum = sum(int(plan["selected_rank"]) for plan in matrix_plans)
            selected_matrix_count = sum(1 for plan in matrix_plans if int(plan["selected_rank"]) > 0)
            print(
                "int5_lora_fp8_budget:"
                f" budget_bytes={budget_bytes}"
                f" base_size_bytes={base_size}"
                f" final_size_bytes={exact_size}"
                f" used_pct={100.0 * exact_size / max(budget_bytes, 1):.1f}"
                f" selected_matrices={selected_matrix_count}/{len(matrix_plans)}"
                f" total_selected_rank={selected_rank_sum}"
                f" max_rank_cap={rank}"
            )
    return result, final_meta


def dequantize_int5_lora(result: dict[str, Tensor], meta: dict[str, object], template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    int5_block_size = _int5_block_size()
    fp8_block_size = _fp8_block_size()
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        if info["type"] == "int5_block_lora_fp8":
            rows, cols = orig.shape
            bs = int5_block_size
            padded_cols = max((cols + bs - 1) // bs, 1) * bs
            q = _unpack_int5(result[name + ".q5"], rows, padded_cols)
            scale = result[name + ".scale"]
            t = _dequantize_int5_blockwise(q, scale, cols, block_size=bs)
            rank = int(info.get("rank", 0))
            if rank > 0:
                lora_b_rows = _dequantize_rows_fp8_blockwise(
                    result[name + ".lora_b_fp8"],
                    result[name + ".lora_b_scale"],
                    rows,
                    block_size=fp8_block_size,
                )
                lora_a_rows = _dequantize_rows_fp8_blockwise(
                    result[name + ".lora_a_fp8"],
                    result[name + ".lora_a_scale"],
                    cols,
                    block_size=fp8_block_size,
                )
                t = t + lora_b_rows.transpose(0, 1) @ lora_a_rows
            out[name] = t.to(orig_dtype)
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


def _qat_min_numel() -> int:
    return _int_env("INT5_LORA_MIN_NUMEL", 65_536)


def _qat_weight(weight: Tensor, training: bool) -> Tensor:
    if not training or weight.ndim != 2 or weight.numel() <= _qat_min_numel():
        return weight
    q, scale = _quantize_int5_blockwise(weight, block_size=_int5_block_size())
    return _dequantize_int5_blockwise(q, scale, weight.shape[1], block_size=_int5_block_size()).to(weight.dtype)


def _qat_embedding(emb: nn.Embedding, token_ids: Tensor, training: bool) -> Tensor:
    weight = _qat_weight(emb.weight, training)
    return F.embedding(token_ids, weight)
