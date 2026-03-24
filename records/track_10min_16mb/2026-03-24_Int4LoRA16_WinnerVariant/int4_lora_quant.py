from __future__ import annotations

import io
import lzma
import os
from pathlib import Path

import torch
from torch import Tensor

_CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)


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


def _int4_clip_pcts() -> list[float]:
    raw = os.environ.get("INT4_LORA_CLIP_PCTS", "0.999,0.9995,0.9999,1.0")
    pcts = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            pcts.append(float(piece))
    return pcts or [1.0]


def _quantize_int4_per_row(t: Tensor, clip_range: int = 7) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(t32.shape)}")
    best_q = None
    best_s = None
    best_err = float("inf")
    for pct in _int4_clip_pcts():
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        scale = (row_clip / clip_range).clamp_min(1.0 / clip_range)
        q = torch.clamp(torch.round(t32 / scale[:, None]), -8, 7).to(torch.int8)
        recon = q.float() * scale[:, None]
        err = (t32 - recon).pow(2).mean().item()
        if err < best_err:
            best_q = q
            best_s = scale.to(torch.float16)
            best_err = err
    return best_q, best_s


def _pack_int4(q: Tensor) -> Tensor:
    rows, cols = q.shape
    q_u = (q.to(torch.int16) + 8).to(torch.uint8)
    if cols % 2:
        q_u = torch.cat([q_u, torch.zeros(rows, 1, dtype=torch.uint8)], dim=1)
    lo = q_u[:, 0::2]
    hi = q_u[:, 1::2] << 4
    return (lo | hi).contiguous()


def _unpack_int4(packed: Tensor, rows: int, cols: int) -> Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    q = torch.empty(rows, packed.shape[1] * 2, dtype=torch.int16)
    q[:, 0::2] = lo.to(torch.int16)
    q[:, 1::2] = hi.to(torch.int16)
    return (q[:, :cols] - 8).to(torch.int8)


def _factorize_residual_lora(residual: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    rank = min(rank, residual.shape[0], residual.shape[1])
    if rank <= 0:
        return (
            torch.empty(residual.shape[0], 0, dtype=torch.float16),
            torch.empty(0, residual.shape[1], dtype=torch.float16),
        )
    with torch.no_grad():
        u, s, v = torch.svd_lowrank(residual.float(), q=rank, niter=2)
        lora_b = (u * s.unsqueeze(0)).to(torch.float16).contiguous()
        lora_a = v.transpose(0, 1).to(torch.float16).contiguous()
    return lora_a, lora_b


def _quantize_matrix_int4_lora(t: Tensor, rank: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    q, scale = _quantize_int4_per_row(t)
    base = q.float() * scale.float()[:, None]
    lora_a, lora_b = _factorize_residual_lora(t.float() - base, rank)
    return _pack_int4(q), scale, lora_a, lora_b


def _quantize_matrix_int4_base(t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    q, scale = _quantize_int4_per_row(t)
    base = q.float() * scale.float()[:, None]
    return _pack_int4(q), scale, base


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


def _code_paths() -> list[Path]:
    paths = [Path(__file__).resolve()]
    extra = os.environ.get("INT4_LORA_CODE_PATHS", "")
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
    total_budget = int(os.environ.get("INT4_LORA_MAX_TOTAL_BYTES", 16_000_000))
    safety = int(os.environ.get("INT4_LORA_SAFETY_BYTES", 16_384))
    code_bytes = sum(path.stat().st_size for path in _code_paths())
    return max(total_budget - code_bytes - safety, 1)


def _payload_size_bytes(result: dict[str, Tensor], meta: dict[str, object]) -> int:
    buf = io.BytesIO()
    torch.save({"w": result, "m": meta}, buf)
    return len(lzma.compress(buf.getvalue(), preset=6))


def _build_result_from_plans(
    passthrough: dict[str, Tensor],
    passthrough_meta: dict[str, object],
    matrix_plans: list[dict[str, object]],
) -> tuple[dict[str, Tensor], dict[str, object]]:
    result = dict(passthrough)
    meta = dict(passthrough_meta)
    for plan in matrix_plans:
        name = plan["name"]
        result[name + ".q4"] = plan["packed_q"]
        result[name + ".scale"] = plan["scale"]
        rank = int(plan["selected_rank"])
        if rank > 0:
            u = plan["u"][:, :rank].contiguous()
            s = plan["s"][:rank]
            v = plan["v"][:, :rank].contiguous()
            result[name + ".lora_b"] = (u * s.unsqueeze(0)).to(torch.float16).contiguous()
            result[name + ".lora_a"] = v.transpose(0, 1).to(torch.float16).contiguous()
        meta[name] = {"type": "int4_lora16", "rank": rank}
    return result, meta


def _selected_component_density(plan: dict[str, object], rank_idx: int) -> float:
    gain = float(plan["gains"][rank_idx])
    cost = float(plan["bytes_per_rank"])
    return gain / max(cost, 1.0)


def mixed_quantize_int4_lora(state_dict: dict[str, Tensor], _unused_categories: set[str]):
    rank = int(os.environ.get("INT4_LORA_RANK", 128))
    min_numel = int(os.environ.get("INT4_LORA_MIN_NUMEL", 65536))
    target_categories = {
        piece.strip()
        for piece in os.environ.get("INT4_LORA_TARGET_CATEGORIES", "attn,mlp,embed,other").split(",")
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
            packed_q, scale, base = _quantize_matrix_int4_base(t)
            plan = {
                "name": name,
                "packed_q": packed_q,
                "scale": scale,
                "selected_rank": 0,
                "bytes_per_rank": 2 * (t.shape[0] + t.shape[1]),
                "gains": [],
            }
            max_rank = min(rank, t.shape[0], t.shape[1])
            if max_rank > 0:
                residual = t.float() - base
                with torch.no_grad():
                    u, s, v = torch.svd_lowrank(residual, q=max_rank, niter=2)
                plan["u"] = u.contiguous()
                plan["s"] = s.to(torch.float32).contiguous()
                plan["v"] = v.contiguous()
                plan["gains"] = s.to(torch.float32).square().tolist()
            else:
                plan["u"] = torch.empty(t.shape[0], 0, dtype=torch.float32)
                plan["s"] = torch.empty(0, dtype=torch.float32)
                plan["v"] = torch.empty(t.shape[1], 0, dtype=torch.float32)
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
    result, final_meta = _build_result_from_plans(passthrough, meta, matrix_plans)
    budget_bytes = _model_budget_bytes()
    if matrix_plans:
        base_size = _payload_size_bytes(result, final_meta)
        if base_size > budget_bytes:
            raise RuntimeError(
                f"int4 base payload already exceeds budget: {base_size} > {budget_bytes}"
            )
        remaining = budget_bytes - base_size
        candidates: list[tuple[float, int, int]] = []
        for plan_idx, plan in enumerate(matrix_plans):
            cost = int(plan["bytes_per_rank"])
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
                raise RuntimeError(
                    f"unable to fit int4 payload within budget: {exact_size} > {budget_bytes}"
                )
            selected.sort()
            _, plan_idx, rank_idx = selected[0]
            plan = matrix_plans[plan_idx]
            plan["selected_rank"] = rank_idx
            result, final_meta = _build_result_from_plans(passthrough, meta, matrix_plans)
            exact_size = _payload_size_bytes(result, final_meta)
        if os.environ.get("RANK", "0") == "0":
            selected_rank_sum = sum(int(plan["selected_rank"]) for plan in matrix_plans)
            selected_matrix_count = sum(1 for plan in matrix_plans if int(plan["selected_rank"]) > 0)
            print(
                "int4_lora_budget:"
                f" budget_bytes={budget_bytes}"
                f" base_size_bytes={base_size}"
                f" final_size_bytes={exact_size}"
                f" used_pct={100.0 * exact_size / max(budget_bytes, 1):.1f}"
                f" selected_matrices={selected_matrix_count}/{len(matrix_plans)}"
                f" total_selected_rank={selected_rank_sum}"
                f" max_rank_cap={rank}"
            )
    return result, final_meta


def dequantize_int4_lora(result: dict[str, Tensor], meta: dict[str, object], template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
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
        if info["type"] == "int4_lora16":
            rows, cols = orig.shape
            q = _unpack_int4(result[name + ".q4"], rows, cols)
            scale = result[name + ".scale"].float()
            t = q.float() * scale[:, None]
            rank = int(info.get("rank", 0))
            if rank > 0:
                lora_a = result[name + ".lora_a"].float()
                lora_b = result[name + ".lora_b"].float()
                t = t + lora_b @ lora_a
            out[name] = t.to(orig_dtype)
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out
