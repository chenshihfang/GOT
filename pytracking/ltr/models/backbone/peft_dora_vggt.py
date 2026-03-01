# peft_dora_vggt.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

# ---------- discovery ----------
def _is_depthwise_or_grouped(m: nn.Conv2d) -> bool:
    return getattr(m, "groups", 1) != 1 or m.in_channels == m.groups

@dataclass
class _TL:
    name: str
    kind: str                 # "linear" | "conv2d"
    shape: Tuple[int, ...]    # weight shape

def _collect_targets(
    root: nn.Module,
    include_linear: bool = True,
    include_conv2d: bool = False,
    skip_depthwise_grouped: bool = True,
    min_inout: int = 16,
    include_name_regex: str | None = None,
    exclude_name_regex: str | None = None,
    attn_only: bool = False,
) -> List[_TL]:
    import re
    inc_re = re.compile(include_name_regex) if include_name_regex else None
    exc_re = re.compile(exclude_name_regex) if exclude_name_regex else None

    out: List[_TL] = []
    for name, m in root.named_modules():
        # name filtering
        if inc_re and not inc_re.search(name):
            continue
        if exc_re and exc_re.search(name):
            continue
        if attn_only and (".attn." not in name):
            continue

        if include_linear and isinstance(m, nn.Linear) and (getattr(m, "weight", None) is not None):
            of, inf = m.weight.shape
            if max(of, inf) >= min_inout:
                out.append(_TL(name, "linear", (of, inf)))
        elif include_conv2d and isinstance(m, nn.Conv2d) and (getattr(m, "weight", None) is not None):
            if skip_depthwise_grouped and _is_depthwise_or_grouped(m):
                continue
            oc, ic, kh, kw = m.weight.shape
            if max(oc, ic) >= min_inout:
                out.append(_TL(name, "conv2d", (oc, ic, kh, kw)))
    return out

# ---------- param counting ----------
def _lora_params_linear(in_f: int, out_f: int, r: int) -> int:
    return r * (in_f + out_f)

def _lora_params_conv2d(in_c: int, out_c: int, kh: int, kw: int, r: int) -> int:
    return r * (in_c * kh * kw + out_c)

def _dora_extra(kind: str, shape: Tuple[int, ...]) -> int:
    if kind == "linear":
        out, _ = shape
        return out
    else:
        out, *_ = shape
        return out

def _count_trainables(tl: Iterable[_TL], r: int, use_dora: bool) -> int:
    n = 0
    for t in tl:
        if t.kind == "linear":
            of, inf = t.shape
            n += _lora_params_linear(inf, of, r)
        else:
            oc, ic, kh, kw = t.shape
            n += _lora_params_conv2d(ic, oc, kh, kw, r)
        if use_dora:
            n += _dora_extra(t.kind, t.shape)
    return n

# ---------- freezing ----------
def _freeze_all(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False

def _unfreeze_adapters_only(
    m: nn.Module,
    scope_regex: str | None = None,
    attn_only: bool = False,
) -> int:
    import re
    scope_re = re.compile(scope_regex) if scope_regex else None
    cnt = 0
    for n, p in m.named_parameters():
        is_adapter = ("lora_" in n) or ("dora_" in n)
        in_scope = (scope_re.search(n) is not None) if scope_re else True
        in_attn  = (".attn." in n) if attn_only else True
        if is_adapter and in_scope and in_attn:
            p.requires_grad = True
            cnt += p.numel()
        else:
            p.requires_grad = False
    return cnt

def collect_adapter_params(m: nn.Module):
    """Return a list of adapter parameters (requires_grad==True and name has lora_/dora_)."""
    params = []
    for n, p in m.named_parameters():
        if p.requires_grad and (("lora_" in n) or ("dora_" in n)):
            params.append(p)
    return params

# ---------- main API ----------
@dataclass
class DoraVGGTArgs:
    r: int = 64
    alpha: int = 64      # DoRA: alpha ≈ r
    dropout: float = 0.0
    include_linear: bool = True
    include_conv2d: bool = False
    skip_depthwise_grouped_convs: bool = True
    min_inout: int = 16

def inject_dora_into_backbone(
    backbone: nn.Module,
    r: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
    freeze_base: bool = True,
    verbose: bool = True,
    # scoping knobs
    include_name_regex: str | None = None,      # e.g., r"\.aggregator\.global_blocks\."
    exclude_name_regex: str | None = None,      # e.g., r"\.mlp\."
    attn_only: bool = False,                    # keep only qkv/proj under .attn.
) -> nn.Module:
    # 1) select targets
    args = DoraVGGTArgs(r=r, alpha=alpha, dropout=dropout)
    targets = _collect_targets(
        backbone,
        include_linear=args.include_linear,
        include_conv2d=args.include_conv2d,
        skip_depthwise_grouped=args.skip_depthwise_grouped_convs,
        min_inout=args.min_inout,
        include_name_regex=include_name_regex,
        exclude_name_regex=exclude_name_regex,
        attn_only=attn_only,
    )
    if not targets:
        raise RuntimeError("No eligible Linear/Conv2d layers discovered for DoRA (after scoping).")

    # 2) build DoRA config + wrap
    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=[t.name for t in targets],
        use_dora=True,
    )
    wrapped = get_peft_model(backbone, cfg)

    # 3) freeze/unfreeze
    if freeze_base:
        _freeze_all(wrapped)
        ntrain = _unfreeze_adapters_only(
            wrapped,
            scope_regex=include_name_regex,
            attn_only=attn_only,
        )
    else:
        ntrain = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)

    # 4) reporting
    if verbose:
        try:
            wrapped.print_trainable_parameters()
        except Exception:
            pass
        lora_n = _count_trainables(targets, args.r, use_dora=False)
        dora_n = _count_trainables(targets, args.r, use_dora=True)
        print(f"[LoRA math] trainable params (r={args.r}) = {lora_n:,}")
        print(f"[DoRA  math] trainable params (r={args.r}) = {dora_n:,}  (+out per adapted layer)")
        print(f"[adapters] requires_grad=True total = {ntrain:,}")
    return wrapped
