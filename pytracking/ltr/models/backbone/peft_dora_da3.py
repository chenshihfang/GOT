# peft_dora_vggt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Sequence
import re

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model


# ---------------------------
# helpers
# ---------------------------
def _is_depthwise_or_grouped(m: nn.Conv2d) -> bool:
    groups = getattr(m, "groups", 1)
    return (groups != 1) or (m.in_channels == groups)


def _is_adapter_param_name(n: str) -> bool:
    # PEFT DoRA uses lora_A/lora_B + lora_magnitude_vector (name contains lora_)
    return ("lora_" in n) or ("dora_" in n) or ("magnitude" in n)


def _in_camera(n: str) -> bool:
    return (".cam_enc." in n) or (".cam_dec." in n)


def _parse_block_id_from_param(n: str) -> Optional[int]:
    m = re.search(r"\.backbone\.pretrained\.blocks\.(\d+)\.", n)
    if not m:
        return None
    return int(m.group(1))


def _is_backbone_param(n: str) -> bool:
    return ".backbone.pretrained.blocks." in n


def _is_head_param(n: str) -> bool:
    # In practice names look like "...head.scratch...."
    return ".head." in n


def _is_attn_param(n: str) -> bool:
    return ".attn." in n


def _is_qkvproj_param(n: str) -> bool:
    return (".qkv." in n) or (".proj." in n)


# ---------------------------
# discovery
# ---------------------------
@dataclass
class _TL:
    name: str
    kind: str                 # "linear" | "conv2d"
    shape: Tuple[int, ...]    # weight shape


def _collect_targets(
    root: nn.Module,
    *,
    include_linear: bool,
    include_conv2d: bool,
    skip_depthwise_grouped: bool,
    min_inout: int,
    include_name_regex: Optional[str],
    exclude_name_regex: Optional[str],
) -> List[_TL]:
    inc_re = re.compile(include_name_regex) if include_name_regex else None
    exc_re = re.compile(exclude_name_regex) if exclude_name_regex else None

    out: List[_TL] = []
    for name, m in root.named_modules():
        if inc_re and not inc_re.search(name):
            continue
        if exc_re and exc_re.search(name):
            continue

        if include_linear and isinstance(m, nn.Linear) and getattr(m, "weight", None) is not None:
            of, inf = m.weight.shape
            if max(of, inf) >= min_inout:
                out.append(_TL(name=name, kind="linear", shape=(of, inf)))

        if include_conv2d and isinstance(m, nn.Conv2d) and getattr(m, "weight", None) is not None:
            if skip_depthwise_grouped and _is_depthwise_or_grouped(m):
                continue
            oc, ic, kh, kw = m.weight.shape
            if max(oc, ic) >= min_inout:
                out.append(_TL(name=name, kind="conv2d", shape=(oc, ic, kh, kw)))

    # dedup by module name
    seen = set()
    uniq: List[_TL] = []
    for t in out:
        if t.name in seen:
            continue
        seen.add(t.name)
        uniq.append(t)
    return uniq


# ---------------------------
# param counting
# ---------------------------
def _lora_params_linear(in_f: int, out_f: int, r: int) -> int:
    return r * (in_f + out_f)  # A + B


def _lora_params_conv2d(in_c: int, out_c: int, kh: int, kw: int, r: int) -> int:
    # A: (r, in_c, kh, kw), B: (out_c, r, 1, 1)
    return r * (in_c * kh * kw + out_c)


def _dora_extra(kind: str, shape: Tuple[int, ...]) -> int:
    # DoRA adds magnitude vector with length = out_features/out_channels
    if kind == "linear":
        out, _ = shape
        return out
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


# ---------------------------
# trainable control + printing
# ---------------------------
def freeze_all_params(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)


def set_trainable_dora_params(
    m: nn.Module,
    *,
    train_backbone: bool,
    train_head: bool,
    backbone_blocks: Optional[Sequence[int]] = None,   # None => all blocks
    backbone_attn_only: bool = True,
    backbone_qkvproj_only: bool = True,
    head_include_aux: bool = True,
    exclude_camera: bool = True,
    train_other_adapters: bool = False,
) -> int:
    """
    Turn on requires_grad only for adapter params that match requested scopes.
    Returns trainable adapter param count.
    """
    blocks_set = set(int(x) for x in backbone_blocks) if backbone_blocks is not None else None

    ntrain = 0
    for n, p in m.named_parameters():
        if not _is_adapter_param_name(n):
            p.requires_grad_(False)
            continue

        if exclude_camera and _in_camera(n):
            p.requires_grad_(False)
            continue

        is_bb = _is_backbone_param(n)
        is_hd = _is_head_param(n)

        ok = False

        # backbone adapters
        if is_bb and train_backbone:
            bid = _parse_block_id_from_param(n)
            if (blocks_set is None) or (bid is not None and bid in blocks_set):
                ok = True
                if backbone_attn_only:
                    ok = ok and _is_attn_param(n)
                if backbone_qkvproj_only:
                    ok = ok and _is_qkvproj_param(n)

        # head adapters
        if is_hd and train_head:
            ok = True
            if not head_include_aux and ("_aux" in n):
                ok = False

        # other adapters (rare, but you can allow)
        if (not is_bb) and (not is_hd) and train_other_adapters:
            ok = True

        p.requires_grad_(ok)
        if ok:
            ntrain += p.numel()

    return ntrain


def collect_adapter_params(m: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for n, p in m.named_parameters():
        if p.requires_grad and _is_adapter_param_name(n):
            params.append(p)
    return params


def print_trainable_dora_params(m: nn.Module) -> None:
    items = []
    total = 0
    for n, p in m.named_parameters():
        if p.requires_grad and _is_adapter_param_name(n):
            items.append((n, p.numel(), tuple(p.shape)))
            total += p.numel()

    print(f"[DoRA] trainable tensors: {len(items)}")
    print(f"[DoRA] trainable params : {total:,}")
    for n, numel, shape in sorted(items, key=lambda x: x[0]):
        print(f"  {n:140s}  numel={numel:10d}  shape={shape}")


# ---------------------------
# main API: inject DoRA into backbone and or head
# ---------------------------
@dataclass
class DoraDA3Args:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    min_inout: int = 16
    skip_depthwise_grouped_convs: bool = True


def inject_dora_into_da3(
    da3: nn.Module,
    *,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    # what to inject
    inject_backbone: bool = True,
    inject_head: bool = True,
    backbone_blocks: Optional[Sequence[int]] = None,   # None => all
    backbone_attn_only: bool = True,
    backbone_qkvproj_only: bool = True,
    backbone_include_linear: bool = True,
    backbone_include_conv2d: bool = False,
    head_include_linear: bool = False,
    head_include_conv2d: bool = True,
    head_include_aux: bool = True,
    # general filters
    exclude_camera: bool = True,
    min_inout: int = 16,
    skip_depthwise_grouped_convs: bool = True,
    # after wrap
    freeze_base: bool = True,
    # after wrap: what to train
    train_backbone: bool = True,
    train_head: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """
    1) Discover target modules.
    2) Wrap with PEFT DoRA (use_dora=True).
    3) Freeze base and enable only requested adapter params.
    4) Print trainable adapter param names.
    """
    args = DoraDA3Args(
        r=r,
        alpha=alpha,
        dropout=dropout,
        min_inout=min_inout,
        skip_depthwise_grouped_convs=skip_depthwise_grouped_convs,
    )

    targets: List[_TL] = []
    blocks_set = set(int(x) for x in backbone_blocks) if backbone_blocks is not None else None

    # build include/exclude regex
    # exclude camera modules by name
    exc_cam = r"(?:^|.*\.)cam_enc\.|(?:^|.*\.)cam_dec\." if exclude_camera else None

    # (A) backbone targets
    if inject_backbone:
        # include only backbone blocks, optionally restrict block ids, optionally restrict attn and qkv/proj
        if blocks_set is None:
            blk_part = r"\d+"
        else:
            blk_part = "(?:" + "|".join(str(b) for b in sorted(blocks_set)) + ")"

        inc = rf"(?:^|.*\.)backbone\.pretrained\.blocks\.{blk_part}\."
        if backbone_attn_only:
            inc += r"attn\."
        if backbone_qkvproj_only:
            inc += r"(?:qkv|proj)$"

        targets += _collect_targets(
            da3,
            include_linear=backbone_include_linear,
            include_conv2d=backbone_include_conv2d,
            skip_depthwise_grouped=args.skip_depthwise_grouped_convs,
            min_inout=args.min_inout,
            include_name_regex=inc,
            exclude_name_regex=exc_cam,
        )

    # (B) head targets
    if inject_head:
        inc = r"(?:^|.*\.)head\."
        exc = exc_cam
        if not head_include_aux:
            # filter modules with "_aux" in name
            exc = (exc + r"|_aux") if exc else r"_aux"

        targets += _collect_targets(
            da3,
            include_linear=head_include_linear,
            include_conv2d=head_include_conv2d,
            skip_depthwise_grouped=args.skip_depthwise_grouped_convs,
            min_inout=args.min_inout,
            include_name_regex=inc,
            exclude_name_regex=exc,
        )

    if not targets:
        raise RuntimeError("No eligible Linear or Conv2d layers discovered for DoRA. Check regex and flags.")

    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=[t.name for t in targets],
        use_dora=True,
    )
    wrapped = get_peft_model(da3, cfg)

    # Freeze base and enable requested adapters
    if freeze_base:
        freeze_all_params(wrapped)
        ntrain = set_trainable_dora_params(
            wrapped,
            train_backbone=train_backbone,
            train_head=train_head,
            backbone_blocks=backbone_blocks,
            backbone_attn_only=backbone_attn_only,
            backbone_qkvproj_only=backbone_qkvproj_only,
            head_include_aux=head_include_aux,
            exclude_camera=exclude_camera,
            train_other_adapters=False,
        )
    else:
        ntrain = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)

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

        # print_trainable_dora_params(wrapped)

    return wrapped



