# profile_tomp_components_PT.py
# Safe FLOPs/MACs profiler for ToMPnet_PT components.
#
# Fixes:
# - No mutation of training net (deepcopy before profiling)
# - Correct device + dtype for dummy inputs (no cuda/cpu mix, no fp16/fp32 mix)
# - Correct input shapes for SideNetwork_D/U
# - Correct cotracker call signature + THOP on CUDA to avoid fvcore segfault/tracing issues
#
# Dependencies:
# - fvcore (optional, used for CPU counting)
# - thop (recommended; required for feature_extractor & cotracker)

from __future__ import annotations

import copy
import contextlib
import gc
import io
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ---------------- optional backends ----------------
_HAS_FVCORE = False
_HAS_THOP = False

try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
    _HAS_FVCORE = True
except Exception:
    _HAS_FVCORE = False

try:
    from thop import profile as thop_profile  # type: ignore
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False


# ---------------- helpers ----------------
def _fmt_units(v: float, unit: str) -> str:
    v = float(v)
    a = abs(v)
    if a >= 1e12:
        return f"{v/1e12:.3f}T {unit}"
    if a >= 1e9:
        return f"{v/1e9:.3f}G {unit}"
    if a >= 1e6:
        return f"{v/1e6:.3f}M {unit}"
    if a >= 1e3:
        return f"{v/1e3:.3f}K {unit}"
    return f"{v:.3f} {unit}"

def _fmt_params(n: int) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n/1_000:.3f}K"
    return str(n)

def _params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

def _first_tensor(out: Any, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        for x in out:
            if torch.is_tensor(x):
                return x
    if isinstance(out, dict):
        for x in out.values():
            if torch.is_tensor(x):
                return x
    return torch.zeros(1, device=like.device, dtype=like.dtype)

def _get_device_dtype(m: nn.Module) -> Tuple[torch.device, torch.dtype]:
    for p in m.parameters(recurse=True):
        return p.device, p.dtype
    for b in m.buffers(recurse=True):
        return b.device, b.dtype
    return torch.device("cpu"), torch.float32

def _cleanup_thop_buffers(m: nn.Module) -> None:
    # prevent total_ops / total_params from polluting state_dict (even on copies)
    for mod in m.modules():
        if hasattr(mod, "_buffers"):
            if "total_ops" in mod._buffers:
                del mod._buffers["total_ops"]
            if "total_params" in mod._buffers:
                del mod._buffers["total_params"]
        if hasattr(mod, "total_ops"):
            try:
                delattr(mod, "total_ops")
            except Exception:
                pass
        if hasattr(mod, "total_params"):
            try:
                delattr(mod, "total_params")
            except Exception:
                pass

@contextlib.contextmanager
def _silence():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ---------------- counting backends ----------------
def _count_fvcore_cpu(module: nn.Module, inputs: Any) -> Dict[str, float]:
    if not _HAS_FVCORE:
        raise RuntimeError("fvcore not installed")
    module.eval()
    with torch.no_grad():
        flops = float(FlopCountAnalysis(module, inputs).total())
    macs = flops / 2.0
    return {"backend": "fvcore", "flops": flops, "macs": macs, "params": float(_params(module))}

def _count_thop(module: nn.Module, inputs: Any) -> Dict[str, float]:
    if not _HAS_THOP:
        raise RuntimeError("thop not installed")
    module.eval()
    with torch.no_grad():
        if isinstance(inputs, tuple):
            macs, params = thop_profile(module, inputs=inputs, verbose=False)
        else:
            macs, params = thop_profile(module, inputs=(inputs,), verbose=False)
    _cleanup_thop_buffers(module)
    flops = float(macs) * 2.0
    return {"backend": "thop", "flops": float(flops), "macs": float(macs), "params": float(params)}


# ---------------- wrappers matching ToMP usage ----------------
class _FeatureExtractorWrap(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m(x)
        return _first_tensor(out, like=x)

class _FilterPredictorWrap(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T,B,C,H,W)
        T, B, C, H, W = x.shape
        train_feat = x
        test_feat = x[:1]
        train_label = torch.zeros(T, B, H, W, dtype=x.dtype, device=x.device)
        train_ltrb_target = torch.zeros(T, B, 4, H, W, dtype=x.dtype, device=x.device)
        try:
            out = self.m(train_feat, test_feat, train_label, train_ltrb_target)
        except TypeError:
            try:
                out = self.m(train_feat, test_feat, train_label)
            except TypeError:
                out = self.m(train_feat, test_feat)
        return _first_tensor(out, like=test_feat.reshape(-1, C, H, W))

class _HeadClsWrap(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nf, ns = 1, 2
        _, C, H, W = x.shape
        feat = x[:nf].repeat(1, ns, 1, 1, 1)
        filt = torch.zeros(ns, C, 1, 1, dtype=x.dtype, device=x.device)
        out = self.m(feat, filt)
        return _first_tensor(out, like=x)

class _HeadBBWrap(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nf, ns = 1, 2
        _, C, H, W = x.shape
        feat = x[:nf].repeat(1, ns, 1, 1, 1)
        filt = torch.zeros(ns, C, 1, 1, dtype=x.dtype, device=x.device)
        out = self.m(feat, filt)
        return _first_tensor(out, like=x)

class _BkMlpWrap(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m(x)
        return _first_tensor(out, like=x)

class _CoTrackerWrap(nn.Module):
    def __init__(self, cot: nn.Module, sd: nn.Module, su: nn.Module, grid_size: int, backward_tracking: bool):
        super().__init__()
        self.cot = cot
        self.sd = sd
        self.su = su
        self.grid_size = int(grid_size)
        self.backward_tracking = bool(backward_tracking)

    def forward(self, video: torch.Tensor, label_first: torch.Tensor, label_last: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        out = self.cot(
            self.sd,
            self.su,
            label_first,
            label_last,
            video,
            queries=queries,
            grid_size=self.grid_size,
            backward_tracking=self.backward_tracking,
        )
        return _first_tensor(out, like=video)


# ---------------- main API ----------------
def profile_tomp_components_PT(
    net: nn.Module,
    *,
    H_img: int,
    W_img: int,
    C_backbone: int = 3,
    C_head: int = 256,
    H_feat: int = 27,
    W_feat: int = 27,
    do_backbone: bool = True,
    point_num: int = 128,
    cot_T: int = 8,
    cot_H: int = 432,
    cot_W: int = 432,
    label_hw: Tuple[int, int] = (96, 128),
    grid_size: int = 100,
    cot_backward_tracking: bool = False,
) -> None:

    net0 = net.module if hasattr(net, "module") else net

    def _maybe(name: str) -> Optional[nn.Module]:
        m = getattr(net0, name, None)
        return m if isinstance(m, nn.Module) else None

    # ---------- Params ----------
    print("[Complexity] Params")
    print(f"net.parameters {_params(net0)} ({_fmt_params(_params(net0))})")
    for k in ["feature_extractor", "head", "bkMlp", "SideNetwork_D", "SideNetwork_U", "PTrackAttentionModel", "cotracker_model"]:
        m = _maybe(k)
        if m is not None:
            print(f"net.{k} {_params(m)} ({_fmt_params(_params(m))})")

    print("\n[Complexity] Component MACs/FLOPs")

    # ---------- feature_extractor (CUDA + THOP) ----------
    if do_backbone and _maybe("feature_extractor") is not None:
        try:
            if not _HAS_THOP:
                print("feature_extractor skipped: THOP not installed.")
            elif not torch.cuda.is_available():
                print("feature_extractor skipped: CUDA not available.")
            else:
                fe_src = _maybe("feature_extractor")
                dev, dt = _get_device_dtype(fe_src)  # should be cuda + half in your run
                fe_copy = copy.deepcopy(fe_src).eval().to(device=dev, dtype=dt)
                fe = _FeatureExtractorWrap(fe_copy).to(device=dev, dtype=dt)

                x = torch.randn(1, C_backbone, H_img, W_img, device=dev, dtype=dt)

                with _silence(), torch.inference_mode():
                    st = _count_thop(fe, x)

                print(f"feature_extractor backend={st['backend']} MACs {_fmt_units(st['macs'],'MACs')} FLOPs {_fmt_units(st['flops'],'FLOPs')}")
        except Exception as e:
            print(f"feature_extractor skipped: {type(e).__name__}: {e}")
    else:
        if not do_backbone:
            print("feature_extractor skipped: disabled (do_backbone=False)")

    # ---------- head.* (profile on SAME device/dtype as head weights, THOP) ----------
    # Your head is on CUDA (based on your errors). So use CUDA+THOP to avoid device mismatch.
    if _maybe("head") is not None:
        head_src = _maybe("head")
        head_dev, head_dt = _get_device_dtype(head_src)

        if not _HAS_THOP:
            print("head.* skipped: THOP not installed (needed since head is on CUDA).")
        else:
            # filter_predictor
            if hasattr(head_src, "filter_predictor") and isinstance(head_src.filter_predictor, nn.Module):
                try:
                    fp = _FilterPredictorWrap(copy.deepcopy(head_src.filter_predictor).eval().to(head_dev, head_dt))
                    x = torch.randn(2, 2, C_head, H_feat, W_feat, device=head_dev, dtype=head_dt)
                    with _silence(), torch.inference_mode():
                        st = _count_thop(fp, x)
                    print(f"head.filter_predictor backend={st['backend']} MACs {_fmt_units(st['macs'],'MACs')} FLOPs {_fmt_units(st['flops'],'FLOPs')}")
                except Exception as e:
                    print(f"head.filter_predictor skipped: {type(e).__name__}: {e}")

            # classifier
            if hasattr(head_src, "classifier") and isinstance(head_src.classifier, nn.Module):
                try:
                    cls = _HeadClsWrap(copy.deepcopy(head_src.classifier).eval().to(head_dev, head_dt))
                    x = torch.randn(3, C_head, H_feat, W_feat, device=head_dev, dtype=head_dt)
                    with _silence(), torch.inference_mode():
                        st = _count_thop(cls, x)
                    print(f"head.classifier   backend={st['backend']} MACs {_fmt_units(st['macs'],'MACs')} FLOPs {_fmt_units(st['flops'],'FLOPs')}")
                except Exception as e:
                    print(f"head.classifier skipped: {type(e).__name__}: {e}")

            # bb_regressor
            if hasattr(head_src, "bb_regressor") and isinstance(head_src.bb_regressor, nn.Module):
                try:
                    bb = _HeadBBWrap(copy.deepcopy(head_src.bb_regressor).eval().to(head_dev, head_dt))
                    x = torch.randn(3, C_head, H_feat, W_feat, device=head_dev, dtype=head_dt)
                    with _silence(), torch.inference_mode():
                        st = _count_thop(bb, x)
                    print(f"head.bb_regressor backend={st['backend']} MACs {_fmt_units(st['macs'],'MACs')} FLOPs {_fmt_units(st['flops'],'FLOPs')}")
                except Exception as e:
                    print(f"head.bb_regressor skipped: {type(e).__name__}: {e}")

    # ---------- bkMlp (same device/dtype as bkMlp weights, THOP) ----------
    if _maybe("bkMlp") is not None:
        try:
            if not _HAS_THOP:
                print("bkMlp skipped: THOP not installed (bkMlp is on CUDA).")
            else:
                bk_src = _maybe("bkMlp")
                bk_dev, bk_dt = _get_device_dtype(bk_src)
                bk_copy = copy.deepcopy(bk_src).eval().to(bk_dev, bk_dt)
                bk = _BkMlpWrap(bk_copy).to(bk_dev, bk_dt)

                # auto detect input channels
                in_ch: Optional[int] = None
                for m in bk_copy.modules():
                    if isinstance(m, nn.Conv2d):
                        in_ch = int(m.in_channels)
                        break
                    if isinstance(m, nn.Linear):
                        in_ch = int(m.in_features)
                        break
                if in_ch is None:
                    in_ch = 1024

                x = torch.randn(1, in_ch, H_feat, W_feat, device=bk_dev, dtype=bk_dt)
                with _silence(), torch.inference_mode():
                    st = _count_thop(bk, x)
                print(f"bkMlp            backend={st['backend']} MACs {_fmt_units(st['macs'],'MACs')} FLOPs {_fmt_units(st['flops'],'FLOPs')}")
        except Exception as e:
            print(f"bkMlp skipped: {type(e).__name__}: {e}")

    # ---------- SideNetwork_D/U (same device/dtype, THOP) ----------
    if _maybe("SideNetwork_D") is not None:
        try:
            if not _HAS_THOP:
                print("SideNetwork_D skipped: THOP not installed.")
            else:
                sd_src = _maybe("SideNetwork_D")
                sd_dev, sd_dt = _get_device_dtype(sd_src)
                sd_copy = copy.deepcopy(sd_src).eval().to(sd_dev, sd_dt)

                x = torch.randn(2, cot_T, point_num, 128, device=sd_dev, dtype=sd_dt)
                with _silence(), torch.inference_mode():
                    st = _count_thop(sd_copy, x)
                print(f"SideNetwork_D    backend={st['backend']} MACs {_fmt_units(st['macs'],'MACs')} FLOPs {_fmt_units(st['flops'],'FLOPs')}")
        except Exception as e:
            print(f"SideNetwork_D skipped: {type(e).__name__}: {e}")

    if _maybe("SideNetwork_U") is not None:
        try:
            if not _HAS_THOP:
                print("SideNetwork_U skipped: THOP not installed.")
            else:
                su_src = _maybe("SideNetwork_U")
                su_dev, su_dt = _get_device_dtype(su_src)
                su_copy = copy.deepcopy(su_src).eval().to(su_dev, su_dt)

                x = torch.randn(2, cot_T, point_num, 64, device=su_dev, dtype=su_dt)
                with _silence(), torch.inference_mode():
                    st = _count_thop(su_copy, x)
                print(f"SideNetwork_U    backend={st['backend']} MACs {_fmt_units(st['macs'],'MACs')} FLOPs {_fmt_units(st['flops'],'FLOPs')}")
        except Exception as e:
            print(f"SideNetwork_U skipped: {type(e).__name__}: {e}")

    # ---------- cotracker_model (same device/dtype, THOP) ----------
    if _maybe("cotracker_model") is not None and _maybe("SideNetwork_D") is not None and _maybe("SideNetwork_U") is not None:
        try:
            if not _HAS_THOP:
                print("cotracker skipped: THOP not installed.")
            else:
                cot_src = _maybe("cotracker_model")
                cot_dev, cot_dt = _get_device_dtype(cot_src)

                sd_src = _maybe("SideNetwork_D")
                su_src = _maybe("SideNetwork_U")
                sd_copy = copy.deepcopy(sd_src).eval().to(cot_dev, cot_dt)
                su_copy = copy.deepcopy(su_src).eval().to(cot_dev, cot_dt)
                cot_copy = copy.deepcopy(cot_src).eval().to(cot_dev, cot_dt)

                ct = _CoTrackerWrap(cot_copy, sd_copy, su_copy, grid_size=grid_size, backward_tracking=cot_backward_tracking).to(cot_dev, cot_dt)

                lh, lw = int(label_hw[0]), int(label_hw[1])

                # IMPORTANT: everything must be cot_dt (likely fp16) to avoid "float vs half" errors
                video = torch.randn(1, cot_T, 3, cot_H, cot_W, device=cot_dev, dtype=cot_dt)
                label_first = torch.randn(1, 1, 128, lh, lw, device=cot_dev, dtype=cot_dt)
                label_last = torch.randn(1, 1, 128, lh, lw, device=cot_dev, dtype=cot_dt)
                queries = torch.randn(1, point_num, 3, device=cot_dev, dtype=cot_dt)

                with _silence(), torch.inference_mode():
                    st = _count_thop(ct, (video, label_first, label_last, queries))

                print(f"cotracker        backend={st['backend']} MACs {_fmt_units(st['macs'],'MACs')} FLOPs {_fmt_units(st['flops'],'FLOPs')}")
        except Exception as e:
            print(f"cotracker skipped: {type(e).__name__}: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    pass


# [Complexity] Component MACs/FLOPs 2 batch
# feature_extractor backend=thop MACs 221.038G MACs FLOPs 442.076G FLOPs
# head.filter_predictor backend=thop MACs 27.829G MACs FLOPs 55.657G FLOPs
# head.classifier   backend=thop MACs 131.072K MACs FLOPs 262.144K FLOPs
# head.bb_regressor backend=thop MACs 3.453G MACs FLOPs 6.907G FLOPs
# bkMlp            backend=thop MACs 1.535G MACs FLOPs 3.070G FLOPs
# SideNetwork_D    backend=thop MACs 2.171G MACs FLOPs 4.342G FLOPs
# SideNetwork_U    backend=thop MACs 25.690M MACs FLOPs 51.380M FLOPs
# cotracker        backend=thop MACs 365.492G MACs FLOPs 730.984G FLOPs

