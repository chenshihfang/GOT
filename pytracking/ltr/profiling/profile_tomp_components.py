# ltr/profiling/profile_tomp_components.py
import os
import time
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib, io, logging, sys

# ---------- Optional MACs/FLOPs backends ----------
_HAS_FVCORE = False
_HAS_PTFLOPS = False
_HAS_THOP = False

try:
    from fvcore.nn import FlopCountAnalysis
    _HAS_FVCORE = True
except Exception:
    pass

try:
    from ptflops import get_model_complexity_info as _ptflops_get_info
    _HAS_PTFLOPS = True
except Exception:
    pass

try:
    from thop import profile as _thop_profile
    _HAS_THOP = True
except Exception:
    pass


# ---------- Quiet external logging during counting ----------
@contextlib.contextmanager
def _silence_external_logs():
    old_out, old_err = sys.stdout, sys.stderr
    buf_out, buf_err = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = buf_out, buf_err
    saved = {}
    for name in ["fvcore", "ptflops", "torch", "xformers", "dinov2"]:
        logger = logging.getLogger(name)
        saved[name] = logger.level
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, lvl in saved.items():
            logging.getLogger(name).setLevel(lvl)
        sys.stdout, sys.stderr = old_out, old_err


# ---------- Formatting & basic utilities ----------
def _fmt(n: float, unit: str) -> str:
    units = ["", "K", "M", "G", "T"]
    i = 0
    while n >= 1000 and i < len(units) - 1:
        n /= 1000.0
        i += 1
    return f"{n:.3f}{units[i]} {unit}"

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

def _first_param_device_dtype(m: nn.Module):
    p = next((p for p in m.parameters() if p is not None), None)
    if p is None:
        return torch.device("cpu"), torch.float32
    return p.device, p.dtype


# ---------- Strong xFormers MEA temporary patch ----------
class _DisableXformersMEAttn:
    """
    Replace xformers memory_efficient_attention in common import paths.
    Restores originals on exit.
    """
    def __init__(self):
        self._targets = []
        self._originals = []

    def _maybe_patch_attr(self, module_name: str, attr_chain: list, new_fn):
        if module_name not in sys.modules:
            return
        obj = sys.modules[module_name]
        for a in attr_chain[:-1]:
            obj = getattr(obj, a, None)
            if obj is None:
                return
        last = attr_chain[-1]
        if not hasattr(obj, last):
            return
        self._targets.append((obj, last))
        self._originals.append(getattr(obj, last))
        setattr(obj, last, new_fn)

    def __enter__(self):
        import math, torch

        def _reference_mea(q, k, v, attn_bias=None, p=0.0, scale=None, *args, **kwargs):
            # q,k,v: (B, L, H, Dh)
            B, L, H, Dh = q.shape
            if scale is None:
                scale = 1.0 / math.sqrt(Dh)
            qh = q.permute(0, 2, 1, 3)      # (B,H,L,Dh)
            kh = k.permute(0, 2, 1, 3)
            vh = v.permute(0, 2, 1, 3)
            scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale   # (B,H,L,L)
            if attn_bias is not None:
                scores = scores + attn_bias
            prob = scores.softmax(dim=-1)
            out = torch.matmul(prob, vh)                              # (B,H,L,Dh)
            return out.permute(0, 2, 1, 3)                            # (B,L,H,Dh)

        self._maybe_patch_attr("xformers.ops", ["memory_efficient_attention"], _reference_mea)
        self._maybe_patch_attr("xformers.ops.fmha", ["memory_efficient_attention"], _reference_mea)
        self._maybe_patch_attr("dinov2.layers.attention", ["memory_efficient_attention"], _reference_mea)

        os.environ["XFORMERS_DISABLE_FLASH_ATTENTION"] = "1"
        os.environ["XFORMERS_DISABLED_ATTN_BACKENDS"] = "all"
        return self

    def __exit__(self, exc_type, exc, tb):
        for (obj, last), orig in zip(self._targets, self._originals):
            setattr(obj, last, orig)
        self._targets.clear()
        self._originals.clear()
        return False


# ---------- Counters (support single or tuple inputs) ----------
def _count_with_fvcore(module: nn.Module, inputs: Union[torch.Tensor, Tuple]) -> Dict[str, float]:
    module.eval().cpu()
    with torch.no_grad():
        ins = inputs if isinstance(inputs, tuple) else (inputs,)
        flops = FlopCountAnalysis(module, ins).total()
    macs = flops / 2.0
    params = _count_params(module)
    return {"backend": "fvcore", "flops": flops, "macs": macs, "params": params}

def _count_with_ptflops(module: nn.Module, input_res: Tuple[int, int, int]) -> Dict[str, float]:
    class _OneBatch(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.m = mod.eval().cpu()
        def forward(self, x):
            return self.m(x)

    m = _OneBatch(module)
    macs_str, params_str = _ptflops_get_info(m, input_res, as_strings=True, print_per_layer_stat=False, verbose=False)

    def _parse(s: str) -> float:
        num = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
        unit_char = "T" if "T" in s else "G" if "G" in s else "M" if "M" in s else "K" if "K" in s else ""
        mult = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
        return float(num) * mult[unit_char]

    macs = _parse(macs_str)
    params = _parse(params_str)
    flops = macs * 2.0
    return {"backend": "ptflops", "flops": flops, "macs": macs, "params": params}

def _count_with_thop(module: nn.Module, inputs: Union[torch.Tensor, Tuple]) -> Dict[str, float]:
    module.eval().cpu()
    with torch.no_grad():
        ins = inputs if isinstance(inputs, tuple) else (inputs,)
        macs, params = _thop_profile(module, inputs=ins)
    flops = macs * 2.0
    return {"backend": "thop", "flops": flops, "macs": macs, "params": params}


def _measure_macs_flops(module: nn.Module, inputs: Union[torch.Tensor, Tuple]) -> Dict[str, float]:
    """
    Count on CPU FP32 with xFormers attention disabled and external prints silenced.
    Accepts a single Tensor or a tuple of Tensors to match module forward signature.
    """
    # Move all tensors to CPU fp32
    def _to_cpu_fp32(x):
        if isinstance(x, tuple):
            return tuple(t.to(device="cpu", dtype=torch.float32) for t in x)
        return x.to(device="cpu", dtype=torch.float32)

    with torch.no_grad():
        module = module.eval().to(device="cpu", dtype=torch.float32)

    cpu_inputs = _to_cpu_fp32(inputs)

    with _DisableXformersMEAttn(), _silence_external_logs():
        if _HAS_FVCORE:
            stats = _count_with_fvcore(module, cpu_inputs)
        elif _HAS_THOP:
            stats = _count_with_thop(module, cpu_inputs)
        else:
            # last resort for plain 4D single input modules
            if not isinstance(cpu_inputs, tuple):
                assert cpu_inputs.dim() == 4 and cpu_inputs.size(0) == 1, "ptflops fallback needs (1,C,H,W)"
                C, H, W = cpu_inputs.shape[1:]
                stats = _count_with_ptflops(module, (C, H, W))
            else:
                raise RuntimeError(
                    "Install one FLOPs backend:\n"
                    "  pip install fvcore   # preferred\n"
                    "  pip install thop\n"
                    "  pip install ptflops"
                )

    stats["flops_str"]  = _fmt(stats["flops"],  "FLOPs")
    stats["macs_str"]   = _fmt(stats["macs"],   "MACs")
    stats["params_str"] = _fmt(stats["params"], "Params")
    return stats


# ---------- Minimal wrappers for your submodules ----------
class _DiNOBackbone(nn.Module):
    """Calls net.extract_dino_features_spatial_intermediate_layers(x) on (1,3,252,252)."""
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x):    return self.net.extract_dino_features_spatial_intermediate_layers(x)

class _VGGTBranchFlat(nn.Module):
    """Calls net.extract_dino_features_spatial_intermediate_layers_vggt_cat on (B,S,3,252,252), returns (S*B,C,H,W)."""
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, xs):   return self.net.extract_dino_features_spatial_intermediate_layers_vggt_cat(xs)

class _VGGTFeatMlp5D(nn.Module):
    """
    Calls net.vggtDPTfeatMlp with 5D input and fixed type string.
    Input: (B,S,256,H,W) for vggtDPTfeat_type = 'fl1'.
    """
    def __init__(self, net, vggt_type="fl1"):
        super().__init__()
        self.net = net
        self.vggt_type = vggt_type
    def forward(self, x5d):
        # module expects (feat5d, vggtDPTfeat_type)
        return self.net.vggtDPTfeatMlp(x5d, self.vggt_type)

class _VGGTHeadMLP(nn.Module):
    """Calls net.vggtDPTfeatMlp_head on (N,C,H,W)."""
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, feats): return self.net.vggtDPTfeatMlp_head(feats.contiguous())

class _BkMlpMOEv2Only(nn.Module):
    """Calls net.bkMlpMOEv2 on a list of 4 inter-layer maps [B,1024,H,W] as used by DiNO fuse."""
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x_list): return self.net.bkMlpMOEv2(x_list)

# ---- Head sub-components: filter_predictor, classifier, bb_regressor ----
class _HeadFilterPredictorWrap(nn.Module):
    """
    Wrap head.filter_predictor and return only encoded test features to keep a Tensor output.
    Forward expects (train_feat, test_feat, train_label).
    Shapes match your Head_ usage:
      train_feat: (T,B,256,18,18), test_feat: (1,B,256,18,18), train_label: (T,B,18,18)
    """
    def __init__(self, flt_pred): super().__init__(); self.f = flt_pred
    def forward(self, train_feat, test_feat, train_label):
        out = self.f(train_feat, test_feat, train_label)
        if isinstance(out, tuple) and len(out) == 2:
            # weights, test_feat_enc
            return out[1]
        if isinstance(out, tuple) and len(out) == 3:
            # cls_weights, bbreg_weights, test_feat_enc
            return out[2]
        return out  # hope it is a Tensor

class _HeadClassifierWrap(nn.Module):
    """Forward expects (feat_enc, cls_filter) and calls classifier(feat, filter)."""
    def __init__(self, clf): super().__init__(); self.clf = clf
    def forward(self, feat, flt): return self.clf(feat, flt)

class _HeadBRegWrap(nn.Module):
    """Forward expects (feat_enc, breg_filter) and calls bb_regressor(feat, filter)."""
    def __init__(self, bbr): super().__init__(); self.bbr = bbr
    def forward(self, feat, flt): return self.bbr(feat, flt)


# ---------- Latency (optional, device dependent) ----------
@torch.no_grad()
def _latency(module: nn.Module, inputs: Union[torch.Tensor, Tuple], device: str = "cuda", warmup=15, iters=100):
    module = module.eval().to(device)

    def _to(x):
        if isinstance(x, tuple):
            return tuple(t.to(device) for t in x)
        return x.to(device)
    inputs = _to(inputs)

    times = []
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup): _ = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
        torch.cuda.synchronize()
        for _ in range(iters):
            s.record()
            _ = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) / 1000.0)
    else:
        for _ in range(warmup): _ = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
        for _ in range(iters):
            t0 = time.time()
            _ = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
            times.append(time.time() - t0)

    times.sort()
    n = len(times)
    med = times[n // 2] if n % 2 else 0.5 * (times[n // 2 - 1] + times[n // 2])
    return {"device": device, "median_ms": med * 1000.0, "iters": iters}


# ---------- Public entry ----------
def profile_tomp_components(net,
                            *,
                            print_latency: bool = False,
                            latency_device: str = "cuda",
                            B: int = None,
                            S: int = None,
                            T: int = None):
    """
    B,S,T default to env METRIC_B, METRIC_S, METRIC_T or sensible small values.
    """
    # resolve shapes
    B = int(os.environ.get("METRIC_B", B or 2))
    S = int(os.environ.get("METRIC_S", S or 2))
    T = int(os.environ.get("METRIC_T", T or 2))

    # backbone feature map sizes used by your heads
    # H = W = 18       # after pooling to DinoPatch
    # IMG = 252

    H = W = 27       # after pooling to DinoPatch
    IMG = 378

    # craft dummy inputs on the same device and dtype as net params
    dev, dt = _first_param_device_dtype(net)
    x_img   = torch.randn(1, 3, IMG, IMG, device=dev, dtype=dt)         # (1,3,252,252)
    x_vggt  = torch.randn(B, S, 3, IMG, IMG, device=dev, dtype=dt)      # (B,S,3,252,252)

    # tensors for head sub-components
    feat_enc   = torch.randn(1, B, 256, H, W, device=dev, dtype=dt)     # test_feat_enc
    cls_filt   = torch.randn(B, 256, 1, 1, device=dev, dtype=dt)        # cls_filter
    breg_filt  = torch.randn(B, 256, 1, 1, device=dev, dtype=dt)        # breg_filter
    train_feat = torch.randn(T, B, 256, H, W, device=dev, dtype=dt)     # train_feat for filter_predictor
    test_feat  = torch.randn(1, B, 256, H, W, device=dev, dtype=dt)     # test_feat  for filter_predictor
    train_lbl  = torch.randn(T, B, H, W, device=dev, dtype=dt)          # train_label

    # 5D VGGT DPT features for vggtDPTfeatMlp
    vggt_feat_5d = torch.randn(B, S, 256, H*8, W*8, device=dev, dtype=dt)  # 256 channels for type 'fl1'
    vggt_head_in = torch.randn(B*S, 1024, H, W, device=dev, dtype=dt)      # typical 1024-C head input

    # Wrappers
    dino_mod    = _DiNOBackbone(net)
    vggt_flat   = _VGGTBranchFlat(net)
    vggt_mlp_5d = _VGGTFeatMlp5D(net, vggt_type="fl1")   # enforce correct path for 256-C
    vggt_head   = _VGGTHeadMLP(net)
    bkmlp_moe   = _BkMlpMOEv2Only(net)

    # optional head parts
    head = getattr(net, "head", None)
    filt_pred = getattr(head, "filter_predictor", None) if isinstance(head, nn.Module) else None
    classifier = getattr(head, "classifier", None) if isinstance(head, nn.Module) else None
    bb_regressor = getattr(head, "bb_regressor", None) if isinstance(head, nn.Module) else None

    print("\n[Complexity] Device-independent MACs/FLOPs/Params (one sample each)")

    # DiNO
    try:
        st = _measure_macs_flops(dino_mod, x_img)
        print(f"  [DiNO]               backend={st['backend']}  params={st['params_str']}  macs={st['macs_str']}  flops={st['flops_str']}")
    except Exception as e:
        print(f"  [DiNO]               skipped: {type(e).__name__}: {e}")

    # VGGT
    try:
        st = _measure_macs_flops(vggt_flat, x_vggt)
        print(f"  [VGGT]              backend={st['backend']}  params={st['params_str']}  macs={st['macs_str']}  flops={st['flops_str']}")
    except Exception as e:
        print(f"  [VGGT]              skipped: {type(e).__name__}: {e}")

    # vggtDPTfeatMlp (5D + type)
    if isinstance(getattr(net, "vggtDPTfeatMlp", None), nn.Module):
        try:
            st = _measure_macs_flops(vggt_mlp_5d, vggt_feat_5d)
            print(f"  [vggtDPTfeatMlp]    backend={st['backend']}  params={st['params_str']}  macs={st['macs_str']}  flops={st['flops_str']}")
        except Exception as e:
            print(f"  [vggtDPTfeatMlp]    skipped: {type(e).__name__}: {e}")
    else:
        print("  [vggtDPTfeatMlp]    skipped: Not found")

    # vggtDPTfeatMlp_head
    if isinstance(getattr(net, "vggtDPTfeatMlp_head", None), nn.Module):
        try:
            st = _measure_macs_flops(vggt_head, vggt_head_in)
            print(f"  [vggtDPTfeatMlp_head] backend={st['backend']}  params={st['params_str']}  macs={st['macs_str']}  flops={st['flops_str']}")
        except Exception as e:
            print(f"  [vggtDPTfeatMlp_head] skipped: {type(e).__name__}: {e}")
    else:
        print("  [vggtDPTfeatMlp_head] skipped: Not found")

    # bkMlpMOEv2 alone (list of 4 maps [1,1024,27,27])
    if isinstance(getattr(net, "bkMlpMOEv2", None), nn.Module):
        try:
            inter = [torch.randn(1, 1024, H, W, device=dev, dtype=dt) for _ in range(4)]
            st = _measure_macs_flops(bkmlp_moe, (inter,))  # pass as tuple since it is a list arg
            print(f"  [bkMlpMOEv2]        backend={st['backend']}  params={st['params_str']}  macs={st['macs_str']}  flops={st['flops_str']}")
        except Exception as e:
            # still show params if counting fails
            try:
                p = _count_params(net.bkMlpMOEv2)
                print(f"  [bkMlpMOEv2]        params={_fmt(float(p), 'Params')}  (FLOPs skipped: {type(e).__name__})")
            except Exception as e2:
                print(f"  [bkMlpMOEv2]        skipped: {type(e2).__name__}: {e2}")
    else:
        print("  [bkMlpMOEv2]        skipped: Not found")

    # ---- Head sub-components ----
    # filter_predictor
    if isinstance(filt_pred, nn.Module):
        try:
            st = _measure_macs_flops(_HeadFilterPredictorWrap(filt_pred), (train_feat, test_feat, train_lbl))
            print(f"  [head.filter_predictor] backend={st['backend']}  params={st['params_str']}  macs={st['macs_str']}  flops={st['flops_str']}")
        except Exception as e:
            try:
                p = _count_params(filt_pred)
                print(f"  [head.filter_predictor] params={_fmt(float(p), 'Params')}  (FLOPs skipped: {type(e).__name__})")
            except Exception as e2:
                print(f"  [head.filter_predictor] skipped: {type(e2).__name__}: {e2}")
    else:
        print("  [head.filter_predictor] skipped: Not found")

    # classifier
    if isinstance(classifier, nn.Module):
        try:
            st = _measure_macs_flops(_HeadClassifierWrap(classifier), (feat_enc, cls_filt))
            print(f"  [head.classifier]   backend={st['backend']}  params={st['params_str']}  macs={st['macs_str']}  flops={st['flops_str']}")
        except Exception as e:
            try:
                p = _count_params(classifier)
                print(f"  [head.classifier]   params={_fmt(float(p), 'Params')}  (FLOPs skipped: {type(e).__name__})")
            except Exception as e2:
                print(f"  [head.classifier]   skipped: {type(e2).__name__}: {e2}")
    else:
        print("  [head.classifier]   skipped: Not found")

    # bb_regressor
    if isinstance(bb_regressor, nn.Module):
        try:
            st = _measure_macs_flops(_HeadBRegWrap(bb_regressor), (feat_enc, breg_filt))
            print(f"  [head.bb_regressor] backend={st['backend']}  params={st['params_str']}  macs={st['macs_str']}  flops={st['flops_str']}")
        except Exception as e:
            try:
                p = _count_params(bb_regressor)
                print(f"  [head.bb_regressor] params={_fmt(float(p), 'Params')}  (FLOPs skipped: {type(e).__name__})")
            except Exception as e2:
                print(f"  [head.bb_regressor] skipped: {type(e2).__name__}: {e2}")
    else:
        print("  [head.bb_regressor] skipped: Not found")

    # # ---- Optional latency for DiNO and VGGT only (stable) ----
    # if print_latency:
    #     print("\n[Latency] Device-dependent (median over 100 iters)")
    #     for name, mod, inp in [
    #         ("DiNO", dino_mod, x_img),
    #         ("VGGT", vggt_flat, x_vggt),
    #     ]:
    #         try:
    #             lat = _latency(mod, inp, device=latency_device, warmup=20, iters=100)
    #             print(f"  [{name}] device={lat['device']}  median={lat['median_ms']:.3f} ms  iters={lat['iters']}")
    #         except Exception as e:
    #             print(f"  [{name}] skipped: {type(e).__name__}: {e}")
    input()

# Example:
# CUDA_VISIBLE_DEVICES=0 \
# METRIC_PROFILE=1 METRIC_LAT=1 METRIC_LAT_DEV=cuda \
# METRIC_B=2 METRIC_S=2 METRIC_T=2 \
# python ltr/run_training_dsA.py tomp <your_config_name>


# CUDA_VISIBLE_DEVICES=1 METRIC_PROFILE=1 METRIC_TOTALS=1 METRIC_LAT=1 METRIC_LAT_DEV=cuda METRIC_B=1 METRIC_S=1 python ltr/run_training_dsA.py tomp tompL_252_bkMlpMOEv2_inter4_JEPAs2_vggt_AlphaEditv2_ds0_magav1_checkptr_trgc_splitCfilter_EditEnergy2em2_VGGTO_wofrbn

#   [DiNO]               backend=fvcore  params=1.639G Params  macs=52.670G MACs  flops=105.340G FLOPs
#   [VGGT]              backend=fvcore  params=1.639G Params  macs=168.106G MACs  flops=336.211G FLOPs
#   [vggtDPTfeatMlp]    backend=fvcore  params=1.639G Params  macs=5.348G MACs  flops=10.697G FLOPs
#   [vggtDPTfeatMlp_head] backend=fvcore  params=1.639G Params  macs=42.550M MACs  flops=85.101M FLOPs
#   [bkMlpMOEv2]        params=26.321M Params  (FLOPs skipped: AttributeError)
#   [head.filter_predictor] params=17.447M Params  (FLOPs skipped: TypeError)
#   [head.classifier]   backend=fvcore  params=65.792K Params  macs=74.240K MACs  flops=148.480K FLOPs
#   [head.bb_regressor] backend=fvcore  params=2.437M Params  macs=384.603M MACs  flops=769.205M FLOPs


#   [DiNO]               backend=fvcore  params=1.639G Params  macs=125.574G MACs  flops=251.148G FLOPs
#   [VGGT]              backend=fvcore  params=1.639G Params  macs=375.492G MACs  flops=750.984G FLOPs
#   [vggtDPTfeatMlp]    backend=fvcore  params=1.639G Params  macs=12.034G MACs  flops=24.068G FLOPs
#   [vggtDPTfeatMlp_head] backend=fvcore  params=1.639G Params  macs=95.738M MACs  flops=191.476M FLOPs
#   [bkMlpMOEv2]        params=26.321M Params  (FLOPs skipped: AttributeError)
#   [head.filter_predictor] params=17.447M Params  (FLOPs skipped: TypeError)
#   [head.classifier]   backend=fvcore  params=65.792K Params  macs=126.080K MACs  flops=252.160K FLOPs
#   [head.bb_regressor] backend=fvcore  params=2.437M Params  macs=865.315M MACs  flops=1.731G FLOPs




# CUDA_VISIBLE_DEVICES=0 METRIC_PROFILE=1 METRIC_TOTALS=1 METRIC_LAT=1 METRIC_LAT_DEV=cuda METRIC_B=1 METRIC_S=3 python ltr/run_training_dsA.py tomp tompL_252_bkMlpMOEv2_inter4_JEPAs2_vggt_AlphaEditv2_ds0_magav1_checkptr_trgc_splitCfilter_EditEnergy2em2_VGGTO_wofrbn

#   [DiNO]               backend=fvcore  params=1.639G Params  macs=52.670G MACs  flops=105.340G FLOPs
#   [VGGT]              backend=fvcore  params=1.639G Params  macs=504.317G MACs  flops=1.009T FLOPs
#   [vggtDPTfeatMlp]    backend=fvcore  params=1.639G Params  macs=16.045G MACs  flops=32.091G FLOPs
#   [vggtDPTfeatMlp_head] backend=fvcore  params=1.639G Params  macs=127.651M MACs  flops=255.302M FLOPs
#   [bkMlpMOEv2]        params=26.321M Params  (FLOPs skipped: AttributeError)
#   [head.filter_predictor] params=17.447M Params  (FLOPs skipped: TypeError)
#   [head.classifier]   backend=fvcore  params=65.792K Params  macs=74.240K MACs  flops=148.480K FLOPs
#   [head.bb_regressor] backend=fvcore  params=2.437M Params  macs=384.603M MACs  flops=769.205M FLOPs

#   [DiNO]               backend=fvcore  params=1.639G Params  macs=125.574G MACs  flops=251.148G FLOPs
#   [VGGT]              backend=fvcore  params=1.639G Params  macs=1.126T MACs  flops=2.253T FLOPs
#   [vggtDPTfeatMlp]    backend=fvcore  params=1.639G Params  macs=36.102G MACs  flops=72.204G FLOPs
#   [vggtDPTfeatMlp_head] backend=fvcore  params=1.639G Params  macs=287.214M MACs  flops=574.429M FLOPs
#   [bkMlpMOEv2]        params=26.321M Params  (FLOPs skipped: AttributeError)
#   [head.filter_predictor] params=17.447M Params  (FLOPs skipped: TypeError)
#   [head.classifier]   backend=fvcore  params=65.792K Params  macs=126.080K MACs  flops=252.160K FLOPs
#   [head.bb_regressor] backend=fvcore  params=2.437M Params  macs=865.315M MACs  flops=1.731G FLOPs



# CUDA_VISIBLE_DEVICES=1 METRIC_PROFILE=1 METRIC_TOTALS=1 METRIC_LAT=1 METRIC_LAT_DEV=cuda METRIC_B=1 METRIC_S=1 python ltr/run_training_dsA.py tomp tompL_378_bkMlpresdv2_inter4_JEPAs2_vast_vggt_AlphaEditv2_ds0_magav1_checkptr_trgc_splitCfilter_EditEnergy2em2_VGGTO_wofrbn_2refs

# [Complexity] Device-independent MACs/FLOPs/Params (one sample each)
#   [DiNO]               backend=fvcore  params=1.614G Params  macs=124.406G MACs  flops=248.812G FLOPs
#   [VGGT]              backend=fvcore  params=1.614G Params  macs=750.983G MACs  flops=1.502T FLOPs
#   [vggtDPTfeatMlp]    backend=fvcore  params=1.614G Params  macs=24.068G MACs  flops=48.136G FLOPs
#   [vggtDPTfeatMlp_head] backend=fvcore  params=1.614G Params  macs=191.476M MACs  flops=382.952M FLOPs
#   [bkMlpMOEv2]        skipped: Not found
#   [head.filter_predictor] params=17.447M Params  (FLOPs skipped: TypeError)
#   [head.classifier]   backend=fvcore  params=65.792K Params  macs=126.080K MACs  flops=252.160K FLOPs
#   [head.bb_regressor] backend=fvcore  params=2.437M Params  macs=865.315M MACs  flops=1.731G FLOPs






# CUDA_VISIBLE_DEVICES=1 METRIC_PROFILE=1 METRIC_TOTALS=1 METRIC_LAT=1 METRIC_LAT_DEV=cuda METRIC_B=1 METRIC_S=3 python ltr/run_training_dsA.py tomp tompL_378_bkMlpresdv2_inter4_JEPAs2_vast_vggt_AlphaEditv2_ds0_magav1_checkptr_trgc_splitCfilter_EditEnergy2em2_VGGTO_wofrbn_2refs
#   [DiNO]               backend=fvcore  params=1.614G Params  macs=124.406G MACs  flops=248.812G FLOPs
#   [VGGT]              backend=fvcore  params=1.614G Params  macs=1.126T MACs  flops=2.253T FLOPs
#   [vggtDPTfeatMlp]    backend=fvcore  params=1.614G Params  macs=36.102G MACs  flops=72.204G FLOPs
#   [vggtDPTfeatMlp_head] backend=fvcore  params=1.614G Params  macs=287.214M MACs  flops=574.429M FLOPs
#   [bkMlpMOEv2]        skipped: Not found
#   [head.filter_predictor] params=17.447M Params  (FLOPs skipped: TypeError)
#   [head.classifier]   backend=fvcore  params=65.792K Params  macs=126.080K MACs  flops=252.160K FLOPs
#   [head.bb_regressor] backend=fvcore  params=2.437M Params  macs=865.315M MACs  flops=1.731G FLOPs

