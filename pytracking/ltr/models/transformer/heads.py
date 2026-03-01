import torch
import torch.nn as nn
import ltr.models.layers.filter as filter_layer
import math
from ltr.models.transformer.position_encoding import PositionEmbeddingSine
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn.init as init
from heatmap import heat_show
from torch import Tensor
from functools import partial
import collections
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import cv2
from timm.models.layers import DropPath

from typing import Tuple, Union, Optional, Dict, Sequence, List

from torch.cuda.amp import autocast

try:
    from timm.models.layers import drop, drop_path, trunc_normal_
except:
    pass

from collections import OrderedDict

# DinoPatch = 18
DinoPatch = 27
print("DinoPatch head", DinoPatch)


############################################ GOT-Edit


### Head_JEPAs2 (wP) AlphaEdit Head_AlphaEditv2_robust  # GOT-Edit Online Model Editing
class Head(nn.Module):
    """
    Context-Aware Perturbation Generation with parallel filter prediction and strong numerical guards.
    Hardened version: gate/JEPA/Wvggt run in fp32, pre/post sanitization, and feature LayerNorm.
    """

    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False,
                 enable_grad_nan_guard: bool = False):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

        self.permute = 1
        self.use_3d = True

        self.splitCfilter = True
        # self.splitCfilter = False

        print("self.splitCfilter head", self.splitCfilter)

        # self.auto_cast_full = True
        self.auto_cast_full = False
        print("self.auto_cast_full head", self.auto_cast_full)

        self.regMethod = "regVGGTFeatW"
        print("self.regMethod head", self.regMethod)

        # Prefer fp32 for stability unless explicitly forced elsewhere
        self.dtype = (torch.bfloat16
                      if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                      else torch.float32)

        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        self.AlphaEditHyperparams = {
            # ===== Flags (set these for ablations) =====

            "use_m2_adaptive_cov_reg": True,     # editACR
            # "use_m2_adaptive_cov_reg": False,     # editACR

            # ===== Shared / original knobs =====
            "alpha": 1e-2,               # fixed ridge (original M2)

            # "energy_threshold": 1e-2,    # hard threshold (original M1)
            "energy_threshold": 2e-2,    # hard threshold (original M1)

            "trust": 0.2,               # tau0 (original M3 and also tau0 for gated M3)

            # optional elementwise clamp after projection (None disables)
            "delta_clip": 1.0,

            # minimum null dims

            # ===== M2 adaptive ridge knobs =====
            "gamma": 3e-3,              # adaptive ridge scale (upgraded M2)   
            "ridge_min": 0.0,           # lower bound for (gamma*mu) to avoid tiny ridge in degenerate cases

        }

        if not self.auto_cast_full:
            self.AlphaEditHyperparams["delta_clip"] = None

        print("self.AlphaEditHyperparams", self.AlphaEditHyperparams)
        print("Head_AlphaEditv2 (hardened)")

        # global numeric clamps for safety
        if not self.auto_cast_full:
            self._finite_clip = 1e6
        else:
            self._finite_clip = 1e5

        # optional training-time guards
        self._enable_grad_nan_guard = enable_grad_nan_guard
        if self._enable_grad_nan_guard:
            self._register_grad_nan_guard()

    # ---------- utility: strict finiteness and clamping ----------

    @staticmethod
    def _is_finite(x: torch.Tensor) -> bool:
        return torch.isfinite(x).all().item()

    def _sanitize(self, x: torch.Tensor, name: str, verbose: bool = True) -> torch.Tensor:
        """
        Logging-only: report non-finite values but DO NOT modify tensor.
        (We avoid nan_to_num by request.)
        """
        if not torch.isfinite(x).all():
            if verbose and self.training:
                print(f"[Sanitize-LOG] Non-finite detected in {name}; left unmodified.")
        return x  # no changes

    def _register_grad_nan_guard(self):
        """
        Gradient guard without nan_to_num:
          - Non-finite grad elements set to 0
          - Mild magnitude clamp to avoid extreme spikes
        """
        g_clip = 1e3

        def _guard(g):
            if g is None:
                return g
            finite = torch.isfinite(g)
            g = torch.where(finite, g, torch.zeros_like(g))
            g = torch.clamp(g, min=-g_clip, max=g_clip)
            return g

        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(_guard)

    # ---------- a safe call wrapper for numerically spiky modules ----------

    def _safe_fp32_call(self, fn, name: str, *args, **kwargs):
        """
        Run the given callable in fp32 with pre/post sanitization.
        Accepts/returns tensors or tuples/lists of tensors.
        """
        def _prep(t):
            if torch.is_tensor(t):
                if t.is_floating_point():
                    t = self._sanitize(t, f"{name}(in)", verbose=False).to(torch.float32)
                return t
            if isinstance(t, (list, tuple)):
                return type(t)(_prep(u) for u in t)
            return t

        def _finish(t):
            if torch.is_tensor(t):
                t = self._sanitize(t, f"{name}(out)", verbose=False)
                return t
            if isinstance(t, (list, tuple)):
                return type(t)(_finish(u) for u in t)
            return t

        args = _prep(args)
        kwargs = {k: _prep(v) for k, v in kwargs.items()}
        with torch.cuda.amp.autocast(enabled=False):
            out = fn(*args, **kwargs)
        return _finish(out)

    class AlphaEditRefiner_mix:
        @staticmethod
        def _assert_finite(t: torch.Tensor, name: str):
            if not torch.isfinite(t).all():
                bad = t[~torch.isfinite(t)]
                raise RuntimeError(f"{name} contains non-finite values; sample={bad[:8]}")

        def _create_null_space_projector(self, k0: torch.Tensor, hparams: dict) -> torch.Tensor:
            """
            Build P_null from k0.

            Supports ablations:
              - M1: hard threshold vs cumulative energy selection
              - M2: fixed alpha ridge (sum-cov) vs adaptive ridge (token-avg cov + gamma*mu)
            """
            device = k0.device
            internal_dtype = torch.float32

            # Normalize to (B, C, N)
            if k0.dim() == 5:
                k0 = k0.reshape(-1, *k0.shape[-3:]).contiguous()
            if k0.dim() == 4:
                B, C, H, W = k0.shape
                k0_batched = k0.to(internal_dtype).reshape(B, C, -1).contiguous()  # (B,C,N)
            elif k0.dim() == 3:
                B, C, _ = k0.shape
                k0_batched = k0.to(internal_dtype).contiguous()
            else:
                raise ValueError(f"k0 must be 3D/4D/5D, got {tuple(k0.shape)}")

            N = k0_batched.shape[2]
            finite_per_sample = torch.isfinite(k0_batched).view(B, -1).all(dim=1)

            # Whitening (kept as-is; not part of the 3 ablation flags)
            mean = k0_batched.mean(dim=2, keepdim=True)
            std = k0_batched.std(dim=2, keepdim=True).clamp_min(1e-4)
            k0w = (k0_batched - mean) / std  # (B,C,N)

            # Raw covariance sum (B,C,C)
            cov_sum = torch.bmm(k0w, k0w.transpose(1, 2))

            # ---------------- M2: covariance regularization mode ----------------
            use_m2_adaptive = bool(hparams.get("use_m2_adaptive_cov_reg", False))

            I = torch.eye(C, device=device, dtype=internal_dtype).expand(B, C, C)

            if use_m2_adaptive:
                # upgraded M2: token-averaged covariance + (gamma*mu)I
                Ctok = cov_sum / float(max(N, 1))  # (1/N)ZZ^T

                # mu = tr(Ctok)/C  (per batch)
                tr = torch.diagonal(Ctok, dim1=1, dim2=2).sum(dim=1)  # (B,)
                mu = tr / float(C)  # (B,)

                gamma = float(hparams.get("gamma", 1e-2))
                ridge_min = float(hparams.get("ridge_min", 0.0))
                ridge = (gamma * mu).clamp_min(ridge_min)  # (B,)

                cov_reg = Ctok + ridge.view(B, 1, 1) * I  # (B,C,C)

                # SVD
                U, S, _ = torch.linalg.svd(cov_reg, full_matrices=False)

                # Floor singular values by ridge (per batch) to avoid tiny values
                S = torch.maximum(S, ridge.view(B, 1).expand_as(S))

            else:
                # original M2: covariance sum + fixed alpha*I
                alpha = float(hparams.get("alpha", 1e-2))
                cov_reg = cov_sum + alpha * I

                U, S, _ = torch.linalg.svd(cov_reg, full_matrices=False)
                S = S.clamp_min(alpha)


            # normalized energies (sum to 1)
            energy = S / (S.sum(dim=1, keepdim=True) + 1e-12)  # (B,C)

            # original M1: hard threshold on normalized energy
            e_thr = float(hparams.get("energy_threshold", 1e-2))
            null_mask = (energy <= e_thr)  # (B,C) bool


            # Ensure at least one null direction per sample
            nothing = (~null_mask).all(dim=1)
            if nothing.any():
                idx_min = torch.argmin(S, dim=1)
                null_mask[nothing, :] = False
                null_mask[nothing, idx_min[nothing]] = True

            # Build projector: U diag(null_mask) U^T
            Msel = torch.diag_embed(null_mask.to(U.dtype))  # (B,C,C)
            P = torch.bmm(U, torch.bmm(Msel, U.transpose(1, 2)))
            P = 0.5 * (P + P.transpose(1, 2))

            # Fallback: neutralize if input had non-finite values
            if (~finite_per_sample).any():
                P[~finite_per_sample] = 0.0

            return P.to(device=device, dtype=internal_dtype)

        def refine_weights_with_alphaedit(
            self,
            k0: torch.Tensor,
            w: torch.Tensor,
            w_vggt_delta: torch.Tensor,
            hparams: dict
        ) -> torch.Tensor:
            """
            Refine w by projecting delta onto null-space of k0.

            Supports ablations:
              - M3: fixed trust vs relative-magnitude gating
            """
            P_N = self._create_null_space_projector(k0=k0, hparams=hparams).to(dtype=torch.float32)

            if w_vggt_delta.dim() != 4 or w.dim() != 4:
                raise ValueError(
                    f"w and w_vggt_delta must be (B, C, H, W); got {tuple(w.shape)}, {tuple(w_vggt_delta.shape)}"
                )

            Bd, C, H, W = w_vggt_delta.shape
            Bw = w.shape[0]
            Bp = P_N.shape[0]

            if not (Bp == Bd == Bw or (Bp == 1 and Bd == Bw) or (Bd == 1 and Bp == Bw) or (Bw == 1 and Bp == Bd)):
                raise ValueError(f"Batch mismatch: P_N(B={Bp}), delta(B={Bd}), w(B={Bw})")

            B = max(Bp, Bd, Bw)
            if P_N.shape[0] != B:
                P_N = P_N.expand(B, C, C).contiguous()
            if w_vggt_delta.shape[0] != B:
                w_vggt_delta = w_vggt_delta.expand(B, C, H, W).contiguous()
            if w.shape[0] != B:
                w = w.expand(B, C, H, W).contiguous()

            delta_vec = w_vggt_delta.to(torch.float32).reshape(B, C, -1)
            self._assert_finite(delta_vec, "delta_vec")
            self._assert_finite(P_N, "P_N before bmm")

            delta_null_vec = torch.bmm(P_N, delta_vec)
            self._assert_finite(delta_null_vec, "delta_null_vec")

            delta_null = delta_null_vec.reshape(B, C, H, W)

            # Optional clamp (kept as  existing knob)
            delta_clip = hparams.get("delta_clip", None)
            if isinstance(delta_clip, (float, int)) and float(delta_clip) > 0:
                dc = float(delta_clip)
                delta_null = torch.clamp(delta_null, -dc, dc)

            # ---------------- M3: step-size / gating mode ----------------

            tau0 = float(hparams.get("trust", 0.2))  # keep  naming: trust == tau0
            # original M3: fixed step
            tau = torch.full((B, 1, 1, 1), tau0, device=delta_null.device, dtype=torch.float32)
            

            w_refined = w.to(torch.float32) + tau * delta_null
            self._assert_finite(w_refined, "w_refined")

            return w_refined

    # ---------- main forward ----------

    def forward(self, train_feat, train_vggt_dpt_feats_head,
                test_vggt_dpt_feats_head, test_feat,
                train_bb,
                JEPA_predictor_cls, JEPA_predictor_breg,
                WvggtLinearCls,
                DiNO_VGGT_Gate,
                auto_cast_full,
                dtype,
                *args, **kwargs):

        num_sequences = train_bb.shape[1]

        # Flatten any 5D input
        if train_feat.dim() == 5: train_feat = train_feat.reshape(-1, *train_feat.shape[-3:]).contiguous()
        if test_feat.dim() == 5:  test_feat  = test_feat.reshape(-1,  *test_feat.shape[-3:]).contiguous()

        # Extract features
        train_feat_orig = self.extract_head_feat(train_feat, num_sequences)
        test_feat_orig  = self.extract_head_feat(test_feat,  num_sequences)

        # Sanitize early to stop bad values at source
        train_feat_orig = self._sanitize(train_feat_orig, "train_feat_orig")
        test_feat_orig  = self._sanitize(test_feat_orig,  "test_feat_orig")

        # Reshape auxiliary features (keep  shapes, just ensure contiguity)
        train_vggt_dpt_feats_head = train_vggt_dpt_feats_head.reshape(
            train_feat_orig.shape[0], train_feat_orig.shape[1], *train_vggt_dpt_feats_head.shape[1:]
        ).contiguous()
        test_vggt_dpt_feats_head  = test_vggt_dpt_feats_head.reshape(
            1, test_feat_orig.shape[1],  *test_vggt_dpt_feats_head.shape[1:]
        ).contiguous()

        # ---- Fusion gate (force fp32) ----
        train_feat_vggt, test_feat_vggt = self._safe_fp32_call(
            lambda a, b, c, d, u3, si: DiNO_VGGT_Gate(a, b, c, d, u3, si),
            "DiNO_VGGT_Gate",
            train_feat_orig,
            train_vggt_dpt_feats_head.to(train_feat_orig.device),
            test_feat_orig,
            test_vggt_dpt_feats_head.to(test_feat_orig.device),
            self.use_3d,
            False
        )


        # Base filters and encodings
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(
            auto_cast_full, dtype, train_feat_orig, test_feat_orig, *args, **kwargs
        )

        # Sanitize enc and weights (base)
        cls_filter    = self._sanitize(cls_filter,    "cls_filter(base)")
        breg_filter   = self._sanitize(breg_filter,   "breg_filter(base)")
        test_feat_enc = self._sanitize(test_feat_enc, "test_feat_enc(base)")

        # ---- Predictor heads (force fp32) ----
        cls_filter = self._safe_fp32_call(JEPA_predictor_cls, "JEPA_predictor_cls", cls_filter)
        breg_filter = self._safe_fp32_call(JEPA_predictor_breg, "JEPA_predictor_breg", breg_filter)

        # VGGT branch filters and encodings with memory context
        cls_filter_vggt, breg_filter_vggt, test_feat_enc_vggt = self.get_filter_and_features(
            auto_cast_full, dtype, train_feat_vggt, test_feat_vggt,*args, **kwargs
        )
        cls_filter_vggt     = self._sanitize(cls_filter_vggt,     "cls_filter_vggt(pre)")
        breg_filter_vggt    = self._sanitize(breg_filter_vggt,    "breg_filter_vggt(pre)")
        test_feat_enc_vggt  = self._sanitize(test_feat_enc_vggt,  "test_feat_enc_vggt(pre)")

        # ---- JEPA on VGGT branch + linear head (force fp32) ----

        if not self.splitCfilter:
            cls_filter_vggt = self._safe_fp32_call(JEPA_predictor_cls, "JEPA_predictor_cls(vggt)", cls_filter_vggt)

        cls_filter_vggt = self._safe_fp32_call(WvggtLinearCls, "WvggtLinearCls", cls_filter_vggt)
        breg_filter_vggt = self._safe_fp32_call(JEPA_predictor_breg, "JEPA_predictor_breg(vggt)", breg_filter_vggt)

        # AlphaEdit refinement (strict fp32)
        refiner = self.AlphaEditRefiner_mix()
        with torch.cuda.amp.autocast(enabled=False):
            test_feat_enc_fp32 = self._sanitize(test_feat_enc.float(), "test_feat_enc(fp32)")
            if not torch.isfinite(test_feat_enc_fp32).all():
                if self.training:
                    print("[AlphaEdit] Non-finite test_feat_enc detected; skipping refinement for this batch.")
                refined_cls_filter = cls_filter.float()
            else:
                refined_cls_filter = refiner.refine_weights_with_alphaedit(
                    k0=test_feat_enc_fp32.contiguous(),
                    w=cls_filter.float(),
                    w_vggt_delta=cls_filter_vggt.float(),
                    hparams=self.AlphaEditHyperparams
                )

        # Last guard before heads
        refined_cls_filter  = self._sanitize(refined_cls_filter,  "refined_cls_filter", verbose=False)
        test_feat_enc_vggt  = self._sanitize(test_feat_enc_vggt,  "test_feat_enc_vggt", verbose=False)
        breg_filter_vggt    = self._sanitize(breg_filter_vggt,    "breg_filter_vggt", verbose=False)

        # Heads (classification + bbox)
        target_scores = self.classifier(test_feat_enc_vggt, refined_cls_filter)
        bbox_preds    = self.bb_regressor(test_feat_enc_vggt, breg_filter_vggt)

        # Final guard for outputs
        target_scores = self._sanitize(target_scores, "target_scores(out)", verbose=False)
        bbox_preds    = self._sanitize(bbox_preds,    "bbox_preds(out)",   verbose=False)

        return target_scores, bbox_preds

    # ---------- helpers from  original code (kept) ----------

    def extract_head_feat(self, feat, num_sequences=None):
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)
        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:]).contiguous()

    def get_filter_and_features(self, auto_cast_full, dtype, train_feat, test_feat, train_label, *args, **kwargs):
        if auto_cast_full:
            with torch.cuda.amp.autocast(dtype=dtype):
                results = self.filter_predictor(
                    train_feat, test_feat, train_label, *args, **kwargs
                )
        else:
            results = self.filter_predictor(
                train_feat, test_feat, train_label,  *args, **kwargs
            )

        weights, test_feat_enc = results

        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights = weights
        else:
            cls_weights = bbreg_weights = weights

        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
        """
        Parallel filter predictor for classification and bounding box regression.
        Supports context-aware architecture.
        """
        if self.auto_cast_full:
            with torch.cuda.amp.autocast(dtype=self.dtype):
                cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
                    = self.filter_predictor.predict_cls_bbreg_filters_parallel(
                        train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
                    )
        else:
            cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
                = self.filter_predictor.predict_cls_bbreg_filters_parallel(
                    train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
                )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc



# DiNO_VGGT_Gate DiNOVGGTGate2 # GOT-Edit Alignment and Fusion
class DiNO_VGGT_Gate(nn.Module):
    """
    Revision 2: Direct Spatial Fusion.
    1. No Explicit Global Gate: 3D features are passed directly to the local fusion stage.
    2. Local Fusion Dominant: The spatial gate in local fusion is solely responsible
       for modulating and integrating 3D features with 2D features.
       - Output emphasizes 2D features: F_out = F_2D + spatial_gate * F_3D_original.
       - Can learn to ignore 3D features if its gate output is near zero.
    3. Initialization: Uses Xavier uniform.
    """
    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels

        # --- Local Fusion Modules (spatial gating is the primary control) ---
        # Takes concatenated 2D and ORIGINAL (ungated) 3D features
        self.local_fusion_gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False), # Depthwise
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1), # Pointwise
            nn.Sigmoid() # Spatial gate map
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("DiNO_VGGT_Gate2 _reset_parameters done")

    def _perform_local_fusion(self, feat_2d: torch.Tensor, feat_3d_original: torch.Tensor) -> torch.Tensor:
        """ Performs local spatial fusion using original 3D features. """
        # feat_2d, feat_3d_original: [BatchSlice, C, H, W]
        combined_feat_for_local_gate = torch.cat((feat_2d, feat_3d_original), dim=1)
        local_spatial_gate = self.local_fusion_gate_conv(combined_feat_for_local_gate)
        return feat_2d + local_spatial_gate * feat_3d_original # Key difference: feat_3d_original

    def forward(self, prev_2d_feats: torch.Tensor, prev_3d_feats: torch.Tensor,
                curr_2d_feats: torch.Tensor, curr_3d_feats: torch.Tensor,
                use_3d: bool = True, single_in: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

        if not single_in:
            N_prev, B, C, H, W = prev_2d_feats.shape
            N_curr, B, C, H, W = curr_2d_feats.shape
        else:
            N_curr, B, C, H, W = curr_2d_feats.shape     

        # Reshape for efficient batch processing
        if not single_in:
            prev_2d_flat = prev_2d_feats.reshape(N_prev * B, C, H, W)

        curr_2d_flat = curr_2d_feats.reshape(N_curr * B, C, H, W)

        # Handle 3D features based on use_3d flag
        if not self.training and not use_3d:
            # If not using 3D at test time, effectively use zero tensors for 3D part,
            # so local_spatial_gate * zero_3d_tensor is zero. Output is just 2D.
            # Or simply return 2D features if fusion is identity with zero 3D.
            # To be absolutely sure only 2D is returned:
            return prev_2d_feats, curr_2d_feats
            # prev_3d_flat = torch.zeros_like(prev_2d_flat) # This also works if fusion must run
            # curr_3d_flat = torch.zeros_like(curr_2d_flat)
        else:
            if not single_in:
                prev_3d_flat = prev_3d_feats.reshape(N_prev * B, C, H, W)
            curr_3d_flat = curr_3d_feats.reshape(N_curr * B, C, H, W)


        # Perform local fusion for previous and current features separately
        if not single_in:
            fused_prev_flat = self._perform_local_fusion(prev_2d_flat, prev_3d_flat)
        fused_curr_flat = self._perform_local_fusion(curr_2d_flat, curr_3d_flat)

        # Reshape back
        if not single_in:
            fused_prev = fused_prev_flat.reshape(N_prev, B, C, H, W)

        fused_curr = fused_curr_flat.reshape(N_curr, B, C, H, W)

        if not single_in:
            return fused_prev, fused_curr
        else:
            return fused_curr



############################################################################################################################################




def conv_layer(inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1):
    layers = [
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.GroupNorm(1, outplanes),
        nn.ReLU(inplace=True),
    ]
    return layers


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = d_model
        d_model = int(np.ceil(d_model / 4) * 2)
        # print("d_model", d_model)
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        B, C, H, W = tensor.shape
        pos_x = torch.arange(H, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(W, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((H, W, self.d_model * 2), device=tensor.device).type(tensor.type())
        emb[:, :, :self.d_model] = emb_x
        emb[:, :, self.d_model:2*self.d_model] = emb_y

        emb = emb.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        if C != self.d_model * 2:
            emb = emb[:, :C, :, :]

        return tensor + emb

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding1D, self).__init__()
        self.channels = int(np.ceil(channels / 2) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        x = tensor.shape[2]  # Assuming tensor shape is [batch, features, length]
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, :self.channels] = emb_x
        return emb[None, :, :].repeat(tensor.size(0), 1, 1)  # Expand to batch size


def heat_show(cls_score, name, patch_size=27):
    size = patch_size * 4

    try:
        heatmap = cv2.resize(cls_score, (size, size), interpolation=cv2.INTER_AREA)
    except:
        heatmap = cv2.resize(cls_score.cpu().detach().numpy(), (size, size), interpolation=cv2.INTER_AREA)

    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, heatmap)
    cv2.waitKey(0)



### Head_ToMP
class Head_(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False,
                 ):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

        self.permute = 1
        print("Head_ToMP")

    def forward(self, train_feat, test_feat, train_bb, 
                *args, **kwargs):
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        
        ### note

        # print("self.separate_filters_for_cls_and_bbreg", self.separate_filters_for_cls_and_bbreg) # False

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        # train_feat.shape 1 torch.Size([B,1024, h, w])
        # test_feat.shape 1 torch.Size([B,1024, h, w])
        
        ### note

        
        
        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        ### note
        # self.extract_head_feat -> pass by residual_bottleneck, 1024 -> 256
        # print("num_sequences", num_sequences) # 3
        # print("train_feat.shape 3", train_feat.shape) # torch.Size([2, 2, 256, h, w])
        # print("test_feat.shape 3", test_feat.shape) # torch.Size([1, 2, 256, h, w])
        # train_feat.shape 3 torch.Size([2, 2, 256, h, w])
        # test_feat.shape 3 torch.Size([1, 2, 256, h, w])
        

        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        # print("cls_filter.shape", cls_filter.shape) # torch.Size([B, 256, 1, 1])
        # print("breg_filter.shape", breg_filter.shape) # torch.Size([B, 256, 1, 1])
        # print("test_feat_enc.shape 2", test_feat_enc.shape) # torch.Size([1, B, 256, h, w])

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)
        # print("target_scores.shape", target_scores.shape) # torch.Size([1, B, h, w])

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)
        # print("bbox_preds.shape", bbox_preds.shape) # torch.Size([1, 3, 4, h, w])
        

        return target_scores, bbox_preds


    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            # print("self.feature_extractor is None") # no pass
            return feat
        if num_sequences is None:
            # print("num_sequences is Non") # no pass
            return self.feature_extractor(feat)
            
        # print("feat.shape", feat.shape) # [B,1024,h,w]

        output = self.feature_extractor(feat)

        # print("output.shape", output.shape) # [6,256,h,w]

        return output.reshape(-1, num_sequences, *output.shape[-3:])


    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):

        # print("train_label.shape", train_label.shape) # train_label.shape torch.Size([2, B, 22, 22])

        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights
            
        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
        # print("num_gth_frames", num_gth_frames)

        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc






class LinearFilterClassifier(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, feat, filter):
        # see filter.py
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter
        return filter_layer.apply_filter(feat, filter_proj)




class DenseBoxRegressor(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        
        self.project_filter = project_filter

        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

        layers = []
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        self.tower = nn.Sequential(*layers)

        self.bbreg_layer = nn.Conv2d(num_channels, 4, kernel_size=3, dilation=1, padding=1)

    def forward(self, feat, filter):
        # print("self.num_channels", self.num_channels)
        nf, ns, c, h, w = feat.shape

        if self.project_filter:
            filter_proj = self.linear(filter.reshape(ns, c)).reshape(ns, c, 1, 1)
        else:
            filter_proj = filter

        # print("filter_proj", filter_proj) #
        

        attention = filter_layer.apply_filter(feat, filter_proj) # (nf, ns, h, w)
        feats_att = attention.unsqueeze(2)*feat # (nf, ns, c, h, w)

        feats_tower = self.tower(feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1])) # (nf*ns, c, h, w)

        ltrb = torch.exp(self.bbreg_layer(feats_tower)).unsqueeze(0) # (nf*ns, 4, h, w)
        
        return ltrb


### for Transformer
###### for ResidualAttentionBlock
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
###### for ResidualAttentionBlock


############################################ PiVOT

import torch
import torch.nn as nn

class test_fuse(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()

        self.wo_shr = 1
        self.CNN_or_trans = 0
        self.train_feat_mlp = 0
        self.train_mlp_res = 0

        if not self.wo_shr and not self.CNN_or_trans:
            self.mlp_shrink2 = nn.Sequential(
                nn.Conv2d(256*3, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.GELU()
            )
        elif self.wo_shr and not self.CNN_or_trans:
            self.mlp_shrink2 = nn.Sequential(
                nn.Conv2d(1024*3, 1024, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.GELU()
            )

        if self.train_feat_mlp or self.train_mlp_res:
            self.mlp1 = nn.Sequential(
                nn.Conv2d(1024, 1024, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.GELU()
            )
            self.gelu = nn.GELU()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("test_fuse _reset_parameters done")
        print("self.train_feat_mlp", self.train_feat_mlp)
        print("self.train_mlp_res", self.train_mlp_res)
        
    def forward(self, test_fuse_feat):
        [train_feat, test_feat] = test_fuse_feat
        if self.train_feat_mlp:
            train_feat = self.mlp1(train_feat)

        if self.train_mlp_res:
            residual = train_feat
            train_feat = self.mlp1(train_feat)
            train_feat += residual
            train_feat = self.gelu(train_feat)

        bat = test_feat.shape[0]
        train1 = train_feat[0:bat]
        train2 = train_feat[bat:]
        test1 = test_feat

        if self.CNN_or_trans == 0:  # CNN
            cat_opt_test_opt_train = torch.cat([train1, train2, test1], dim=1)
            merge = self.mlp_shrink2(cat_opt_test_opt_train)

        return merge


class test_fuse_head(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()

        self.wo_shr = 1
        self.wlabel = 1
        self.interpolate_mode = 'nearest' 

        self.mlp_head = nn.Sequential(
            nn.Conv2d(1024 if self.wo_shr else 256, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.GELU()
        )

        self.hardtanh01 = nn.Hardtanh(0, 1)
        self.sigmoid = nn.Sigmoid()
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("test_fuse_head _reset_parameters done")
        
    def forward(self, merge):
        x_out = self.mlp_head(merge)
        x_out_small = self.hardtanh01(x_out) if self.wlabel else x_out

        scale_factor = 288 / 18
        interpolation_mode = 'nearest' if self.interpolate_mode == 'nearest' else 'bilinear'
        x_out = F.interpolate(x_out, scale_factor=scale_factor, mode=interpolation_mode, align_corners=True if interpolation_mode == 'bilinear' else None)
        
        return x_out, x_out_small



class test_shrink(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()

        # Version 2
        self.mlp_shrink = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        # Version 3
        self.mlp_shrink_t = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("test_shrink _reset_parameters done")
        
    def forward(self, test_fuse_feat):
        [train_feat, test_feat] = test_fuse_feat
        train_feat = self.mlp_shrink(train_feat)
        test_feat = self.mlp_shrink_t(test_feat)
        return [train_feat, test_feat]




# RMv1
class hint_test_feat_head(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()

        self.mlp_shrink = nn.Sequential(
            nn.Conv2d(1024*2, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.GELU(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("hint_test_feat_head _reset_parameters done")
        
    def forward(self, test_feat_head, prob_map_exp):
        cat_feat = torch.cat((test_feat_head, prob_map_exp), 1)
        out = self.mlp_shrink(cat_feat)
        return out


# bkMlpv1
class bkMlp_(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, bias=False),

            nn.BatchNorm2d(1024),

            nn.GELU(),
        )


        self.gelu = nn.GELU()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("bkMlpv1 _reset_parameters done")
        
    def forward(self, bk_feat):

        # print("bk_feat.shape", bk_feat.shape)
        
        out = self.mlp(bk_feat)
        # print("out.shape", out.shape)
        

        return out


# bkMlpresdv1
class bkMlp_(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(bkMlp_.ResidualBlock, self).__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(1024, 1024, 1, bias=False),
                nn.BatchNorm2d(1024)
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return nn.GELU()(out)

    def __init__(self):
        super(bkMlp_, self).__init__()
        self.residual_cnn = nn.Sequential(
            self.ResidualBlock(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("bkMlpresdv1 _reset_parameters done")

    def forward(self, bk_feat):
        out = self.residual_cnn(bk_feat)
        return out




############################################ PiVOT


class TF_mlp(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("TF_mlp _reset_parameters done")
        
    def forward(self, bk_feat):
        """
        Args:
            bk_feat: Tensor of shape (1, B, C, H, W)
        
        Returns:
            Tensor of shape (1, B, C, H, W) after processing.
        """
        B, C, H, W = bk_feat.shape[1:]  # Extract dimensions
        
        # Reshape to (1*B, C, H, W) before passing to MLP
        x = bk_feat.view(-1, C, H, W)
        
        # Pass through MLP
        x = self.mlp(x)
        
        # Reshape back to (1, B, C, H, W)
        x = x.view(1, B, C, H, W)
        
        return x



############################################ ToMP_JEPA_PT

### PT Head_PT
class Head_PT(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False,
                 ):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

        self.auxPTcurloss = True
        # self.auxPTcurloss = False
        print("self.auxPTcurloss", self.auxPTcurloss)

        # self.auxPTtrackloss = True
        self.auxPTtrackloss = False
        print("self.auxPTtrackloss", self.auxPTtrackloss)

        self.TFwPTT = False   # testFeat_w_PTfeat # train 
        # self.TFwPTT = True   # testFeat_w_PTtrack # infer  # train 

        print("self.TFwPTT Head init", self.TFwPTT)

        # self.TFEcatAttn = False
        self.TFEcatAttn = True
        print("self.TFEcatAttn", self.TFEcatAttn)

        print("Head_PT")

    def forward(self, 
                train_feat, 
                test_feat, 
                train_bb, 
                test_tracks_transformed, 
                PTrackAttentionModel,
                TFEcatmlp,
                JEPA_predictor_cls, 
                JEPA_predictor_breg,
                *args, **kwargs):
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        
        ### note

        # print("self.separate_filters_for_cls_and_bbreg", self.separate_filters_for_cls_and_bbreg) # False

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        # train_feat.shape 1 torch.Size([B,1024, h, w])
        # test_feat.shape 1 torch.Size([B,1024, h, w])

        
        ### note

        
        
        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        # heat_show(test_feat[0][0][0], "test_feat[0][0][0]")
        # heat_show(test_feat[0][0][100], "test_feat[0][0][100]")
        # heat_show(test_feat[0][0][200], "test_feat[0][0][200]")

        ### note
        # self.extract_head_feat -> pass by residual_bottleneck, 1024 -> 256
        # print("num_sequences", num_sequences) # 3
        # print("train_feat.shape 3", train_feat.shape) # torch.Size([2, 2, 256, h, w])
        # print("test_feat.shape 3", test_feat.shape) # torch.Size([1, 2, 256, h, w])
        # train_feat.shape 3 torch.Size([2, 2, 256, h, w])
        # test_feat.shape 3 torch.Size([1, 2, 256, h, w])

        
        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        cls_filter = JEPA_predictor_cls(cls_filter)
        breg_filter = JEPA_predictor_breg(breg_filter)

        # heat_show(test_feat_enc[0][0][0], "test_feat_enc[0][0][0]")
        # heat_show(test_feat_enc[0][0][100], "test_feat_enc[0][0][100]")
        # heat_show(test_feat_enc[0][0][200], "test_feat_enc[0][0][200]")

        # print("test_feat_enc.shape 1", test_feat_enc.shape) # torch.Size([1, 3, 256, h, w])

        # PT

        PT_promPTweight = PTrackAttentionModel(test_tracks_transformed, test_feat, self.TFwPTT) # F test_feat T track
        # print("PT_promPTweight.shape", PT_promPTweight.shape, PT_promPTweight.requires_grad) # True   [1, B, 256, H, W]

        ###################################### Using PT cur_feat to predict

        if self.auxPTcurloss:
            target_scores_PT = self.classifier(PT_promPTweight, cls_filter)
            bbox_preds_PT = self.bb_regressor(PT_promPTweight, breg_filter)
        
        if self.auxPTtrackloss:
            # Create cls_filter_PT and breg_filter_PT on the same device as cls_filter and breg_filter
            cls_filter_PT = torch.ones(cls_filter.shape[0], 256, 1, 1).to(cls_filter.device)
            breg_filter_PT = torch.ones(breg_filter.shape[0], 256, 1, 1).to(breg_filter.device)

            target_scores_PT = self.classifier(PT_promPTweight, cls_filter_PT)
            bbox_preds_PT = self.bb_regressor(PT_promPTweight, breg_filter_PT)

        if not self.TFEcatAttn:
            test_feat_enc = TFEcatmlp(torch.cat([test_feat_enc.squeeze(0), PT_promPTweight.squeeze(0)], dim=1)).unsqueeze(0)
        else:
            test_feat_enc = TFEcatmlp(test_feat_enc, PT_promPTweight).unsqueeze(0)

        # PT
        
        # print("test_feat_enc.shape 2", test_feat_enc.shape, test_feat_enc.requires_grad) # True
        # print("test_feat_enc.shape 2", test_feat_enc.shape) # torch.Size([1, 3, 256, h, w])

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)

        # heat_show(target_scores[0][0], "target_scores[0][0]")

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

        # return target_scores, bbox_preds
        return target_scores, bbox_preds, target_scores_PT, bbox_preds_PT



    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            # print("self.feature_extractor is None") # no pass
            return feat
        if num_sequences is None:
            # print("num_sequences is Non") # no pass
            return self.feature_extractor(feat)

        # print("feat.shape", feat.shape) # [B,1024,h,w]

        output = self.feature_extractor(feat)

        # print("output.shape", output.shape) # [6,256,h,w]

        return output.reshape(-1, num_sequences, *output.shape[-3:])

        
    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):

        # print("train_label.shape", train_label.shape) # train_label.shape torch.Size([2, B, 22, 22])

        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights
            
        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):

        # print("num_gth_frames", num_gth_frames) # 

        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc


### Head_JEPAs2 (wP) tompnet_JEPAp  / VGGT_early_fuse_True
class Head_(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False,
                 ):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

        self.permute = 1

        # print("Head_JEPAs2 norm", self.norm)

        print("Head_JEPAs2")

    def forward(self, train_feat, test_feat, train_bb, 
                JEPA_predictor_cls, JEPA_predictor_breg,
                auto_cast_full,dtype,infer_bf16,
                *args, **kwargs):
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        
        ### note

        # print("self.separate_filters_for_cls_and_bbreg", self.separate_filters_for_cls_and_bbreg) # False

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        # train_feat.shape 1 torch.Size([B,1024, h, w])
        # test_feat.shape 1 torch.Size([B,1024, h, w])

        
        ### note

        
        
        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        ### note
        # self.extract_head_feat -> pass by residual_bottleneck, 1024 -> 256
        # print("num_sequences", num_sequences) # 3
        # print("train_feat.shape 3", train_feat.shape) # torch.Size([2, 2, 256, h, w])
        # print("test_feat.shape 3", test_feat.shape) # torch.Size([1, 2, 256, h, w])
        # train_feat.shape 3 torch.Size([2, 2, 256, h, w])
        # test_feat.shape 3 torch.Size([1, 2, 256, h, w])

        
        # Train filter
        # print("self.auto_cast_full", self.auto_cast_full)
        # print("self.training", self.training)
        # print("self.infer_bf16", self.infer_bf16)

        # cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(auto_cast_full, dtype, infer_bf16, train_feat, test_feat, *args, **kwargs)

        cls_filter = JEPA_predictor_cls(cls_filter)
        breg_filter = JEPA_predictor_breg(breg_filter)

        # print("cls_filter.shape", cls_filter.shape) # torch.Size([B, 256, 1, 1])
        # print("breg_filter.shape", breg_filter.shape) # torch.Size([B, 256, 1, 1])

        # print("test_feat_enc.shape 2", test_feat_enc.shape) # torch.Size([1, B, 256, h, w])

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)
        # print("target_scores.shape", target_scores.shape) # torch.Size([1, B, h, w])

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)
        # print("bbox_preds.shape", bbox_preds.shape) # torch.Size([1, 3, 4, h, w])

        

        return target_scores, bbox_preds

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            # print("self.feature_extractor is None") # no pass
            return feat
        if num_sequences is None:
            # print("num_sequences is Non") # no pass
            return self.feature_extractor(feat)

        # print("feat.shape", feat.shape) # [B,1024,h,w]

        output = self.feature_extractor(feat)

        # print("output.shape", output.shape) # [6,256,h,w]

        return output.reshape(-1, num_sequences, *output.shape[-3:])

        
    # def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):
    def get_filter_and_features(self, auto_cast_full, dtype, infer_bf16, train_feat, test_feat, train_label, *args, **kwargs):
        
        # print("train_label.shape", train_label.shape) # train_label.shape torch.Size([2, B, h, w])

        if (auto_cast_full and self.training) or (auto_cast_full and infer_bf16):
            with torch.cuda.amp.autocast(dtype=dtype):       
                # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
                if self.separate_filters_for_cls_and_bbreg:
                    cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
                else:
                    weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
                    cls_weights = bbreg_weights = weights
        else:
            # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
            if self.separate_filters_for_cls_and_bbreg:
                cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            else:
                weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
                cls_weights = bbreg_weights = weights
            
        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):

        # print("num_gth_frames", num_gth_frames) # 

        print("self.net.auto_cast_full", self.net.auto_cast_full) # 

        if (self.net.auto_cast_full and self.training) or (self.net.auto_cast_full and self.infer_bf16):
            with torch.cuda.amp.autocast(dtype=self.net.dtype):       
                cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
                    = self.filter_predictor.predict_cls_bbreg_filters_parallel(
                    train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
                )
        else:
            cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
                = self.filter_predictor.predict_cls_bbreg_filters_parallel(
                train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
            )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc




### Head_JEPAs2 (wP) Head_bf16
class Head_bf16(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False,
                 ):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

        self.permute = 1

        # self.auto_cast_full = True
        self.auto_cast_full = False
        print("self.auto_cast_full head", self.auto_cast_full)  #

        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32

        print("Head_JEPAs2")

    def forward(self, train_feat, test_feat, \
                train_bb, \
                JEPA_predictor_cls, JEPA_predictor_breg, \
                auto_cast_full_all, auto_cast_full, dtype, infer_bf16, \
                *args, **kwargs):

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        
        ### note

        # print("self.separate_filters_for_cls_and_bbreg", self.separate_filters_for_cls_and_bbreg) # False

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]

        
        ### note

        
        
        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        # print("train_feat.shape 1", train_feat.shape) # torch.Size([2, B, 256, H, W])
        # print("test_feat.shape 1", test_feat.shape) # torch.Size([1, B, 256, H, W])

        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(auto_cast_full, self.dtype, infer_bf16, train_feat, test_feat, *args, **kwargs)

        if auto_cast_full_all:
            with torch.cuda.amp.autocast(dtype=self.dtype):
                cls_filter = JEPA_predictor_cls(cls_filter)
                breg_filter = JEPA_predictor_breg(breg_filter)
                # print("cls_filter.shape", cls_filter.shape) # torch.Size([B, 256, 1, 1])
                # print("breg_filter.shape", breg_filter.shape) # torch.Size([B, 256, 1, 1])

                # print("test_feat_enc.shape 2", test_feat_enc.shape) # torch.Size([1, B, 256, h, w])

                # fuse encoder and decoder features to one feature map
                target_scores = self.classifier(test_feat_enc, cls_filter)
                # print("target_scores.shape", target_scores.shape) # torch.Size([1, B, h, w])

                # compute the final prediction using the output module
                bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)
                # print("bbox_preds.shape", bbox_preds.shape) # torch.Size([1, 3, 4, h, w])
        else:
            cls_filter = JEPA_predictor_cls(cls_filter)
            breg_filter = JEPA_predictor_breg(breg_filter)

            # print("cls_filter.shape", cls_filter.shape) # torch.Size([B, 256, 1, 1])
            # print("breg_filter.shape", breg_filter.shape) # torch.Size([B, 256, 1, 1])

            # print("test_feat_enc.shape 2", test_feat_enc.shape) # torch.Size([1, B, 256, h, w])

            # fuse encoder and decoder features to one feature map
            target_scores = self.classifier(test_feat_enc, cls_filter)
            # print("target_scores.shape", target_scores.shape) # torch.Size([1, B, h, w])

            # compute the final prediction using the output module
            bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

            # print("bbox_preds.shape", bbox_preds.shape) # torch.Size([1, 3, 4, h, w])

        

        return target_scores, bbox_preds



    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            # print("self.feature_extractor is None") # no pass
            return feat
        if num_sequences is None:
            # print("num_sequences is Non") # no pass
            return self.feature_extractor(feat)

        # print("feat.shape", feat.shape) # [B,1024,h,w]

        output = self.feature_extractor(feat)

        # print("output.shape", output.shape) # [6,256,h,w]

        return output.reshape(-1, num_sequences, *output.shape[-3:])

        
   
    # def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):
    def get_filter_and_features(self, auto_cast_full, dtype, infer_bf16, train_feat, test_feat, train_label, *args, **kwargs):
        
        # print("train_label.shape", train_label.shape) # train_label.shape torch.Size([2, B, h, w])

        if (auto_cast_full and self.training) or (auto_cast_full and infer_bf16):
            with torch.cuda.amp.autocast(dtype=self.dtype):       
                # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
                if self.separate_filters_for_cls_and_bbreg:
                    cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
                else:
                    weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
                    cls_weights = bbreg_weights = weights
        else:
            # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
            if self.separate_filters_for_cls_and_bbreg:
                cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            else:
                weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
                cls_weights = bbreg_weights = weights
            
        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):

        # print("num_gth_frames", num_gth_frames) # 

        # print("self.auto_cast_full", self.auto_cast_full) # 

        if self.auto_cast_full:
            with torch.cuda.amp.autocast(dtype=self.dtype):       
                cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
                    = self.filter_predictor.predict_cls_bbreg_filters_parallel(
                    train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
                )
        else:
            cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
                = self.filter_predictor.predict_cls_bbreg_filters_parallel(
                train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
            )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc




class Head_JEPA_context(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False,
                 ):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

        self.permute = 1
        print("Head_JEPA_context")

    def forward(self, train_feat, test_feat, train_bb, 
                *args, **kwargs):
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        
        ### note

        # print("self.separate_filters_for_cls_and_bbreg", self.separate_filters_for_cls_and_bbreg) # False

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        # train_feat.shape 1 torch.Size([B,1024, h, w])
        # test_feat.shape 1 torch.Size([B,1024, h, w])

        
        ### note

        
        
        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        ### note
        # self.extract_head_feat -> pass by residual_bottleneck, 1024 -> 256
        # print("num_sequences", num_sequences) # 3
        # print("train_feat.shape 3", train_feat.shape) # torch.Size([2, 2, 256, h, w])
        # print("test_feat.shape 3", test_feat.shape) # torch.Size([1, 2, 256, h, w])
        # train_feat.shape 3 torch.Size([2, 2, 256, h, w])
        # test_feat.shape 3 torch.Size([1, 2, 256, h, w])

        
        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        # print("cls_filter.shape", cls_filter.shape) # torch.Size([B, 256, 1, 1])
        # print("breg_filter.shape", breg_filter.shape) # torch.Size([B, 256, 1, 1])

        # print("test_feat_enc.shape 2", test_feat_enc.shape) # torch.Size([1, B, 256, h, w])

        # fuse encoder and decoder features to one feature map
        # target_scores = self.classifier(test_feat_enc, cls_filter)
        # print("target_scores.shape", target_scores.shape) # torch.Size([1, B, h, w])

        # compute the final prediction using the output module
        # bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)
        # print("bbox_preds.shape", bbox_preds.shape) # torch.Size([1, 3, 4, h, w])

        

        return cls_filter, breg_filter



    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            # print("self.feature_extractor is None") # no pass
            return feat
        if num_sequences is None:
            # print("num_sequences is Non") # no pass
            return self.feature_extractor(feat)

        # print("feat.shape", feat.shape) # [B,1024,h,w]

        output = self.feature_extractor(feat)

        # print("output.shape", output.shape) # [6,256,h,w]

        return output.reshape(-1, num_sequences, *output.shape[-3:])


    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):

        # print("train_label.shape", train_label.shape) # train_label.shape torch.Size([2, B, 22, 22])

        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights
            
        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):

        # print("num_gth_frames", num_gth_frames) # 

        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc




### tompnet_GOTPT_JEPA_s2 GOTPT_JEPA_PTAttadd_initMPJEPAnHead_s2
class Head_(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False,
                 ):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

        self.TFwPTT = False   # testFeat_w_PTfeat # train auxPTcurloss2x2
        # self.TFwPTT = True   # testFeat_w_PTtrack # infer  # train auxPTcurloss2x2_TFwPTT1_hclsw_s40

        print("self.TFwPTT Head init", self.TFwPTT)

        self.PT_cur_TEnc = True
        # self.PT_cur_TEnc = False

        self.PTAttadd = True
        # self.PTAttadd = False

        print("self.PT_cur_TEnc", self.PT_cur_TEnc)

        print("tompnet_GOTPT_JEPA_s2")

    def forward(self, 
                train_feat, 
                test_feat, 
                train_bb, 
                test_tracks_transformed, # torch.Size([1, B, 256, h, w])
                PTrackAttentionModel,
                JEPA_predictor_cls, 
                JEPA_predictor_breg,
                # TF_mlp,
                *args, **kwargs):

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])
        
        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences) # torch.Size([2, B, 256, h, w])
        test_feat = self.extract_head_feat(test_feat, num_sequences) # torch.Size([1, B, 256, h, w])

        if self.PT_cur_TEnc and self.PTAttadd:
            test_feat_ori = test_feat.clone()

        if self.PT_cur_TEnc:
            test_feat = PTrackAttentionModel(test_tracks_transformed, test_feat, self.TFwPTT) # [1, B, 256, H, W]

        if self.PT_cur_TEnc and self.PTAttadd:
            test_feat = test_feat_ori + test_feat

        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        # TFEaddTF
        # test_feat_enc = test_feat_enc + TF_mlp(test_feat)

        cls_filter_p = JEPA_predictor_cls(cls_filter)
        breg_filter_p = JEPA_predictor_breg(breg_filter)

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter_p)

        # heat_show(target_scores[0][0], "target_scores[0][0]")

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter_p)

        return target_scores, bbox_preds


    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            # print("self.feature_extractor is None") # no pass
            return feat
        if num_sequences is None:
            # print("num_sequences is Non") # no pass
            return self.feature_extractor(feat)

        # print("feat.shape", feat.shape) # [B,1024,h,w]

        output = self.feature_extractor(feat)

        # print("output.shape", output.shape) # [6,256,h,w]

        return output.reshape(-1, num_sequences, *output.shape[-3:])

        
    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):

        # print("train_label.shape", train_label.shape) # train_label.shape torch.Size([2, B, 22, 22])

        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights
            
        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):

        # print("num_gth_frames", num_gth_frames) # 

        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc





class Head_JEPA_target(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, 
                 classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False,
                 ):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

        self.permute = 1

        # Set requires_grad to False for all parameters
        for param in self.parameters():
            param.requires_grad = False

        print("Head_JEPA_target")

    def forward(self, train_feat, test_feat, train_bb, 
                *args, **kwargs):
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        
        ### note

        # print("self.separate_filters_for_cls_and_bbreg", self.separate_filters_for_cls_and_bbreg) # False

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # ### note
        # print("train_feat.shape 1", train_feat.shape) # [B,1024,h,w]
        # print("test_feat.shape 1", test_feat.shape) # [B,1024,h,w]
        # train_feat.shape 1 torch.Size([B,1024, h, w])
        # test_feat.shape 1 torch.Size([B,1024, h, w])

        
        ### note

        
        
        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        ### note
        # self.extract_head_feat -> pass by residual_bottleneck, 1024 -> 256
        # print("num_sequences", num_sequences) # 3
        # print("train_feat.shape 3", train_feat.shape) # torch.Size([2, 2, 256, h, w])
        # print("test_feat.shape 3", test_feat.shape) # torch.Size([1, 2, 256, h, w])
        # train_feat.shape 3 torch.Size([2, 2, 256, h, w])
        # test_feat.shape 3 torch.Size([1, 2, 256, h, w])

        
        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        # print("cls_filter.shape", cls_filter.shape) # torch.Size([B, 256, 1, 1])
        # print("breg_filter.shape", breg_filter.shape) # torch.Size([B, 256, 1, 1])

        # print("test_feat_enc.shape 2", test_feat_enc.shape) # torch.Size([1, B, 256, h, w])

        # fuse encoder and decoder features to one feature map
        # target_scores = self.classifier(test_feat_enc, cls_filter)
        # print("target_scores.shape", target_scores.shape) # torch.Size([1, B, h, w])

        # compute the final prediction using the output module
        # bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)
        # print("bbox_preds.shape", bbox_preds.shape) # torch.Size([1, 3, 4, h, w])

        

        return cls_filter, breg_filter



    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            # print("self.feature_extractor is None") # no pass
            return feat
        if num_sequences is None:
            # print("num_sequences is Non") # no pass
            return self.feature_extractor(feat)

        # print("feat.shape", feat.shape) # [B,1024,h,w]

        output = self.feature_extractor(feat)

        # print("output.shape", output.shape) # [6,256,h,w]

        return output.reshape(-1, num_sequences, *output.shape[-3:])

        
    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):

        # print("train_label.shape", train_label.shape) # train_label.shape torch.Size([2, B, 22, 22])

        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights
            
        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):

        # print("num_gth_frames", num_gth_frames) # 

        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc



# DA3_bkMlp_B
class DA3_bkMlp_B(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            # Preserves  'MLP' style (kernel_size=1)
            # If  need spatial context, change kernel_size to 3 and padding to 1
            self.conv_block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.activation = nn.GELU()

        def forward(self, x):
            identity = x
            out = self.conv_block(x)
            out = out + identity
            out = self.activation(out)
            return out

    def __init__(self):
        super(DA3_bkMlp, self).__init__()

        self.net = nn.Sequential(
            # Stage 1: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.ResidualBlock(128), # Refine at 128

            # Stage 2: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            self.ResidualBlock(256)  # Refine at 256
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("Net64to256 initialized")

    def forward(self, x):
        """
        x: (B, 64, H, W)
        returns: (B, 256, H, W)
        """
        return self.net(x)

# DA3_bkMlp_L
class DA3_bkMlp(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(DA3_bkMlp.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
            )
            self.skip_connection = nn.Identity()
            self.activation = nn.GELU()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out = out + identity
            out = self.activation(out)
            return out

    def __init__(self):
        super(DA3_bkMlp, self).__init__()

        # 128 -> 256 + residual refinement
        self.proj = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            self.ResidualBlock(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("DA3_bkMlp _reset_parameters done")

    def forward(self, x):
        """
        x: (B, 128, H, W) or (B*S, 128, H, W)
        returns:
            (B, 256, H, W) or (B*S, 256, H, W) with the same spatial size
        """
        return self.proj(x)


# vggtDPTfeatMlp
class vggtDPTfeatMlp(nn.Module):
    """
    Revised network to process 5D feature tensors with a focus on:
    - Retaining more original features by using more gradual downsampling.
    - Addressing OOM issues by potentially reducing intermediate feature map sizes.
    - Replacing the pooling layer in the skip connection with a strided convolution.
    Output shape is (B, S, 1024, dino_patch_size, dino_patch_size).
    """

    class BottleneckBlock(nn.Module):
        expansion = 4

        def __init__(self, in_channels, bottleneck_channels, stride=1):
            super().__init__()
            out_channels = self.expansion * bottleneck_channels
            self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(bottleneck_channels)
            self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(bottleneck_channels)
            self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.activation = nn.GELU()

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            identity = self.shortcut(x)
            out = self.activation(self.bn1(self.conv1(x)))
            out = self.activation(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += identity
            out = self.activation(out)
            return out

    class StridedConvSkipConnection(nn.Module):
        """Replaces AvgPool with a strided convolution for downsampling in skip connection."""
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.activation = nn.GELU()

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.activation(x)
            return x

    class FeatureRefinementBlock(nn.Module):
        """A simple block for refining feature maps (e.g., after concatenation)."""
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(channels)
            self.activation = nn.GELU()

        def forward(self, x):
            return self.activation(self.bn(self.conv(x)))

    def __init__(self, dino_patch_size=DinoPatch):
        super().__init__()
        self.dino_patch_size = dino_patch_size
        target_channels = 1024
        reduction_factor = (DinoPatch*8) // dino_patch_size

        # if reduction_factor <= 0 or 144 % dino_patch_size != 0:
        #     raise ValueError(f"Input size 144 must be divisible by dino_patch_size {dino_patch_size}")

        # --- Gradual Downsampling Paths ---
        # Path for input (B, S, 256, 144, 144)
        self.path256_stages = nn.Sequential(
            self.BottleneckBlock(256, 64, stride=2),      # -> (B*S, 256, 72, 72)
            self.BottleneckBlock(256, 128, stride=2),     # -> (B*S, 512, 36, 36)
            self.BottleneckBlock(512, 256, stride=2),     # -> (B*S, 1024, h, w)
        )
        self.path256_skip = self.StridedConvSkipConnection(256, target_channels, stride=reduction_factor)
        self.path256_fusion = self.FeatureRefinementBlock(target_channels)

        self._reset_parameters()
        print(f"Refined vggtDPTfeatMlp initialized for dino_patch_size={dino_patch_size}.")

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, vggtDPTfeat_type):
        if x.ndim != 5:
            raise ValueError(f"Input tensor must be 5D (B, S, C, H, W), but got {x.ndim}D")

        B, S, C, H, W = x.size()
        # if H != 144 or W != 144:
        #     raise ValueError(f"Expected input spatial dimensions (144, 144), but got ({H}, {W})")

        #print("x.shape", x.shape) # 
        

        reshaped_x = x.view(B * S, C, H, W)
        identity = reshaped_x # For the skip connection

        if vggtDPTfeat_type == "fl1":
            main_out = self.path256_stages(reshaped_x)
            skip_out = self.path256_skip(identity)
            fused_out = self.path256_fusion(main_out + skip_out)
        else:
            raise ValueError(f"Unsupported input channel size: {C}. Expected 128 or 256.")

        #print("fused_out.shape", fused_out.shape) # 
        

        return fused_out.view(B, S, 1024, self.dino_patch_size, self.dino_patch_size)



class vggtDPTfeatMlp_head(nn.Module):
    """
    A network module to reduce feature channels from 1024 to 256,
    while preserving spatial dimensions (H, W).

    Input shape: (B, 1024, H, W)
    Output shape: (B, 256, H, W)
    """
    def __init__(self):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(num_features=256),
            nn.GELU()
        )
        self._reset_parameters()
        print("vggtDPTfeatMlp_head initialized: 1024 channels -> 256 channels (spatial dims preserved).")

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: # Should not happen with bias=False, but good practice
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to reduce channel dimensions.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1024, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, 256, H, W).
        """
        if x.ndim != 4:
            raise ValueError(f"Input tensor must be 4D (B, C, H, W), but got {x.ndim}D")
        if x.size(1) != 1024:
            raise ValueError(f"Input tensor must have 1024 channels, but got {x.size(1)} channels")

        return self.reducer(x)



class LabelExpNet(nn.Module):
    """
    A network module to reshape and expand channels of two input tensors.

    Input 1: (2, B, H, W) -> reshaped to (2*B, 1, H, W) -> expanded to (2*B, 256, H, W)
    Input 2: (2, B, 4, H, W) -> reshaped to (2*B, 4, H, W) -> expanded to (2*B, 256, H, W)
    """
    def __init__(self):
        """
        Initializes the convolutional layers and applies weight initialization.
        """
        super().__init__()

        # Convolutional layer to expand channels for the first input (label)
        # Input channels: 1 (after unsqueeze)
        # Output channels: 256
        # Kernel size: 1x1 to only change channels, not spatial dimensions
        self.label_channel_expander = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1, bias=True)
        # self.label_channel_expander = nn.Conv2d(in_channels=1, out_channels=1024, kernel_size=1, bias=True)

        # Convolutional layer to expand channels for the second input (ltrb)
        # Input channels: 4
        # Output channels: 256
        # Kernel size: 1x1 to only change channels, not spatial dimensions
        self.ltrb_channel_expander = nn.Conv2d(in_channels=4, out_channels=256, kernel_size=1, bias=True)
        # self.ltrb_channel_expander = nn.Conv2d(in_channels=4, out_channels=1024, kernel_size=1, bias=True)

        # Initialize weights
        self._reset_parameters()
        print("LabelExpNet initialized")

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: # Should not happen with bias=False, but good practice
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, vggt_train_label_enc: torch.Tensor, vggt_train_ltrb_target_enc: torch.Tensor):
        """
        Performs the forward pass.

        Args:
            vggt_train_label_enc: The first input tensor (2, B, H, W).
            vggt_train_ltrb_target_enc: The second input tensor (2, B, 4, H, W).

        Returns:
            A tuple containing two tensors:
            - The expanded label tensor (2*B, 256, H, W).
            - The expanded ltrb tensor (2*B, 256, H, W).
        """
        # Get dimensions
        dim2, B, H, W = vggt_train_label_enc.shape
        # assert dim2 == 2, "First dimension of vggt_train_label_enc must be 2"
        # assert vggt_train_ltrb_target_enc.shape[:3] == (2, B, 4), \
        #     "Shape mismatch for vggt_train_ltrb_target_enc"
        # assert vggt_train_ltrb_target_enc.shape[3:] == (H, W), \
        #     "Spatial dimensions mismatch between inputs"

        # print("kwargs['train_label'].shape:", kwargs['train_label'].shape) # torch.Size([2, B, H, W])
        # print("kwargs['train_ltrb_target'].shape:", kwargs['train_ltrb_target'].shape) # torch.Size([2, B, 4, H, W])

        # Reshape and add channel for the first input (label)
        # (2, B, H, W) -> (2*B, H, W)
        reshaped_label = vggt_train_label_enc.reshape(dim2 * B, H, W)
        # (2*B, H, W) -> (2*B, 1, H, W) - add channel dimension at dim 1
        reshaped_label = reshaped_label.unsqueeze(1)

        # Reshape the second input (ltrb)
        # (2, B, 4, H, W) -> (2*B, 4, H, W)
        reshaped_ltrb = vggt_train_ltrb_target_enc.reshape(dim2 * B, 4, H, W)

        # Expand channels using convolutional layers
        expanded_label = self.label_channel_expander(reshaped_label)
        expanded_ltrb = self.ltrb_channel_expander(reshaped_ltrb)

        return expanded_label, expanded_ltrb

        

# bkMlpresdv2_B
class bkMlp_(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(bkMlp.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(768, 768, 1, bias=False),
                nn.BatchNorm2d(768)
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return nn.GELU()(out)

    def __init__(self):
        super(bkMlp, self).__init__()

        self.residual_cnn = nn.Sequential(
            nn.Conv2d(768, 768, 1),  # Reduction layer
            nn.BatchNorm2d(768),
            nn.GELU(),
            self.ResidualBlock(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("bkMlpresdv2 _reset_parameters done")


    def forward(self, bk_feat):
        # Processing through residual CNN layers
        out = self.residual_cnn(bk_feat)
        return out


# bkMlpresdv2_L
class bkMlp(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(bkMlp.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(1024, 1024, 1, bias=False),
                nn.BatchNorm2d(1024)
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return nn.GELU()(out)

    def __init__(self):
        super(bkMlp, self).__init__()

        self.residual_cnn = nn.Sequential(
            nn.Conv2d(1024, 1024, 1),  # Reduction layer
            nn.BatchNorm2d(1024),
            nn.GELU(),
            self.ResidualBlock(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("bkMlpresdv2 _reset_parameters done")


    def forward(self, bk_feat):
        # Processing through residual CNN layers
        out = self.residual_cnn(bk_feat)
        return out


class bkMlpMOEv2(nn.Module):
    """
    Layer-gated TMoE fusion:
      - per-layer 1x1 alignment + BN + GELU
      - sample-wise soft gate across K layers (K = len(feats) at runtime)
      - post adapter = 1x1 conv + token-wise low-rank MoE residual
    Input:  list of K tensors [B, C, H, W]
    Output: tensor [B, C, H, W] with H,W taken from the first map
    """
    def __init__(self, channels: int = 1024, num_layers: int = 4,  # num_layers kept for compat
                 temperature: float = 1.0, r: int = 8, alpha: float = 8.0,
                 num_experts: int = 4, dropout: float = 0.0, use_shared_expert: bool = True,
                 max_layers: int = 24):
        super().__init__()
        self.C = int(channels)
        self.tau = max(float(temperature), 1e-6)
        self.r = int(r)
        self.alpha = float(alpha)
        self.num_experts = int(num_experts)
        self.use_shared = bool(use_shared_expert)
        self.max_layers = int(max_layers)
        self.moe_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # banks for up to max_layers; we will use the first K at runtime
        self.align = nn.ModuleList([nn.Conv2d(self.C, self.C, 1, bias=False) for _ in range(self.max_layers)])
        self.bn    = nn.ModuleList([nn.BatchNorm2d(self.C) for _ in range(self.max_layers)])

        # shared gate head and adapter
        self.gate = nn.Linear(self.C, 1, bias=False)
        self.adapt_conv = nn.Conv2d(self.C, self.C, 1, bias=True)
        self.adapt_bn   = nn.BatchNorm2d(self.C)
        self.adapt_act  = nn.GELU()

        # token-wise MoE (low-rank)
        self.compress = nn.Parameter(torch.empty(self.r, self.C))
        self.experts  = nn.ParameterList([nn.Parameter(torch.empty(self.C, self.r)) for _ in range(self.num_experts)])
        self.shared   = nn.Parameter(torch.empty(self.C, self.r)) if self.use_shared else None
        self.router   = nn.Linear(self.C, self.num_experts, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.compress, std=0.02)
        for e in self.experts:
            nn.init.trunc_normal_(e, std=0.02)
        if self.shared is not None:
            nn.init.trunc_normal_(self.shared, std=0.02)
        nn.init.trunc_normal_(self.router.weight, std=0.02)

        print("bkMlpMOEv2 _reset_parameters done")

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C]
        gates = self.router(x).softmax(dim=-1)            # [N, E]
        z = self.moe_drop(x) @ self.compress.t()          # [N, r]
        scale = self.alpha / float(self.r)
        out = (z @ self.shared.t()) * scale if self.shared is not None else 0.0
        for i in range(self.num_experts):
            wi = gates[:, i:i+1]                          # [N, 1]
            out = out + wi * ((z @ self.experts[i].t()) * scale)
        return out                                         # [N, C]

    def forward(self, feats):
        """
        feats: list of K tensors [B, C, H, W]
        returns: fused tensor [B, C, H, W] with H,W from feats[0]
        """
        assert isinstance(feats, (list, tuple)) and len(feats) > 0, "feats must be non-empty list"
        K = len(feats)
        assert K <= self.max_layers, f"K={K} > max_layers={self.max_layers}"
        B, C = feats[0].shape[:2]
        assert C == self.C, f"channel mismatch: got {C}, expected {self.C}"

        # ensure device/dtype match module params (avoids CPU/GPU or fp32/bf16 mismatches)
        ref = self.align[0].weight
        dev, dt = ref.device, ref.dtype
        feats = [x.to(device=dev, dtype=dt, non_blocking=True) for x in feats]

        Ht, Wt = feats[0].shape[-2:]            # spatial reference to preserve
        adapted, logits = [], []

        for i in range(K):
            x = feats[i]
            if x.shape[-2:] != (Ht, Wt):
                x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
            x = F.gelu(self.bn[i](self.align[i](x)))      # [B, C, Ht, Wt]
            adapted.append(x)
            gap = x.mean(dim=(-2, -1))                    # [B, C]
            logits.append(self.gate(gap))                 # [B, 1]

        stack  = torch.stack(adapted, dim=1)              # [B, K, C, Ht, Wt]
        logits = torch.stack(logits, dim=1)               # [B, K, 1]
        w = F.softmax(logits / self.tau, dim=1).view(B, K, 1, 1, 1)
        fused = (w * stack).sum(dim=1)                    # [B, C, Ht, Wt]  ← preserved

        # adapter: 1x1 + token-wise MoE residual (keeps H,W)
        y = self.adapt_conv(fused)                        # [B, C, Ht, Wt]
        N = B * Ht * Wt
        tokens = fused.permute(0, 2, 3, 1).reshape(N, C)  # [N, C]
        moe = self._moe(tokens).reshape(B, Ht, Wt, C).permute(0, 3, 1, 2).contiguous()

        y = y + moe
        y = self.adapt_act(self.adapt_bn(y))
        return y                                          # [B, C, H, W]

# bkMlpresdv2_G
class bkMlp_(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(bkMlp.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(1536, 1536, 1, bias=False),
                nn.BatchNorm2d(1536)
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return nn.GELU()(out)

    def __init__(self):
        super(bkMlp, self).__init__()

        self.residual_cnn = nn.Sequential(
            nn.Conv2d(1536, 1536, 1),  # Reduction layer
            nn.BatchNorm2d(1536),
            nn.GELU(),
            self.ResidualBlock(),
        )

        self.down = nn.Conv2d(1536, 1024, 1, bias=False)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("bkMlpresdv2 _reset_parameters done")


    def forward(self, bk_feat):
        # Processing through residual CNN layers
        out = self.residual_cnn(bk_feat)
        out = self.down(out)

        return out





# bkMlpresdv2_H
class bkMlp_H(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(bkMlp_H.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(1280, 1280, 1, bias=False),
                nn.BatchNorm2d(1280)
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return nn.GELU()(out)

    def __init__(self):
        super(bkMlp_H, self).__init__()

        self.residual_cnn = nn.Sequential(
            nn.Conv2d(1280, 1280, 1),  # Reduction layer
            nn.BatchNorm2d(1280),
            nn.GELU(),
            self.ResidualBlock(),
        )

        self.down = nn.Conv2d(1280, 1024, 1, bias=False)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("bkMlpresdv2 _reset_parameters done")


    def forward(self, bk_feat):
        # Processing through residual CNN layers
        out = self.residual_cnn(bk_feat)
        out = self.down(out)

        return out


# bkMlpresdv2_g
class bkMlp_g(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(bkMlp_g.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(1408, 1408, 1, bias=False),
                nn.BatchNorm2d(1408)
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return nn.GELU()(out)

    def __init__(self):
        super(bkMlp_g, self).__init__()

        self.residual_cnn = nn.Sequential(
            nn.Conv2d(1408, 1408, 1),  # Reduction layer
            nn.BatchNorm2d(1408),
            nn.GELU(),
            self.ResidualBlock(),
        )

        self.down = nn.Conv2d(1408, 1024, 1, bias=False)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("bkMlpresdv2 _reset_parameters done")


    def forward(self, bk_feat):
        # Processing through residual CNN layers
        out = self.residual_cnn(bk_feat)
        out = self.down(out)

        return out


# bkMlpresdv2_L
class bkMlp_vggt(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(bkMlp_vggt.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(1024, 1024, 1, bias=False),
                nn.BatchNorm2d(1024)
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return nn.GELU()(out)

    def __init__(self):
        super(bkMlp_vggt, self).__init__()

        self.residual_cnn = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),  # Reduction layer
            nn.BatchNorm2d(1024),
            nn.GELU(),
            self.ResidualBlock(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("bkMlp_vggt _reset_parameters done")

    def forward(self, bk_feat):
        # Processing through residual CNN layers
        out = self.residual_cnn(bk_feat)
        return out





# NTBCwshrMXpos
class SideNetwork_D(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(SideNetwork_D, self).__init__()
        self.feature_dim = feature_dim // 2
        self.transformer = nn.Transformer(
            d_model=self.feature_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        self.linear_down = nn.Conv1d(128, self.feature_dim, kernel_size=1)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.positional_encoding = PositionalEncoding1D(self.feature_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("SideNetwork_D NTBCwshrMXpos reset_parameters True")

    def forward(self, bk_feat):
        B, T, N, C = bk_feat.shape
        # Reshape and permute to (B, N, T, C) and then flatten N and T into a single dimension
        bk_feat = bk_feat.permute(0, 2, 1, 3).contiguous().view(B, N * T, C)
        
        # Now apply the linear down operation which expects input in the form (B, C, N*T)
        bk_feat = bk_feat.permute(0, 2, 1)
        bk_feat = self.linear_down(bk_feat)
 
 
        # Apply positional encoding adjusted for (B, C, N*T)
        pos_embeddings = self.positional_encoding(bk_feat).permute(0, 2, 1)
        bk_feat += pos_embeddings

        # Permute back to (N*T, B, C) for transformer input
        bk_feat = bk_feat.permute(2, 0, 1)
        

        # Transformer processing
        output = self.transformer(bk_feat, bk_feat)
        output = self.dropout(self.norm(output))
        output = self.activation(output)

        # Reshape back to (N*T, B, C)   ->  (B, N, T, C)
        output = output.permute(1,0,2).view(B, T, N, self.feature_dim)

        return output


# BTCN MLP-Mixer from cotracker/blocks.py
class SideNetwork_U(nn.Module):

    def _ntuple(self, n):
        def parse(x):
            if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
                return tuple(x)
            return tuple(repeat(x, n))
        return parse

    def __init__(self, in_features=64, hidden_features=64, out_features=128, act_layer=nn.GELU, bias=True, drop=0.0, use_conv=False):
        super(SideNetwork_U, self).__init__()
        to_2tuple = self._ntuple(2)
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.linear = nn.Conv1d(in_features, in_features, kernel_size=1, bias=bias[0])  # Use Conv1d for 1D data
        
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = nn.LayerNorm(128, elementwise_affine=True)
        self.linear_up = nn.Conv1d(hidden_features, out_features, kernel_size=1, bias=bias[1])  # Use Conv1d for 1D data
        self.drop2 = nn.Dropout(drop_probs[1])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("SideNetwork_U _reset_parameters True")

    def forward(self, bk_feat):
        original_shape = bk_feat.shape # torch.Size([2, 8, 128, 64]) # B T N C
        B, T, N, C = original_shape
        # Reshape by merging B and T, then permuting C and N
        bk_feat = bk_feat.reshape(-1, N, C).permute(0, 2, 1)  # Now the size will be [B*T, C, N]

        # print("bk_feat.shape", bk_feat.shape)  # 16 64 128

        bk_feat = self.linear(bk_feat)

        # print("bk_feat.shape", bk_feat.shape)  # 16 64 128 B*T C N 

        bk_feat = self.act(bk_feat)
        bk_feat = self.drop1(bk_feat)
        bk_feat = self.norm(bk_feat)  # norm by N is perform better than dim

        bk_feat = self.linear_up(bk_feat)

        # print("bk_feat.shape", bk_feat.shape)  # 16 128 128 B*T C N 

        bk_feat = self.drop2(bk_feat)

        # print("bk_feat.shape", bk_feat.shape)

        # Restoring original dimensions and permute back
                
        # Permute back from [B*T, C, N] to [B*T, N, C]
        permuted_back_feat = bk_feat.permute(0, 2, 1)
        # View (or reshape) back to original shape [B, T, N, C]
        out = permuted_back_feat.view(B, T, N, C*2)

        # print("out.shape", out.shape)

        
            
        return out



# PTrackAttentionModel_PTAHWaSPointEmb
class PTrackAttentionModel(nn.Module):
    def __init__(self, feature_dim=256, track_dim=256, num_heads=4, num_layers=2, height=DinoPatch, width=DinoPatch):

        super(PTrackAttentionModel, self).__init__()
        self.feature_dim = feature_dim
        self.transformer = nn.Transformer(
            d_model=feature_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )

        self.wQpos = True
        # self.wQpos = False
        print("self.wQpos", self.wQpos)

        # Define the positional embeddings for the 2D features
        self.positional_embedding = nn.Parameter(torch.randn(1, feature_dim, height, width))
        self.positional_embedding_track = nn.Parameter(torch.randn(1, feature_dim, height, width))

        # Define the additional positional embeddings for the combined sequence
        # self.combined_positional_embedding = nn.Parameter(torch.randn(1, 1, feature_dim, (height * width)*2*2))

        # Define separate positional embeddings for demonstration and query data
        # self.demo_positional_embedding = nn.Parameter(torch.randn(1, 1, feature_dim, (height * width)*2))
        if self.wQpos:
            self.query_positional_embedding = nn.Parameter(torch.randn(1, 1, feature_dim, (height * width)*2))

        # self.linear1 = nn.Linear(feature_dim, feature_dim)
        # self.linear2 = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

        self.activation = F.relu

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("PTAHWaSPointEmb_PTAHWaSPointEmb _reset_parameters True")

    def forward(self, test_tracks, test_features, TFwPTT):

        # print("test_tracks.shape:", test_tracks.shape)  # torch.Size([1, 3, 256, h, w])
        # print("test_features.shape:", test_features.shape)  # torch.Size([1, 3, 256, h, w])

        # Add the positional embeddings to the features
        test_features_ = test_features + self.positional_embedding

        test_tracks_ = test_tracks + self.positional_embedding_track

        # Combine tracks and features
        test_combined = self.combine_tracks_features(test_tracks_, test_features_)

        # print("test_combined.shape:", test_combined.shape)  # Expected: [1, 3, 256, 648]

        # Apply separate positional embeddings to demonstration and query data
        if self.wQpos:
            test_combined = test_combined + self.query_positional_embedding

        # Squeeze the unnecessary dimension
        squeezed_input = test_combined.squeeze(0) 
        # print("squeezed_input.shape:", squeezed_input.shape)  # 3, 256, 648

        # Permute the dimensions to get the desired shape [648*2, 3, 256]
        combined_input_seq = squeezed_input.permute(2, 0, 1).contiguous()  # Final shape [648, 3, 256] # data bat cha

        # print("combined_input_seq.shape:", combined_input_seq.shape)  # torch.Size([648, 3, 256])

        # Apply the transformer
        output = self.transformer(combined_input_seq, combined_input_seq)

        output = self.dropout(self.norm(output))
        output = self.activation(output)

        # Assuming the transformer output is correctly reshaped
        # print("output.shape:", output.shape)  # Expected: [648, 3, 256]

        # Calculate the number of samples (batch size) and other dimensions
        feature_dim = output.shape[2]
        height = test_features_.shape[-2]
        width = test_features_.shape[-1]
        num_data = test_combined.shape[-1]

        # print("num_samples:", num_samples)  # 3
        # print("feature_dim:", feature_dim)  # 256
        # print("num_data:", num_data)  # 648

        # Reshape the output to the original shape
        output = output.permute(1, 0, 2).contiguous()  # Rearrange dimensions to [3,648, 256]

        if not TFwPTT: # test features
            # test features: Selects the last half of the second dimension
            output = output[:, -output.shape[1] // 2:, :]  # Expected: [3,324, 256]
            # print("output_half.shape:", output_half.shape)  # 
        else: # tracks
            # tracks: Selects the first half of the second dimension
            output = output[:, :num_data//2, :]
            # print("output.shape:", output.shape)  #   # Expected: [3,324, 256]

        output_last = output.permute(0,2,1).view(-1, feature_dim, height, width).unsqueeze(0)
        # print("output_last.shape after reshape:", output_last.shape)  # Expected: [1, 3, 256, h, w]

        

        return output_last

    def combine_tracks_features(self, tracks, features):

        # Reshape features 1 3 256 18 18
        tracks_ = tracks.view(tracks.shape[0], tracks.shape[1], tracks.shape[2], -1)  # Shape: [batch, channels, feature_dim, height * width]
        features_ = features.view(features.shape[0], features.shape[1], features.shape[2], -1)  # Shape: [batch, channels, feature_dim, height * width]
        # print("tracks.shape:", tracks_.shape)  # torch.Size([1, 3, 256, 324])
        # print("features.shape:", features_.shape)  # torch.Size([1, 3, 256, 324])
        
        # Concatenate tracks and features along the last dimension
        combined = torch.cat([tracks_, features_], dim=3)  # Shape: [channels, batch, feature_dim, height * width + seq_length]
        # print("combined.shape:", combined.shape)  # Expected shape: [1, 3, 256, 648]

        return combined


# _gauss_PTA weight1
class PointEmbeddingNetwork(nn.Module):
    def __init__(self, DinoPatch=DinoPatch):
        super(PointEmbeddingNetwork, self).__init__()
        self.conv1x1 = nn.Conv2d(128, 256, kernel_size=1)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("PointEmbeddingNetwork _reset_parameters done")

    def forward(self, x):
        # 1x1 Convolution
        x = self.conv1x1(x).unsqueeze(0)  # [1, B, 256, DinoPatch, DinoPatch]
        # print("PointEmbeddingNetwork.shape:", x.shape)  # torch.Size([1,3,256,h,w])

        return x



# TFEcatmlp_v1
class TFEcatmlp_(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(256*2, 256, 1, bias=False),

            nn.BatchNorm2d(256),

            nn.GELU(),
        )

        self.gelu = nn.GELU()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("TFEcatmlp_v1 _reset_parameters done")
        
    def forward(self, bk_feat):

        # print("bk_feat.shape", bk_feat.shape)
        out = self.mlp(bk_feat)
        # print("out.shape", out.shape)
        

        return out



# TFEcatmlpconv3bt
class TFEcatmlp_(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()

        self.mlp = nn.Sequential(
      
            nn.Conv2d(256*2, 256, kernel_size=3, padding=1),

            # nn.Conv2d(256*2, 256, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(256),

            nn.GELU(),
        )

        self.gelu = nn.GELU()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("TFEcatmlpconv3 _reset_parameters done")
        
    def forward(self, bk_feat):

        # print("bk_feat.shape", bk_feat.shape)
        out = self.mlp(bk_feat)
        # print("out.shape", out.shape)
        

        return out




# TFEcatmlpconv3btAttn
# TFEcatmlpconv1btAttn
class TFEcatmlp(nn.Module):
    def __init__(self, feature_dim=256, track_dim=256, num_heads=4, num_layers=2, height=DinoPatch, width=DinoPatch):

        super(TFEcatmlp, self).__init__()
        self.feature_dim = feature_dim
        self.transformer = nn.Transformer(
            d_model=feature_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )

        # Define the positional embeddings for the 2D features
        self.positional_embedding = nn.Parameter(torch.randn(1, feature_dim, height, width))
        self.positional_embedding_track = nn.Parameter(torch.randn(1, feature_dim, height, width))

        # Define the additional positional embeddings for the combined sequence
        # self.combined_positional_embedding = nn.Parameter(torch.randn(1, 1, feature_dim, (height * width)*2*2))

        # Define separate positional embeddings for demonstration and query data
        # self.demo_positional_embedding = nn.Parameter(torch.randn(1, 1, feature_dim, (height * width)*2))

        self.query_positional_embedding = nn.Parameter(torch.randn(1, 1, feature_dim, (height * width)*2))

        # self.linear1 = nn.Linear(feature_dim, feature_dim)
        # self.linear2 = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

        self.activation = F.relu

        self.mlp = nn.Sequential(
      
            nn.Conv2d(256*2, 256, kernel_size=3, padding=1),
            #nn.Conv2d(256*2, 256, 1),
            # nn.Conv2d(256*2, 256, 1, bias=False),

            nn.BatchNorm2d(256),

            nn.GELU(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("TFEcatmlpconvbtAttn _reset_parameters True")

    def forward(self, test_tracks, test_features):


        # print("test_tracks.shape:", test_tracks.shape)  # torch.Size([1, B, 256, h, w])
        # print("test_features.shape:", test_features.shape)  # torch.Size([1, B, 256, h, w])

        # Add the positional embeddings to the features
        test_features_ = test_features + self.positional_embedding

        test_tracks_ = test_tracks + self.positional_embedding_track

        # Combine tracks and features
        test_combined = self.combine_tracks_features(test_tracks_, test_features_)

        # print("test_combined.shape:", test_combined.shape)  # torch.Size([1, B, 256, 1458])

        # Apply separate positional embeddings to demonstration and query data

        test_combined = test_combined + self.query_positional_embedding

        # Squeeze the unnecessary dimension
        squeezed_input = test_combined.squeeze(0) 
        # print("squeezed_input.shape:", squeezed_input.shape)  # torch.Size([B, 256, 1458])

        # Permute the dimensions to get the desired shape [648*2, 3, 256]
        combined_input_seq = squeezed_input.permute(2, 0, 1).contiguous()  # Final shape [648, 3, 256] # data bat cha

        # print("combined_input_seq.shape:", combined_input_seq.shape)  # torch.Size([1458, B, 256])

        # Apply the transformer
        output = self.transformer(combined_input_seq, combined_input_seq)

        output = self.dropout(self.norm(output))
        output = self.activation(output)

        # Assuming the transformer output is correctly reshaped
        # print("output.shape:", output.shape)  # Expected: [1458, 3, 256]

        # Calculate the number of samples (batch size) and other dimensions
        feature_dim = output.shape[2]
        height = test_features_.shape[-2]
        width = test_features_.shape[-1]
        num_data = test_combined.shape[-1]
        bat = output.shape[1]

        # print("num_samples:", num_samples)  # 3
        # print("feature_dim:", feature_dim)  # 256
        # print("num_data:", num_data)  # 648

        # Reshape the output to the original shape
        output = output.permute(1, 0, 2).contiguous()  # torch.Size([B, 1458, 256])
        # print("output.shape:", output.shape)  # torch.Size([B, 1458, 256])

        # test features: Extract the last half along the second dimension
        attn_test_feat = output[:, -output.shape[1] // 2:, :]
        # print("attn_test_feat.shape:", attn_test_feat.shape)  # torch.Size([B, 729, 256])

        # tracks: Extract the first 'num_data' elements along the second dimension
        attn_tracks_feat = output[:, :num_data//2, :]
        # print("attn_tracks_feat.shape:", attn_tracks_feat.shape)  # torch.Size([B, 729, 256])

        attn_test_feat_last = attn_test_feat.permute(0,2,1).view(-1, bat, feature_dim, height, width).squeeze(0)
        # print("attn_test_feat_last.shape:", attn_test_feat_last.shape)  # torch.Size([B, 256, h, w])

        attn_tracks_feat_last = attn_tracks_feat.permute(0,2,1).view(-1, bat, feature_dim, height, width).squeeze(0)
        # print("attn_tracks_feat_last.shape:", attn_tracks_feat_last.shape)  # torch.Size([B, 256, h, w])

        attn_feat_cat = torch.cat([attn_test_feat_last, attn_tracks_feat_last], dim=1)
        # print("attn_feat_cat.shape:", attn_feat_cat.shape)  # torch.Size([B, 512, h, w])

        output_last = self.mlp(attn_feat_cat)
        # print("output_last.shape:", output_last.shape)  # torch.Size([B, 256, h, w])

        

        return output_last

    def combine_tracks_features(self, tracks, features):

        # Reshape features 1 3 256 18 18
        tracks_ = tracks.view(tracks.shape[0], tracks.shape[1], tracks.shape[2], -1)  # Shape: [batch, channels, feature_dim, height * width]
        features_ = features.view(features.shape[0], features.shape[1], features.shape[2], -1)  # Shape: [batch, channels, feature_dim, height * width]
        # print("tracks.shape:", tracks_.shape)  # torch.Size([1, 3, 256, 324])
        # print("features.shape:", features_.shape)  # torch.Size([1, 3, 256, 324])
        
        # Concatenate tracks and features along the last dimension
        combined = torch.cat([tracks_, features_], dim=3)  # Shape: [channels, batch, feature_dim, height * width + seq_length]
        # print("combined.shape:", combined.shape)  # Expected shape: [1, 3, 256, 648]

        return combined

# _PTlb1
class PTlabelEmbeddingNetwork(nn.Module):
    def __init__(self):
        super(PTlabelEmbeddingNetwork, self).__init__()
        # Convolutional layer to increase channels from 1 to 128
        self.conv = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("PTlabelEmbeddingNetwork1 _reset_parameters done")

    def forward(self, x):

        # print("x.shape", x.shape)

        # out = self.conv(x.permute(1,0,2,3)).contiguous()

        # Interpolate to expand spatial dimensions
        out = F.interpolate(x, size=(96, 128), mode='bilinear', align_corners=False)
        # out = F.interpolate(out, size=(96, 128), mode='bilinear', align_corners=False)
        # print("x.shape", x.shape) # B, 4, 96, 128

        # Apply convolution
        out = self.conv(out.permute(1,0,2,3)).contiguous()
        # print("x.shape", x.shape) # B, 128, 96, 128

        return out



# PTltrb2
class ltrbtargetEmbeddingNetwork(nn.Module):
    def __init__(self):
        super(ltrbtargetEmbeddingNetwork, self).__init__()
        # Convolutional layer to increase channels from 4 to 128
        self.conv = nn.Conv2d(4, 128, kernel_size=3, padding=1)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("ltrbtargetEmbeddingNetwork2 _reset_parameters done")

    def forward(self, x):
        # x shape: torch.Size([1, 3, 4, h, w])
        # print("x.shape", x.shape)

        # Apply convolution
        # out = self.conv(x.squeeze(0))
        # print("x.shape", x.shape) # B, 128, h, w

        # Interpolate to expand spatial dimensions
        # out = F.interpolate(out, size=(96, 128), mode='bilinear', align_corners=False)
        out = F.interpolate(x.squeeze(0), size=(96, 128), mode='bilinear', align_corners=False)
        # print("x.shape", x.shape) # torch.Size([3, 4, 96, 128])

        # Apply convolution
        out = self.conv(out)
        # print("x.shape", x.shape) # B, 128, 96, 128

        return out






Pbias = [False, True]
print("Pbias", Pbias)

# JEPA_predictor_cls_resdv2
class JEPA_predictor_cls(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(JEPA_predictor_cls.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=Pbias[0]),
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return out

    def __init__(self):
        super(JEPA_predictor_cls, self).__init__()

        self.residual_cnn = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=Pbias[1]), 
            self.ResidualBlock(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("JEPA_predictor_cls_resdv2 _reset_parameters done")


    def forward(self, bk_feat):
        # Processing through residual CNN layers
        out = self.residual_cnn(bk_feat)
        return out



# JEPA_predictor_breg_resdv2
class JEPA_predictor_breg(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(JEPA_predictor_breg.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=Pbias[0]),
                # nn.BatchNorm2d(256)
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return out

    def __init__(self):
        super(JEPA_predictor_breg, self).__init__()

        self.residual_cnn = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=Pbias[1]), 
            self.ResidualBlock(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("JEPA_predictor_breg_resdv2 _reset_parameters done")


    def forward(self, bk_feat):
        # Processing through residual CNN layers
        out = self.residual_cnn(bk_feat)
        return out




class WvggtLinearCls(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(WvggtLinearCls.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=Pbias[0]),
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return out

    def __init__(self):
        super(WvggtLinearCls, self).__init__()

        self.residual_cnn = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=Pbias[1]), 
            self.ResidualBlock(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("WvggtLinearCls _reset_parameters done")


    def forward(self, bk_feat):
        # Processing through residual CNN layers
        out = self.residual_cnn(bk_feat)
        return out


class WvggtLinearReg(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self):
            super(WvggtLinearReg.ResidualBlock, self).__init__()

            self.conv_block = nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=Pbias[0]),
            )
            self.skip_connection = nn.Identity()

        def forward(self, x):
            identity = self.skip_connection(x)
            out = self.conv_block(x)
            out += identity
            return out

    def __init__(self):
        super(WvggtLinearReg, self).__init__()

        self.residual_cnn = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=Pbias[1]), 
            self.ResidualBlock(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("WvggtLinearReg _reset_parameters done")


    def forward(self, bk_feat):
        # Processing through residual CNN layers
        out = self.residual_cnn(bk_feat)
        return out


class JEPA_VICReg_Exp_c(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()

        # cha = 256
        # cha = 512
        # cha = 1024
        # cha = 2048
        bis = False
        # bis = True

        # method = "256"
        # method = "256512"

        method = "25610241024"
        # method = "25651210242048"

        # print("JEPA_VICReg_Exp_c cha", cha)
        print("JEPA_VICReg_Exp_c bis", bis)
        print("method", method)

        if method == "256":
            self.mlp = nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=bis),
                # nn.Conv2d(256, cha, 1, bias=bis),
                # nn.BatchNorm2d(cha),
                # nn.GELU(),
            )

        elif method == "256512":
            self.mlp = nn.Sequential(
                nn.Conv2d(256, 512, 1, bias=bis),
            )

        elif method == "25610241024":
            self.mlp = nn.Sequential(
                nn.Conv2d(256, 1024, 1, bias=bis),
                nn.Conv2d(1024, 1024, 1, bias=bis),
            )

        elif method == "25651210242048":
            self.mlp = nn.Sequential(
                nn.Conv2d(256, 512, 1, bias=bis),
                nn.Conv2d(512, 1024, 1, bias=bis),
                nn.Conv2d(1024, 2048, 1, bias=bis),
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("JEPA_VICReg_Exp_c _reset_parameters done")
        
    def forward(self, bk_feat):

        # print("bk_feat.shape", bk_feat.shape)
        out = self.mlp(bk_feat)
        # print("out.shape", out.shape)
        

        return out

class JEPA_VICReg_Exp_r(nn.Module):
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()

        # cha = 256
        # cha = 512
        # cha = 1024
        # cha = 2048
        bis = False
        # bis = True

        # method = "256"
        # method = "256512"

        method = "25610241024"
        # method = "25651210242048"

        # print("JEPA_VICReg_Exp_r cha", cha)
        print("JEPA_VICReg_Exp_r bis", bis)
        print("method", method)

        if method == "256":
            self.mlp = nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=bis),
                # nn.Conv2d(256, cha, 1, bias=bis),
                # nn.BatchNorm2d(cha),
                # nn.GELU(),
            )

        elif method == "256512":
            self.mlp = nn.Sequential(
                nn.Conv2d(256, 512, 1, bias=bis),
            )

        elif method == "25610241024":
            self.mlp = nn.Sequential(
                nn.Conv2d(256, 1024, 1, bias=bis),
                nn.Conv2d(1024, 1024, 1, bias=bis),
            )

        elif method == "25651210242048":
            self.mlp = nn.Sequential(
                nn.Conv2d(256, 512, 1, bias=bis),
                nn.Conv2d(512, 1024, 1, bias=bis),
                nn.Conv2d(1024, 2048, 1, bias=bis),
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("JEPA_VICReg_Exp_r _reset_parameters done")
        
    def forward(self, bk_feat):

        # print("bk_feat.shape", bk_feat.shape)
        out = self.mlp(bk_feat)
        # print("out.shape", out.shape)
        

        return out



class LinearFilterClassifierJC(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, feat, filter):
        # see filter.py
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter
        return filter_layer.apply_filter(feat, filter_proj)


############################################ ToMP_JEPA_PT

# print("start debugging")
# while True:
#     try:
#         print(eval(input()))
#     except:
#         print("error")
