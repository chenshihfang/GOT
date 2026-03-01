
# Guidance

# This file contains the main training call for GOT-Edit training. 
# The default path uses Depth Anything 3 as geometry. To switch to VGGT or StreamVGGT:
# 1. Enable all comments related to `vggtDPTfeatMlp` and `vggtDPTfeatMlp_head`, then comment out `DA3_bkMlp`.
# 2. Select `self.use_geo_type` and `backbone_net_VGGT` for the intended type.

# Resolution
    # The default resolution uses 378 × 378 (patch 27).
    # To change to 252 × 252 (patch 18), modify `DinoPatch` to 18 in `heads.py` and in the evaluation code `tomp.py`.
    # Also change `bkMlp` to `bkMlpMOEv2`.

import math
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.features as clf_features
import ltr.models.backbone as backbones
from ltr import model_constructor

import ltr.models.transformer.transformer as trans
import ltr.models.transformer.filter_predictor as fp
import ltr.models.transformer.heads as heads
import random
from torch.cuda.amp import autocast
import torch
import torchvision.transforms.functional as fn
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import sys
import os.path
from os import path
import glob
import numpy as np
from PIL import Image
from pytracking.utils.plotting import draw_figure, overlay_mask

from heatmap import heat_show

import time

import ltr.data.processing_utils as prutils

from typing import Optional


try:
    import deepspeed
except Exception:
    deepspeed = None

# DinoPatch = 16
# DinoPatch = 18
DinoPatch = 27

# backbone = "VIT-B"
backbone = "VIT-L"
# backbone = "VIT-g"
print("backbone tompnet_JEPAp_vggt", backbone)

# 1. Import depth
import math
import matplotlib


cpu_lim = 0

if cpu_lim:
    import os
    cpu_num = 8  # Num of CPUs  want to use
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


device = "cuda" if torch.cuda.is_available() else "cpu"


class ToMPnet(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer,
        bkMlp,
        # bkMlpMOEv2,
        JEPA_predictor_cls,
        JEPA_predictor_breg,
        WvggtLinearCls, 
        feature_extractor_VGGT,

        # vggtDPTfeatMlp, # enable when VGGT/StreamVGGT
        # vggtDPTfeatMlp_head, # enable when VGGT/StreamVGGT

        DA3_bkMlp, # enable when DA3

        DiNO_VGGT_Gate,
        ):
        super().__init__()

        print('ToMPnet_JEPAp')

        print("DinoPatch tompnet_JEPAp vggt", DinoPatch)

        # self.use_geo_type = "VGGT"
        # self.use_geo_type = "StreamVGGT"
        self.use_geo_type = "DA3" 

        self.DiNO_VGGT_Gate = DiNO_VGGT_Gate

        self.JEPA_predictor_cls = JEPA_predictor_cls
        self.JEPA_predictor_breg = JEPA_predictor_breg

        self.WvggtLinearCls = WvggtLinearCls

        self.bkMlp = bkMlp
        # self.bkMlpMOEv2 = bkMlpMOEv2

        self.feature_extractor_VGGT = feature_extractor_VGGT

        # self.vggtDPTfeatMlp = vggtDPTfeatMlp  # enable when VGGT/StreamVGGT
        # self.vggtDPTfeatMlp_head = vggtDPTfeatMlp_head  # enable when VGGT/StreamVGGT

        self.DA3_bkMlp = DA3_bkMlp  # enable when DA3

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

        self.show = 0
  
        # self.auto_cast = True
        self.auto_cast = False
        print("self.auto_cast vggt", self.auto_cast)  #

        # self.auto_cast_full = True
        self.auto_cast_full = False
        print("self.auto_cast_full vggt", self.auto_cast_full)  #

        self.bf16_vggtonly = True #
        # self.bf16_vggtonly = False # 
        print("self.bf16_vggtonly", self.bf16_vggtonly)  #

        # self.bf16_dinoonly = True #
        self.bf16_dinoonly = False # 
        print("self.bf16_dinoonly", self.bf16_dinoonly)  #

        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32
        print("self.dtype", self.dtype)  #

        self.vggt_tokens_per_frame = None             # infer once from ps_idx[-1]

    # extract_dino_features_spatial_intermediate_layers_bkMlp
    def extract_dino_features_spatial_intermediate_layers(self, images, ran_idx = None):

        # print(f"images.shape: {images.shape}") # torch.Size([B*S, 3, 378, 378])

        if self.training and self.use_geo_type == "DA3":
            self.bf16_vggtonly = False # 

        with torch.no_grad():
            if (self.auto_cast and self.training):
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    if backbone == "VIT-L":
                        total_features_inter4= self.feature_extractor.get_intermediate_layers(images, [4,11,17,23], reshape=True, return_class_token=False, norm=True)
                    elif  backbone == "VIT-g":
                        total_features_inter4= self.feature_extractor.get_intermediate_layers(images, [9,19,29,39], reshape=True, return_class_token=False, norm=True)   
                    elif  backbone == "VIT-B":
                        total_features_inter4= self.feature_extractor.get_intermediate_layers(images, [2, 5, 8, 11], reshape=True, return_class_token=False, norm=True)   
            else:
                if backbone == "VIT-L":
                    total_features_inter4= self.feature_extractor.get_intermediate_layers(images, [4,11,17,23], reshape=True, return_class_token=False, norm=True)
                elif  backbone == "VIT-g":
                    total_features_inter4= self.feature_extractor.get_intermediate_layers(images, [9,19,29,39], reshape=True, return_class_token=False, norm=True)   
                elif  backbone == "VIT-B":
                    total_features_inter4= self.feature_extractor.get_intermediate_layers(images, [2, 5, 8, 11], reshape=True, return_class_token=False, norm=True)   
            total_features = total_features_inter4

        # print(f"total_features[0].shape: {total_features[0].shape}") # torch.Size([B*S, 1024, h, w])

        total_features = torch.mean(torch.stack(total_features), dim=0)

        # print(f"total_features.shape: {total_features.shape}") # torch.Size([B*S, 1024, h, w])

        # print("total_features.dtype", total_features.dtype)  #

        with torch.cuda.amp.autocast(enabled=False):    
            total_features = self.bkMlp(total_features)

        total_features = F.adaptive_avg_pool2d(total_features, DinoPatch)  

        # print("total_features bkMlp.dtype", total_features.dtype)  #

        return total_features


    # extract_dino_features_spatial_intermediate_layers_bkMlpMOEv2
    def extract_dino_features_spatial_intermediate_layers_bkMlpMOEv2(self, images, ran_idx=None):
        """
        Fuse intermediate DINO features with bkMlpMOEv2.
        Input:  list of K tensors [B, C, H, W] from inter layers
        Output: fused tensor [B, C, H, W] (H,W from the first map), then pooled to DinoPatch
        """
        if self.training and self.use_geo_type == "DA3":
            self.bf16_vggtonly = False # 
            
        with torch.no_grad():
            if (self.auto_cast and self.training):
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    if backbone == "VIT-L":
                        inter_feats = self.feature_extractor.get_intermediate_layers(
                            images, [4, 11, 17, 23], reshape=True, return_class_token=False, norm=True
                        )
                    elif backbone == "VIT-g":
                        inter_feats = self.feature_extractor.get_intermediate_layers(
                            images, [9, 19, 29, 39], reshape=True, return_class_token=False, norm=True
                        )
                    elif backbone == "VIT-B":
                        inter_feats = self.feature_extractor.get_intermediate_layers(
                            images, [2, 5, 8, 11], reshape=True, return_class_token=False, norm=True
                        )
                    else:
                        raise ValueError(f"unsupported backbone: {backbone}")
            else:
                if backbone == "VIT-L":
                    inter_feats = self.feature_extractor.get_intermediate_layers(
                            images, [4, 11, 17, 23], reshape=True, return_class_token=False, norm=True
                        )
                elif backbone == "VIT-B":
                    inter_feats = self.feature_extractor.get_intermediate_layers(
                        images, [2, 5, 8, 11], reshape=True, return_class_token=False, norm=True
                    )
                else:
                    raise ValueError(f"unsupported backbone: {backbone}")

        # fusion returns [B, C, H, W] (same H,W as first map)

        with torch.cuda.amp.autocast(enabled=False):
            total_features = self.bkMlpMOEv2(inter_feats)

        # downstream pooling ()
        total_features = F.adaptive_avg_pool2d(total_features, DinoPatch)
        return total_features


    def extract_da3_dpt_features_intermediate_cat(self, images, ran_idx=None):
        # images: [B, S, 3, H, W]
        B, S, C, H, W = images.shape

        with torch.no_grad():
            if (self.auto_cast and self.training) or self.bf16_vggtonly:
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    feat_list = self.feature_extractor_VGGT.forward_dpt_features(images)
            else:
                feat_list = self.feature_extractor_VGGT.forward_dpt_features(images)

        # take last DA3 feature
        feat_last = feat_list[-1]   # (B, S, 128, H_dpt, W_dpt)

        # flatten to (B*S, 128, H_dpt, W_dpt)
        feat_last_flat = feat_last.reshape(B * S, *feat_last.shape[-3:]).to(torch.float32)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            # adapt feature
            feat_256_flat = self.DA3_bkMlp(feat_last_flat)   # (B*S, 256, H_dpt, W_dpt)

        # reshape back to (B, S, 256, H_dpt, W_dpt)
        feat_256 = feat_256_flat.reshape(B, S, *feat_256_flat.shape[1:])

        vggt_dpt_feats_ = feat_256.permute(1,0,2,3,4) # S*B

        # print("vggt_dpt_feats_.shape:", vggt_dpt_feats_.shape)

        return vggt_dpt_feats_

    
    # extract_dino_features_spatial_intermediate_layers_vggt_cat_wocache (streamvggt wocache path for training)
    def extract_dino_features_spatial_intermediate_layers_vggt_cat(self, images, ran_idx=None):

        with torch.no_grad():
            if (self.auto_cast and self.training) or self.bf16_vggtonly:
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    # print("images.shape", images.shape) # torch.Size([B, 3, 3, 252, 252])
                    try:
                        aggregated_tokens_list, ps_idx = self.feature_extractor_VGGT.aggregator(images, res_norm = False)
                    except:    
                        aggregated_tokens_list, ps_idx = self.feature_extractor_VGGT.aggregator(images)
                    # print("aggregated_tokens_list[0].dtype", aggregated_tokens_list[0].dtype)  #
            else:
                    # print("images.shape", images.shape) # torch.Size([B, 3, 3, 252, 252])
                    try:
                        aggregated_tokens_list, ps_idx = self.feature_extractor_VGGT.aggregator(images, res_norm = False)
                    except:    
                        aggregated_tokens_list, ps_idx = self.feature_extractor_VGGT.aggregator(images)


            if self.use_geo_type == "StreamVGGT" or self.use_geo_type == "VGGT": # auto1
                try:
                    f0, fl1 = self.feature_extractor_VGGT.depth_head(aggregated_tokens_list, images, ps_idx, vggt_DPT_feat_ToMP = True)
                except:    
                    f0, fl1 = self.feature_extractor_VGGT.depth_head(aggregated_tokens_list, images, ps_idx)
            else:
                # force fp32 tokens for depth head
                agg_fp32 = [t.to(torch.float32) for t in aggregated_tokens_list]
                with torch.cuda.amp.autocast(enabled=False):
                    try:
                        f0, fl1 = self.feature_extractor_VGGT.depth_head(
                            agg_fp32, images.to(torch.float32), ps_idx,
                            vggt_DPT_feat_ToMP=True
                        )
                    except:   
                        f0, fl1 = self.feature_extractor_VGGT.depth_head(
                            agg_fp32, images.to(torch.float32), ps_idx
                        )
                        
            # print("f0.shape, fl1.shape", f0.shape, fl1.shape)
            # torch.Size([2, 3, 128, 144, 144]) 
            # torch.Size([2, 3, 256, 144, 144]) 
        
            vggt_dpt_feats = [f0, fl1]

        # with torch.cuda.amp.autocast(enabled=False):    

        # Cast each tensor inside the list to fp32 # remove when infer # self.bf16_vggtonly = True


        if torch.is_grad_enabled():  # auto2 (remove these)
            if self.bf16_vggtonly and not self.auto_cast:
                vggt_dpt_feats = [t.to(torch.float32) for t in vggt_dpt_feats]
        else:
            if not self.auto_cast:
                vggt_dpt_feats = [t.to(torch.float32) for t in vggt_dpt_feats] # FP32 All


        f0_per, fl1_per = vggt_dpt_feats
        # print("f0_per.shape", f0_per.shape, fl1_per.dtype) # torch.Size([T, B, 128, 72, 72]) torch.float32
        # print("fl1_per.shape", fl1_per.shape) # torch.Size([T, B, 256, 144, 144])


        with torch.cuda.amp.autocast(enabled=False):    
            vggt_dpt_feats_ = self.vggtDPTfeatMlp(fl1_per, "fl1")

        
        vggt_dpt_feats_ = vggt_dpt_feats_.permute(1,0,2,3,4) # S*B
    
        return vggt_dpt_feats_


    # Streamvggt uses the cache path function
    def _kv_count_frames(self, pkv) -> int:
        """How many frames are represented by KV (given tokens_per_frame)."""
        if pkv is None or self.vggt_tokens_per_frame is None or self.vggt_tokens_per_frame <= 0:
            return 0
        first = pkv[0]
        if first is None or first[0] is None:
            return 0
        total_tokens = int(first[0].shape[2])  # [B, heads, T_tokens, head_dim]
        return total_tokens // int(self.vggt_tokens_per_frame)

    # Streamvggt uses the cache path function
    def _last_val(self, x):
        # list / tuple
        if isinstance(x, (list, tuple)):
            return x[-1]
        # torch tensor (scalar or 1-D)
        if torch.is_tensor(x):
            return x[-1] if x.ndim > 0 else x
        # plain scalar (int/float)
        return x


    # extract_dino_features_spatial_intermediate_layers_vggt_cat_cache (Streamvggt uses the cache path function)
    @torch.no_grad()
    def extract_dino_features_spatial_intermediate_layers_vggt_cat_cache(
        self,
        images: torch.Tensor,                 # [B,S,C,H,W] or [C,H,W]; streaming per frame -> [1,1,C,H,W]
        ran_idx=None,
        *,
        past_key_values=None,                 # None for Case 1/2; list for Case 3
        return_cache = False,           # Case 1: False; Case 2/3: True
        past_frame_idx = None,    # optional override; else inferred from past_key_values
        res_norm = False
    ):
        """
        Cases:
        1) past_key_values is None AND return_cache is False  -> run WITHOUT cache; return (features, None)
        2) past_key_values is None AND return_cache is True   -> start a fresh cache; return (features, new_kv)
        3) past_key_values is not None AND return_cache is True -> reuse cache; return (features, updated_kv)

        Returns:
            vggt_dpt_feats_: torch.Tensor  # [S, B, C', H', W']
            new_kv: list | None            # None for Case 1; list for Case 2/3
        """

        # --- ensure 5D for aggregator: [B,S,C,H,W] ---
        if images.ndim == 4:
            # assume [C,H,W], promote to [1,1,C,H,W]
            images = images.unsqueeze(0).unsqueeze(0)
        elif images.ndim != 5:
            raise ValueError(f"images must be [B,S,C,H,W] or [C,H,W], got {tuple(images.shape)}")

        # --- decide cache mode ---
        use_cache = bool(return_cache)

        # past_frame_idx defaulting
        if past_frame_idx is None:
            # infer from provided cache (safe); 0 if no cache
            past_frame_idx = self._kv_count_frames(past_key_values)

        if use_cache and past_key_values is None:
            # Aggregator updates KV in-place and expects a list sized to its depth
            try:
                with torch.no_grad():
                    if (self.auto_cast and self.training) or self.bf16_vggtonly:
                        with torch.cuda.amp.autocast(dtype=self.dtype):
                            agg_depth = int(self.feature_extractor_VGGT.aggregator.depth)
                    else:
                        agg_depth = int(self.feature_extractor_VGGT.aggregator.depth)
            except Exception:
                agg_depth = 24  # safe default for VGGT-large; adjust if  use another backbone
            past_key_values = [None] * agg_depth
            
        # --- build aggregator kwargs (robust to forks) ---
        agg_kwargs = {}
        if use_cache:
            agg_kwargs.update({
                "past_key_values": past_key_values,   # None (Case 2) or list (Case 3)
                "use_cache": True,
                "past_frame_idx": int(past_frame_idx),
            })
        else:
            agg_kwargs.update({
                "past_key_values": None,              # Case 1
                "use_cache": False,
                "past_frame_idx": 0,
            })
        if res_norm is not False:
            agg_kwargs["res_norm"] = res_norm

        def _call_aggregator():

            with torch.no_grad():
                if (self.auto_cast and self.training) or self.bf16_vggtonly:
                    with torch.cuda.amp.autocast(dtype=self.dtype):
                        return self.feature_extractor_VGGT.aggregator(images, **agg_kwargs)
                else:
                    return self.feature_extractor_VGGT.aggregator(images, **agg_kwargs)

        # --- AMP policy preserved ---
        if (self.auto_cast and self.training) or self.bf16_vggtonly:
            with torch.cuda.amp.autocast(dtype=self.dtype):
                agg_out = _call_aggregator()
        else:
            agg_out = _call_aggregator()

        # --- unpack aggregator outputs ---
        if isinstance(agg_out, (list, tuple)) and len(agg_out) == 3:
            aggregated_tokens_list, ps_idx, new_kv = agg_out
        else:
            aggregated_tokens_list, ps_idx = agg_out
            new_kv = None  # Case 1 or fork keeps KV internally

        # learn tokens_per_frame once
        if getattr(self, "vggt_tokens_per_frame", None) is None:
            last_val = self._last_val(ps_idx)
            self.vggt_tokens_per_frame = int(last_val.item() if torch.is_tensor(last_val) else last_val)

        # --- heads ( original logic) ---
        # autocast
        if self.use_geo_type == "StreamVGGT": 
            try:
                f0, fl1 = self.feature_extractor_VGGT.depth_head(
                    aggregated_tokens_list, images, ps_idx, vggt_DPT_feat_ToMP=True
                )
            except Exception:
                f0, fl1 = self.feature_extractor_VGGT.depth_head(
                    aggregated_tokens_list, images, ps_idx
                )
        else:
            # vggt uses fp32 for training
            agg_fp32 = [t.to(torch.float32) for t in aggregated_tokens_list]
            with torch.cuda.amp.autocast(enabled=False):
                try:
                    f0, fl1 = self.feature_extractor_VGGT.depth_head(
                        agg_fp32, images.to(torch.float32), ps_idx,
                        vggt_DPT_feat_ToMP=True
                    )
                except:   
                    f0, fl1 = self.feature_extractor_VGGT.depth_head(
                        agg_fp32, images.to(torch.float32), ps_idx
                    )

        vggt_dpt_feats = [f0, fl1]

        # dtype casts (preserved)
        if torch.is_grad_enabled():
            if self.bf16_vggtonly and not self.auto_cast:
                vggt_dpt_feats = [t.to(torch.float32) for t in vggt_dpt_feats]
        else:
            if not self.auto_cast:
                vggt_dpt_feats = [t.to(torch.float32) for t in vggt_dpt_feats]

        f0_per, fl1_per = vggt_dpt_feats

        # print("fl1_per.shape", fl1_per.shape)

        with torch.cuda.amp.autocast(enabled=False):
            vggt_dpt_feats_ = self.vggtDPTfeatMlp(fl1_per, "fl1")

        vggt_dpt_feats_ = vggt_dpt_feats_.permute(1, 0, 2, 3, 4)  # [S, B, ...]

        return vggt_dpt_feats_, new_kv

    def resize_5d_tensor(self, dclip_test_imgs, size=512):
        """
        Resizes a 5D tensor (or 4D tensor) using bilinear interpolation and reshapes the output.
        Handles cases where S or B dimensions are 1, and preserves input dimensionality.
        Skips interpolation if input size matches the target size.

        Args:
            dclip_test_imgs (torch.Tensor): The input tensor, which can be either 5D (S, B, C, H, W)
                                             or 4D (B, C, H, W).
            size (int, optional): The target size for the height and width of the images.
                                     Defaults to 512.

        Returns:
            torch.Tensor: The resized and reshaped tensor.
                          Returns as (S, B, C, size, size) for 5D input
                          or (B, C, size, size) for 4D input.
        """
        if dclip_test_imgs.ndim == 5:
            # Input is 5D: (S, B, C, H, W)
            s, b, c, h, w = dclip_test_imgs.shape
            if h == size and w == size:
                return dclip_test_imgs
            # Reshape to (S * B, C, H, W) to apply interpolation across batches and samples
            reshaped_input = dclip_test_imgs.reshape(s * b, c, h, w)
            # Interpolate the last two dimensions (H, W)
            resized_imgs = F.interpolate(reshaped_input, size=(size, size), mode='bilinear', align_corners=False)
            # Reshape back to (S, B, C, size, size)
            reshaped_imgs = resized_imgs.reshape(s, b, c, size, size)
        elif dclip_test_imgs.ndim == 4:
            # Input is 4D: (B, C, H, W)
            b, c, h, w = dclip_test_imgs.shape
            if h == size and w == size:
                return dclip_test_imgs
            # Interpolate the last two dimensions (H, W)
            resized_imgs = F.interpolate(dclip_test_imgs, size=(size, size), mode='bilinear', align_corners=False)
            # Reshape to (B, C, size, size)
            reshaped_imgs = resized_imgs
        else:
            raise ValueError(f"Input tensor must be 4D or 5D, but got {dclip_test_imgs.ndim}D")
        return reshaped_imgs


    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # print("kwargs['train_label'].shape:", kwargs['train_label'].shape) # torch.Size([ref_im_num, B, H, W]) torch.Size([2, B, H, W]) torch.Size([3, 2, H, W])
        # print("kwargs['train_ltrb_target'].shape:", kwargs['train_ltrb_target'].shape) # torch.Size([ref_im_num, B, 4, H, W]) torch.Size([2, B, 4, H, W]) torch.Size([3, 2, 4, H, W])
        # print("train_imgs.shape:", train_imgs.shape) # torch.Size([ref_im_num, B, 3, H, W])
        # print("test_imgs.shape:", test_imgs.shape) # torch.Size([1, B, 3, H, W])

        train_ims_resize = self.resize_5d_tensor(train_imgs, size = DinoPatch*14)
        test_ims_resize = self.resize_5d_tensor(test_imgs, size = DinoPatch*14)

        ### Semantic
        train_ims_reshape = train_ims_resize.reshape(-1, *train_ims_resize.shape[-3:])
        test_ims_reshape = test_ims_resize.reshape(-1, *test_ims_resize.shape[-3:])

        train_feat_head_2d = self.extract_dino_features_spatial_intermediate_layers(train_ims_reshape, None) # S*B
        test_feat_head_2d = self.extract_dino_features_spatial_intermediate_layers(test_ims_reshape, None) # S*B

        # print("train_feat_head_2d.shape:", train_feat_head_2d.shape) # torch.Size([ref_im_num*B, 1024 , h, w])
        # print("test_feat_head_2d.shape:", test_feat_head_2d.shape) # torch.Size([B, 1024, h, w])

        ### Geometry
        test_ims_resize_permute = test_ims_resize.permute(1,0,2,3,4)

        train_ims_resize_permute = train_ims_resize.permute(1,0,2,3,4) # B*S
        # combined_ims_reshape = torch.cat((test_ims_resize_permute, train_ims_resize_permute), dim=1)
        combined_ims_reshape = torch.cat((train_ims_resize_permute, test_ims_resize_permute), dim=1)

        if self.use_geo_type == "VGGT" or self.use_geo_type == "StreamVGGT": 
            vggt_dpt_feats_ = self.extract_dino_features_spatial_intermediate_layers_vggt_cat(combined_ims_reshape, None) # S*B
        else:
            vggt_dpt_feats_ = self.extract_da3_dpt_features_intermediate_cat(combined_ims_reshape, None) # S*B
            
        # print("vggt_dpt_feats_.shape:", vggt_dpt_feats_.shape) # torch.Size([ref_im_num+test_im_num, B, 3, H, W])  torch.Size([ref_im_num, B, 3, H, W]) torch.Size([2, B, 3, H, W]) torch.Size([3, B, 3, H, W])

        # --- Step 2: Split ---
        # Get the first slice along dimension 0 (index 0)
        # Using [0:1] keeps the dimension, resulting in shape [1, B, C, H, W]
        # test_vggt_dpt_feats_head = vggt_dpt_feats_[0:1]

        # Get the remaining slices along dimension 0 (indices 1 and 2)
        # Using [1:] slices from index 1 to the end
        # train_vggt_dpt_feats_head = vggt_dpt_feats_[1:]


        # --- Step 2: Split ---
        train_vggt_dpt_feats_head = vggt_dpt_feats_[:train_imgs.shape[0]]
        test_vggt_dpt_feats_head = vggt_dpt_feats_[-1:]

        # print("train_vggt_dpt_feats_head.shape:", train_vggt_dpt_feats_head.shape) #
        # print("test_vggt_dpt_feats_head.shape:", test_vggt_dpt_feats_head.shape) # 

        # --- Verification ---
        # print(f"\nShape of test_feat_head: {test_feat_head.shape}")  # Expected: [1, B, 1024, h, w]
        # print(f"Shape of train_feat_head: {train_feat_head.shape}") # Expected: [2, B, 1024, h, w]

        test_vggt_dpt_feats_head = test_vggt_dpt_feats_head.reshape(-1, *test_vggt_dpt_feats_head.shape[-3:])
        train_vggt_dpt_feats_head = train_vggt_dpt_feats_head.reshape(-1, *train_vggt_dpt_feats_head.shape[-3:])

        ### Semantic + Geometry

        if self.use_geo_type == "VGGT" or self.use_geo_type == "StreamVGGT":  
            test_vggt_dpt_feats_head_ = self.vggtDPTfeatMlp_head(test_vggt_dpt_feats_head.contiguous())
            train_vggt_dpt_feats_head_ = self.vggtDPTfeatMlp_head(train_vggt_dpt_feats_head.contiguous())
        else: # DA3
            test_vggt_dpt_feats_head_= test_vggt_dpt_feats_head.contiguous()
            train_vggt_dpt_feats_head_ = train_vggt_dpt_feats_head.contiguous()

        # print("test_vggt_dpt_feats_head_.shape", test_vggt_dpt_feats_head_.shape) # torch.Size([B, 256, h, w])
        # print("train_vggt_dpt_feats_head_.shape", train_vggt_dpt_feats_head_.shape) # torch.Size([B*ref_num, 256, h, w])

        test_scores, bbox_preds = self.head(train_feat_head_2d, train_vggt_dpt_feats_head_, \
        test_vggt_dpt_feats_head_, test_feat_head_2d, \
        train_bb, \
        self.JEPA_predictor_cls, self.JEPA_predictor_breg, \
        self.WvggtLinearCls, \
        self.DiNO_VGGT_Gate, \
        self.auto_cast_full, self.dtype, \
        *args, **kwargs)  


        return test_scores, bbox_preds

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]
        return feat

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def tompnet50(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
              final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
              num_decoder_layers=6,
              dim_feedforward=2048, 
              feature_sz=18, # Automatically adjust based on the training configuration
              use_test_frame_encoding=True,
              use_activation_checkpointing=False,
              ckpt_impl="torch"
              ):

    # Backbone
    if backbone == "VIT-L":
        dinov2_name = "dinov2_vitl14"
    elif  backbone == "VIT-g":
        dinov2_name = "dinov2_vitg14_reg"   
    elif  backbone == "VIT-B":
        dinov2_name = "dinov2_vitb14"

    print("dinov2_name", dinov2_name)

    backbone_net = backbones.dinov2(dinov2_name)

    # backbone_net_VGGT = backbones.dinov2_VGGT()
    # backbone_net_VGGT = backbones.dinov2_StreamVGGT()
    backbone_net_VGGT = backbones.dinov2_DA3(checkpoint_path="depth-anything/da3-large")

    bkMlp = heads.bkMlp()
    # bkMlpMOEv2 = heads.bkMlpMOEv2()

    JEPA_predictor_cls = heads.JEPA_predictor_cls()
    JEPA_predictor_breg = heads.JEPA_predictor_breg()

    WvggtLinearCls = heads.WvggtLinearCls()

    # vggtDPTfeatMlp = heads.vggtDPTfeatMlp()  # enable when VGGT/StreamVGGT
    # vggtDPTfeatMlp_head = heads.vggtDPTfeatMlp_head()  # enable when VGGT/StreamVGGT

    DA3_bkMlp = heads.DA3_bkMlp()  # enable when DA3


    DiNO_VGGT_Gate = heads.DiNO_VGGT_Gate()

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))


    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    # when dim = 1024
    head_feature_extractor = clf_features.residual_bottleneck_reshape(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    # when dim = 768
    # head_feature_extractor = clf_features.residual_bottleneck_reshape_768(feature_dim=feature_dim,
    #                                                             num_blocks=head_feat_blocks, l2norm=head_feat_norm,
    #                                                             final_conv=final_conv, norm_scale=norm_scale,
    #                                                             out_dim=out_feature_dim)


    # Build transformer with the toggle threaded in
    transformer = trans.Transformer(
        d_model=out_feature_dim, nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        use_ckpt=use_activation_checkpointing,
        ckpt_impl=ckpt_impl,     # <- propagate choice
    )


    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer,
    # bkMlpMOEv2 = bkMlpMOEv2,
    bkMlp = bkMlp,
    JEPA_predictor_cls = JEPA_predictor_cls,
    JEPA_predictor_breg = JEPA_predictor_breg,
    WvggtLinearCls = WvggtLinearCls,
    feature_extractor_VGGT=backbone_net_VGGT,

    # vggtDPTfeatMlp = vggtDPTfeatMlp,  # enable when VGGT/StreamVGGT
    # vggtDPTfeatMlp_head=vggtDPTfeatMlp_head,  # enable when VGGT/StreamVGGT

    DA3_bkMlp = DA3_bkMlp,  # enable when DA3

    DiNO_VGGT_Gate=DiNO_VGGT_Gate,
    )

    # Configure DS activation checkpointing once if requested
    if use_activation_checkpointing and str(ckpt_impl).lower().strip() == "deepspeed" and deepspeed is not None:
        import inspect
        # Try both module locations
        ds_ckpt = None
        try:
            from deepspeed.runtime.activation_checkpointing import checkpointing as ds_ckpt  # new path
        except Exception:
            ds_ckpt = getattr(deepspeed, "checkpointing", None)  # older path

        # If DeepSpeed’s wrapper exists, try to configure conservatively; else skip
        if ds_ckpt is not None and hasattr(ds_ckpt, "checkpoint"):
            did_configure = False
            if hasattr(ds_ckpt, "configure"):
                try:
                    sig = inspect.signature(ds_ckpt.configure)
                    kwargs = {}
                    # Only pass kwargs that exist in this DS version
                    for k, v in dict(
                        partition_activations=False,
                        contiguous_checkpointing=True,
                        checkpoint_in_cpu=False,
                        # If the installed DS exposes RNG preservation, pass True.
                        preserve_rng_state=True,
                    ).items():
                        if k in sig.parameters:
                            kwargs[k] = v
                    ds_ckpt.configure(**kwargs)
                    did_configure = True
                except Exception:
                    print("[WARN] DeepSpeed checkpointing.configure not supported on this node; continuing without it.")
            print("[INFO] Activation checkpointing ready via DeepSpeed." + (" (configured)" if did_configure else ""))
        else:
            print("[INFO] DeepSpeed checkpoint wrapper not found; will use torch.utils.checkpoint fallback.")
    else:
        # torch path or AC disabled — nothing to configure here
        pass



    return net














