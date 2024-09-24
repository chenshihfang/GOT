import torch
import torch.nn as nn
import ltr.models.layers.filter as filter_layer
import math
from ltr.models.transformer.position_encoding import PositionEmbeddingSine
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn.init as init
from torch import Tensor
from functools import partial
import collections
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import cv2
from timm.models.layers import DropPath

from typing import Tuple, Union

try:
    from timm.models.layers import drop, drop_path, trunc_normal_
except:
    pass

from collections import OrderedDict

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
class Head(nn.Module):
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

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        # Train filter
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)

        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feat_enc, cls_filter)
        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)

        return target_scores, bbox_preds


    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""

        if self.feature_extractor is None:
            # print("self.feature_extractor is None") # no pass
            return feat

        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)

        return output.reshape(-1, num_sequences, *output.shape[-3:])


    def extract_head_feat2(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""

        output = self.feature_extractor(feat)  # Shape: [3*num_sequences*B, C, H, W]
        
        # Calculate N
        N = output.shape[0] // (num_sequences * 3)

        # Reshape to [N, 3*num_sequences, C, H, W]
        final_output = output.view(N, 3 * num_sequences, -1, output.shape[-2], output.shape[-1])

        return final_output  # Shape: [num_sequences, 3*B, C, H, W]

        
    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):
        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights
            
        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
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
class bkMlp(nn.Module):
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

        out = self.mlp(bk_feat)
        return out
        
############################################ PiVOT

