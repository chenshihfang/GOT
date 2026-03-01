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

# DinoPatch = 16
# DinoPatch = 18
DinoPatch = 27

# backbone = "VIT-B"
backbone = "VIT-L"
# backbone = "VIT-g"
print("backbone tompnet_JEPAp", backbone)


import math
import matplotlib

cpu_lim = 0

if cpu_lim:
    import os
    cpu_num = 8  # Num of CPUs you want to use
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
        JEPA_predictor_cls,
        JEPA_predictor_breg,
        ):
        super().__init__()

        print('ToMPnet_JEPAp')

        print("DinoPatch tompnet_JEPAp", DinoPatch)

        self.JEPA_predictor_cls = JEPA_predictor_cls
        self.JEPA_predictor_breg = JEPA_predictor_breg

        self.bkMlp = bkMlp

        self.use_ltrblabels = 0
        print('self.use_ltrblabels', self.use_ltrblabels)

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

        self.show = 0

        # self.auto_cast = True
        self.auto_cast = False
        print("self.auto_cast", self.auto_cast)  #

        # self.auto_cast_full = True
        self.auto_cast_full = False
        print("self.auto_cast_full", self.auto_cast_full)  #

        # self.auto_cast_full_all = True
        self.auto_cast_full_all = False
        print("self.auto_cast_full_all", self.auto_cast_full_all)  #

        # self.infer_bf16 = True # 
        self.infer_bf16 = False # 
        print("self.infer_bf16", self.infer_bf16)  #

        # self.bf16_dinoonly = True #  bf16iDinoO
        self.bf16_dinoonly = False # 
        print("self.bf16_dinoonly", self.bf16_dinoonly)  #

        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32
        print("self.dtype", self.dtype)  #


    def extract_dino_features_spatial_intermediate_layers(self, images, ran_idx = None):

        # print(f"images.shape: {images.shape}") # torch.Size([B*S, 3, 378, 378])

        with torch.no_grad():
            if (self.auto_cast and self.training) or self.infer_bf16  or self.bf16_dinoonly:
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

        # print(f"total_features[0].shape: {total_features[0].shape}") # torch.Size([B*S, 1024, 27, 27])

        total_features = torch.mean(torch.stack(total_features), dim=0)

        # print(f"total_features.shape: {total_features.shape}") # torch.Size([B*S, 1024, 27, 27])

        # print("total_features.dtype", total_features.dtype)  #

        if self.bf16_dinoonly:
            with torch.cuda.amp.autocast(enabled=False):    
                total_features = self.bkMlp(total_features)
        else:
            total_features = self.bkMlp(total_features)
        
        total_features = F.adaptive_avg_pool2d(total_features, DinoPatch)  

        # print("total_features bkMlp.dtype", total_features.dtype)  #


        return total_features

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

        train_ims = train_imgs.reshape(-1, *train_imgs.shape[-3:])
        test_ims = test_imgs.reshape(-1, *test_imgs.shape[-3:])

        train_ims_reshape = self.resize_5d_tensor(train_ims, size = DinoPatch*14)
        test_ims_reshape = self.resize_5d_tensor(test_ims, size = DinoPatch*14)
        
        train_feat_head = self.extract_dino_features_spatial_intermediate_layers(train_ims_reshape, None)
        test_feat_head = self.extract_dino_features_spatial_intermediate_layers(test_ims_reshape, None)

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb, \
        self.JEPA_predictor_cls, self.JEPA_predictor_breg, \
        self.auto_cast_full_all, self.auto_cast_full, self.dtype, self.infer_bf16 \
        *args, **kwargs)

        # print("test_scores.dtype", test_scores.dtype)  #

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
              num_decoder_layers=6, dim_feedforward=2048, 
              feature_sz=18, 
              use_test_frame_encoding=True):

    # Backbone
    if backbone == "VIT-L":
        dinov2_name = "dinov2_vitl14"
    elif  backbone == "VIT-g":
        dinov2_name = "dinov2_vitg14_reg"   
    elif  backbone == "VIT-B":
        dinov2_name = "dinov2_vitb14"

    print("dinov2_name", dinov2_name)

    backbone_net = backbones.dinov2(dinov2_name)

    bkMlp = heads.bkMlp()

    JEPA_predictor_cls = heads.JEPA_predictor_cls()
    JEPA_predictor_breg = heads.JEPA_predictor_breg()

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

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)


    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer,
    bkMlp = bkMlp,
    JEPA_predictor_cls = JEPA_predictor_cls,
    JEPA_predictor_breg = JEPA_predictor_breg,
    )
    return net














