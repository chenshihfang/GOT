# This is GOT-JEPA code for the model pretraining stage.
# We will use the trained model for tracking adaptation in tompnet_JEPAp.py.

# Related files for GOT-JEPA:
# filter_predictor_JEPA_context.py
# transformer_JEPA_context.py
# There is also some related code in heads.py and transformer.py (predictor of the ToMP model).

import math
import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.features as clf_features
import ltr.models.backbone as backbones
from ltr import model_constructor

import ltr.models.transformer.transformer as trans
import ltr.models.transformer.filter_predictor as fp

import ltr.models.transformer.transformer_JEPA_context as transJC
import ltr.models.transformer.filter_predictor_JEPA_context as fpJC

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

import ltr.data.processing_utils as prutils


# DinoPatch = 16
# DinoPatch = 18
DinoPatch = 27
print("DinoPatch tompnet_JEPA", DinoPatch)

# backbone = "VIT-B"
backbone = "VIT-L"
# backbone = "VIT-g"
print("backbone tompnet_JEPA", backbone)


# 1. Import depth
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

    def __init__(self, feature_extractor, 
        head_layer,
        bkMlp,
        Head_JEPA_context,
        Head_JEPA_target,
        JEPA_predictor_cls,
        JEPA_predictor_breg,
        JEPA_VICReg_Exp_c,
        JEPA_VICReg_Exp_r
        ):
        super().__init__()

        print('ToMPnet_JEPA')

        self.JEPA_VICReg_Exp_c = JEPA_VICReg_Exp_c
        self.JEPA_VICReg_Exp_r = JEPA_VICReg_Exp_r

        self.Head_JEPA_context = Head_JEPA_context
        self.Head_JEPA_target = Head_JEPA_target

        self.bkMlp = bkMlp

        self.JEPA_predictor_cls = JEPA_predictor_cls
        self.JEPA_predictor_breg = JEPA_predictor_breg

        # self.use_dino_interfeat = 0
        self.use_dino_interfeat = 1
        print('self.use_dino_interfeat', self.use_dino_interfeat)

        self.feature_extractor = feature_extractor
        # self.head = head

        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

        self.show = 0


    def extract_dino_features_spatial_intermediate_layers(self, images, ran_idx = None):

        with torch.no_grad():

            if backbone == "VIT-L":
                total_features_inter4= self.feature_extractor.get_intermediate_layers(images, [4,11,17,23], reshape=True, return_class_token=False, norm=True)
            elif  backbone == "VIT-g":
                total_features_inter4= self.feature_extractor.get_intermediate_layers(images, [9,19,29,39], reshape=True, return_class_token=False, norm=True)   
            elif  backbone == "VIT-B":
                total_features_inter4= self.feature_extractor.get_intermediate_layers(images, [2, 5, 8, 11], reshape=True, return_class_token=False, norm=True)   

            total_features = total_features_inter4

        total_features = torch.mean(torch.stack(total_features), dim=0)

        total_features = self.bkMlp(total_features)
        
        total_features = F.adaptive_avg_pool2d(total_features, DinoPatch)  

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

    def apply_random_relocation_mask(self, input_tensor, max_mask_percentage=0.3):
        """
        Applies a uniform random relocation mask to each batch in the input tensor on CUDA.
        All batches relocate the same randomly determined percentage of elements,
        between 0% and the specified maximum, in each 18x18 grid of the tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape [B, 1024, 18, 18] on CUDA.
            max_mask_percentage (float): The maximum fraction of elements to relocate, defaults to 0.3.

        Returns:
            torch.Tensor: The tensor with applied relocation, where masked elements are moved to other locations.
        """
        B, C, H, W = input_tensor.shape  # Extract dimensions
        num_elements = H * W  # Total elements in the grid

        # Ensure the tensor is on a CUDA device
        device = input_tensor.device

        # Generate a single random relocation percentage for all batches
        current_mask_percentage = torch.rand(1, device=device) * max_mask_percentage
        num_to_relocate = int(num_elements * current_mask_percentage.item())  # Compute number of elements to relocate

        # Relocation operation
        relocated_tensor = input_tensor.clone()  # Clone to avoid modifying the original tensor
        for i in range(B):
            # Randomly choose indices to mask and to which to relocate
            idx_to_relocate = torch.randperm(num_elements, device=device)[:num_to_relocate]
            idx_target = torch.randperm(num_elements, device=device)[:num_to_relocate]

            # Create a flat view of the tensor for easy indexing
            flat_tensor = relocated_tensor[i].view(C, -1)

            # Swap the elements
            temp = flat_tensor[:, idx_target].clone()
            flat_tensor[:, idx_target] = flat_tensor[:, idx_to_relocate]
            flat_tensor[:, idx_to_relocate] = temp

        return relocated_tensor


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

        # Extract backbone features
        train_ims = train_imgs.reshape(-1, *train_imgs.shape[-3:])
        test_ims = test_imgs.reshape(-1, *test_imgs.shape[-3:])

        train_ims_reshape = self.resize_5d_tensor(train_ims, size = DinoPatch*14)
        test_ims_reshape = self.resize_5d_tensor(test_ims, size = DinoPatch*14)
        
        if not self.use_dino_interfeat:
            train_feat_head = self.extract_dino_features_spatial(train_ims_reshape)
            test_feat_head = self.extract_dino_features_spatial(test_ims_reshape)
        else:
            with torch.no_grad():
                train_feat_head = self.extract_dino_features_spatial_intermediate_layers(train_ims_reshape, None)
                test_feat_head = self.extract_dino_features_spatial_intermediate_layers(test_ims_reshape, None)


        # JEPA
        masked_test_feat_head = self.apply_random_relocation_mask(test_feat_head)
        # JEPA

        # Run head module
        with torch.no_grad():
            cls_filter_target, breg_filter_target = self.Head_JEPA_target(train_feat_head, test_feat_head, train_bb, *args, **kwargs)

        # Run head module
        cls_filter_context, breg_filter_context = self.Head_JEPA_context(train_feat_head, masked_test_feat_head, train_bb, *args, **kwargs)

        cls_filter_context_p = self.JEPA_predictor_cls(cls_filter_context)
        breg_filter_context_p = self.JEPA_predictor_breg(breg_filter_context)

        ### VICReg_Exp
        cls_filter_context_Exp = self.JEPA_VICReg_Exp_c(cls_filter_context)
        breg_filter_context_Exp = self.JEPA_VICReg_Exp_r(breg_filter_context)
        ### VICReg_Exp
        
        return cls_filter_context, breg_filter_context, cls_filter_context_Exp, breg_filter_context_Exp, cls_filter_context_p, breg_filter_context_p, \
                cls_filter_target, breg_filter_target


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

    JEPA_VICReg_Exp_c = heads.JEPA_VICReg_Exp_c()
    JEPA_VICReg_Exp_r = heads.JEPA_VICReg_Exp_r()

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

    ### JEPA
    transformerJC = transJC.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)


    filter_predictorJC = fpJC.FilterPredictor(transformerJC, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)        

    classifierJC = heads.LinearFilterClassifierJC(num_channels=out_feature_dim)

    Head_JEPA_context = heads.Head_JEPA_context(filter_predictor=filter_predictorJC, feature_extractor=head_feature_extractor,
                      classifier=classifierJC, bb_regressor=bb_regressor)


    Head_JEPA_target = heads.Head_JEPA_target(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)
    ### JEPA

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, 
    # head=head, 
    head_layer=head_layer,
    bkMlp = bkMlp,
    Head_JEPA_context = Head_JEPA_context,
    Head_JEPA_target = Head_JEPA_target,
    JEPA_predictor_cls = JEPA_predictor_cls,
    JEPA_predictor_breg = JEPA_predictor_breg,
    JEPA_VICReg_Exp_c = JEPA_VICReg_Exp_c,
    JEPA_VICReg_Exp_r = JEPA_VICReg_Exp_r
    )
    return net