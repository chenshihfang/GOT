# We call CoTracker in /ltr/models/backbone/resnet.py

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
# from dclip.untils import tokenize
import sys
import os.path
from os import path
import glob
import numpy as np
from PIL import Image
from pytracking.utils.plotting import draw_figure, overlay_mask

# from heatmap import heat_show

import time
# t0 = time.time()
# print('{} seconds'.format(time.time() - t0))

import ltr.data.processing_utils as prutils

try:
    from ltr.cotracker2.co_tracker.cotracker.utils.visualizer import Visualizer, read_video_from_path
    print("cotracker2 load")
except:
    pass


# DinoPatch = 16
# DinoPatch = 18
DinoPatch = 27
print("DinoPatch tompnet_PT", DinoPatch)

# backbone = "VIT-B"
backbone = "VIT-L"
# backbone = "VIT-g"
print("backbone tompnet_PT", backbone)

# infer = True
# infer = False
# print("infer", infer)

# 1. Import depth
import math
import matplotlib

from ltr.models.tracking.tompnet import ToMPnet


device = "cuda" if torch.cuda.is_available() else "cpu"


class ToMPnet_PT(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer,
        bkMlp,
        JEPA_predictor_cls,
        JEPA_predictor_breg,
        cotracker_model,
        SideNetwork_D,
        SideNetwork_U,
        PTrackAttentionModel,
        PointEmbeddingNetwork,
        TFEcatmlp,
        PTlabelEmbeddingNetwork
        ):
        super().__init__()

        print('ToMPnet_PT')

        self.JEPA_predictor_cls = JEPA_predictor_cls
        self.JEPA_predictor_breg = JEPA_predictor_breg

        self.bkMlp = bkMlp
        self.cotracker_model = cotracker_model  # Store the CoTracker model
        self.SideNetwork_D = SideNetwork_D
        self.SideNetwork_U = SideNetwork_U 
        self.PTrackAttentionModel = PTrackAttentionModel
        self.PointEmbeddingNetwork = PointEmbeddingNetwork
        self.TFEcatmlp = TFEcatmlp
        self.PTlabelEmbeddingNetwork = PTlabelEmbeddingNetwork

        self.use_ltrblabels = 0
        print('self.use_ltrblabels', self.use_ltrblabels)

        # self.use_dino_interfeat = 0
        self.use_dino_interfeat = 1
        print('self.use_dino_interfeat', self.use_dino_interfeat)

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

        self.show = 0
        # self.show = 1

        self.grid_size = 100 # No Use

        # self.point_num = 64
        self.point_num = 128

        self.labels_num = 2

        self.last_N = 1

        self.PToutgauss = True

        self.gaud = 16 # PT uses the original resolution of ToMP input frames

        print('self.grid_size', self.grid_size)
        print('self.point_num', self.point_num)
        print('self.gaud', self.gaud)
        print('self.labels_num', self.labels_num)
        print('self.last_N', self.last_N)
        print('self.PToutgauss', self.PToutgauss)

    def gen_gaussian_label_function_center(self, target_center, last_element_test_visibility):

        B, T, _ = target_center.shape  # B is batch size, T is number of targets
        flat_target_center = target_center.view(-1, 2)  # Flatten the target_center

        gauss_label = prutils.gaussian_label_function_center(
            flat_target_center,
            0.9,
            (1, 1),
            (DinoPatch, DinoPatch), 
            (DinoPatch*self.gaud, DinoPatch*self.gaud),
            end_pad_if_even=True
        )

        # Reshape gauss_label to match the expected output shape
        gauss_label = gauss_label.view(B, T, gauss_label.shape[-2], gauss_label.shape[-1])

        # Handling visibility: inverse the visibility to create a mask for absence
        target_absent = 1 - last_element_test_visibility.view(B, T, 1, 1).float()  # Create a broadcastable mask
        gauss_label *= (1 - target_absent)  # Apply the absent mask

        # print("target_absent.shape", target_absent.shape)  # Should now show the expected shape [B, T, 18, 18]
        # print("gauss_label.shape", gauss_label.shape)  # Should now show the expected shape [B, T, 18, 18]
        # input()

        return gauss_label

    def tensorim_2_cvim(self, ts_im, norm, RGB2BGR=0):

        tmp_ts_im = ts_im.clone()

        if norm:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            _mean = torch.Tensor(mean).view(-1, 1, 1)
            _std = torch.Tensor(std).view(-1, 1, 1)

            try:
                tmp_ts_im *= _std.cuda()
                tmp_ts_im += _mean.cuda()
            except:
                ts_a = tmp_ts_im.clone()
                ts_a *= _std.cuda()
                ts_a += _mean.cuda()
                tmp_ts_im = ts_a

        try:
            tmp_ts_im_numpy = tmp_ts_im.cpu().numpy()
        except:
            tmp_ts_im_numpy = tmp_ts_im.cpu().detach().numpy()
        
        cvim = tmp_ts_im_numpy.transpose(1, 2, 0)

        if RGB2BGR:
            cvim = cv2.cvtColor(cvim, cv2.COLOR_RGB2BGR)

        return cvim

            
    def infer_cotracker(self, test_imgs_all, \
        test_bb_first,
        test_bb_mid,
        test_label_first, 
        test_label_mid,
        filtered_points = None
        ):


        def prepare_queries(bb_first, bb_mid, device):
            # print("Initial bb_first.shape:", bb_first.shape)  # Expecting [1, B, 4]
            # print("Initial bb_mid.shape:", bb_mid.shape)      # Expecting [1, B, 4]

            # print("Initial bb_first:", bb_first)  # Expecting [1, B, 4]
            # print("Initial bb_mid:", bb_mid)      # Expecting [1, B, 4]

            # Assuming self.point_num = 128, so half is 64
            num_samples_per_box = self.point_num // 2

            # Squeeze out the first dimension
            bb_first = bb_first.squeeze(0)  # Now expected [B, 4]
            bb_mid = bb_mid.squeeze(0)      # Now expected [B, 4]

            # print("After squeeze bb_first.shape:", bb_first.shape)  # Expecting [B, 4]
            # print("After squeeze bb_mid.shape:", bb_mid.shape)      # Expecting [B, 4]

            B, _ = bb_first.shape

            # Generate random offsets in [0, w] and [0, h]
            random_x_offsets_first = torch.rand(B, num_samples_per_box, device=device) * bb_first[:, 2].unsqueeze(-1)
            random_y_offsets_first = torch.rand(B, num_samples_per_box, device=device) * bb_first[:, 3].unsqueeze(-1)

            random_x_offsets_mid = torch.rand(B, num_samples_per_box, device=device) * bb_mid[:, 2].unsqueeze(-1)
            random_y_offsets_mid = torch.rand(B, num_samples_per_box, device=device) * bb_mid[:, 3].unsqueeze(-1)

            # Compute sampled coordinates
            sampled_x_first = bb_first[:, 0].unsqueeze(-1) + random_x_offsets_first
            sampled_y_first = bb_first[:, 1].unsqueeze(-1) + random_y_offsets_first

            sampled_x_mid = bb_mid[:, 0].unsqueeze(-1) + random_x_offsets_mid
            sampled_y_mid = bb_mid[:, 1].unsqueeze(-1) + random_y_offsets_mid

            # IDs
            ids_first = torch.zeros_like(sampled_x_first, device=device)
            ids_mid = 4 * torch.ones_like(sampled_x_mid, device=device)

            # Stack to get queries
            queries_first = torch.stack([ids_first, sampled_x_first, sampled_y_first], dim=-1)  # [B, 64, 3]
            queries_mid = torch.stack([ids_mid, sampled_x_mid, sampled_y_mid], dim=-1)          # [B, 64, 3]

            # Concatenate along sample dimension: [B, 128, 3]
            queries = torch.cat([queries_first, queries_mid], dim=1)

            # print("queries.shape:", queries.shape)  # Expecting [B, 128, 3]

            # print("queries:", queries)  # Expecting

            # input()

            return queries


        def tensor_unnorm(ts_im, norm, RGB2BGR=0):
            # Ensure the input tensor ts_im has the expected dimensions
            if ts_im.dim() != 5:
                raise ValueError("Expected a 5D tensor with dimensions (N, B, C, H, W)")

            # Clone the tensor to avoid modifying the original data
            tmp_ts_im = ts_im.clone()

            if norm:
                # Define mean and std for normalization
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=ts_im.device).view(1, 1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=ts_im.device).view(1, 1, 3, 1, 1)
                # Normalize the image, applying mean and std across the correct dimensions
                tmp_ts_im = (tmp_ts_im * std) + mean
                # tmp_ts_im = (tmp_ts_im -mean )/std

            if RGB2BGR:
                # Permute the color channels from RGB to BGR
                tmp_ts_im = tmp_ts_im[:, :, [2, 1, 0], :, :]  # Assume C is the 3rd dimension (0-indexed)

            # Find the min and max values of the tensor
            min_val = torch.min(tmp_ts_im)
            max_val = torch.max(tmp_ts_im)

            # Perform min-max normalization to scale the image to [0, 255]
            # This scales the tensor's values from the range [min_val, max_val] to [0, 1]
            normalized_tensor = (tmp_ts_im - min_val) / (max_val - min_val)

            # Scale to [0, 255] and convert to uint8
            tmp_ts_im_ = (normalized_tensor * 255)

            return tmp_ts_im_

        # def infer_cotracker_model(label_first_batch, label_last_batch, imgs_all, bb_first, device, filtered_points=None):
        def infer_cotracker_model(label_first_batch, label_last_batch, imgs_all, bb_first, bb_mid, device, filtered_points=None):

            # queries = prepare_queries(bb_first, device) # Expecting [B, 128, 3]

            queries = prepare_queries(bb_first, bb_mid, device) # Expecting [B, 128, 3]

            ################  vis point only 64
            if filtered_points != None:
                half_num = queries.shape[1] // 2  # For example, if queries.shape[1] is 128, half_num = 64
                num_filtered = min(filtered_points.shape[1], half_num)

                # Replace only up to num_filtered points in the first half of queries
                queries[:, :num_filtered, :] = filtered_points[:, :num_filtered, :]

                # print("queries 2", queries)

            ################ B


            # print("imgs_all.shape", imgs_all.shape) # torch.Size([8, 1, 3, 432, 432])
            imgs_all = imgs_all.permute(1, 0, 2, 3, 4)
            # imgs_all = imgs_all.squeeze(1).unsqueeze(0)
            # print("imgs_all.shape", imgs_all.shape) # torch.Size([1, 8, 3, 432, 432])

            grid_size = self.grid_size 

            imgs_all_unnorm = tensor_unnorm(imgs_all, 1, 0)


            pred_tracks, pred_visibility = self.cotracker_model(self.SideNetwork_D, self.SideNetwork_U, label_first_batch, label_last_batch, imgs_all_unnorm, \
                                                                    queries=queries, grid_size=grid_size, backward_tracking=False)

            # print("pred_tracks.shape", pred_tracks.shape) # torch.Size([1, 24, 128, 2])
            # print("pred_visibility.shape", pred_visibility.shape) # torch.Size([1, 24, 128])
            # print("torch.sum(pred_visibility[:,-1,:])", torch.sum(pred_visibility[:,-1,:]))


        if self.labels_num == 2:
            # 2 labels
            test_label_first = self.PTlabelEmbeddingNetwork(test_label_first).unsqueeze(1)
            test_label_mid = self.PTlabelEmbeddingNetwork(test_label_mid).unsqueeze(1)
            # print("test_label_first.shape:", test_label_first.shape) # torch.Size([B, 1, 128, 96, 128])
            # print("test_label_mid.shape:", test_label_mid.shape) # torch.Size([B, 1, 128, 96, 128])

        if eva_time:
            print("infer_cotracker_model test s")
            start.record()

        # print("test_imgs_all.shape", test_imgs_all.shape) 

        test_imgs_all_ = test_imgs_all.permute(1,0,2,3,4)
        # Copy the first image in each sequence to the second and third positions
        # print("test_imgs_all_.shape", test_imgs_all_.shape)

        # if self.PTcopyF:
        if self.training or test_label_mid.shape[0]>=4:
            test_imgs_all_[:, 1, :, :, :] = test_imgs_all_[:, 0, :, :, :]
            test_imgs_all_[:, 2, :, :, :] = test_imgs_all_[:, 0, :, :, :]
            test_imgs_all_[:, -2, :, :, :] = test_imgs_all_[:, -1, :, :, :]

        test_imgs_all = test_imgs_all_.permute(1,0,2,3,4)

        # print("Modified test_imgs_all.shape", test_imgs_all.shape)

        ##################################

        pred_tracks_test, pred_visibility_test = infer_cotracker_model(test_label_first, test_label_mid, \
                                                            test_imgs_all, test_bb_first, test_bb_mid, test_imgs_all.device, filtered_points)    
        ##################################   

        # print("CoTracker model inference completed for testing data")
        # input()

        if eva_time:
            end.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            print("infer_cotracker_model test", start.elapsed_time(end))

        return pred_tracks_test, pred_visibility_test


    def PT_prompt(self, test_tracks_split):
        test_tracks_transformed = self.PointEmbeddingNetwork(test_tracks_split)

        # return train_tracks_transformed, test_tracks_transformed
        return test_tracks_transformed


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


    def forward(self, 
        train_imgs,
        test_imgs,
        train_imgs_all,
        test_imgs_all,
        train_bb,
        test_bb_first,
        test_bb_mid,
        test_label_first,
        test_label_mid,
        *args, **kwargs):


        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            train_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
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

        ####################################################
        pred_tracks_test, pred_visibility_test = \
            self.infer_cotracker(test_imgs_all, test_bb_first, test_bb_mid,  test_label_first, test_label_mid)
        ####################################################

        last_element_test_tracks = pred_tracks_test[:, -1, :, :]
        last_element_test_visibility = pred_visibility_test[:, -1, :]

        ###########################################################

        test_tracks_split = self.gen_gaussian_label_function_center(last_element_test_tracks.contiguous(), last_element_test_visibility.contiguous())

        test_tracks_transformed = self.PT_prompt(test_tracks_split)

        if not self.use_dino_interfeat:
            train_feat_head = self.extract_dino_features_spatial(train_ims_reshape)
            test_feat_head = self.extract_dino_features_spatial(test_ims_reshape)
        else:
            train_feat_head = self.extract_dino_features_spatial_intermediate_layers(train_ims_reshape, None)
            test_feat_head = self.extract_dino_features_spatial_intermediate_layers(test_ims_reshape, None)


        target_scores, bbox_preds, target_scores_PT, bbox_preds_PT = self.head(train_feat_head, test_feat_head, train_bb, test_tracks_transformed, \
        self.PTrackAttentionModel, self.TFEcatmlp,  \
        self.JEPA_predictor_cls, self.JEPA_predictor_breg,  \
        *args, **kwargs)

        return target_scores, bbox_preds, target_scores_PT, bbox_preds_PT

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
def tompnet50_PT(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
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

    cotracker_model = backbones.cotracker_predictor()

    bkMlp = heads.bkMlp()

    JEPA_predictor_cls = heads.JEPA_predictor_cls()
    JEPA_predictor_breg = heads.JEPA_predictor_breg()

    SideNetwork_D = heads.SideNetwork_D()

    SideNetwork_U = heads.SideNetwork_U()

    PTrackAttentionModel = heads.PTrackAttentionModel()

    PointEmbeddingNetwork = heads.PointEmbeddingNetwork()

    TFEcatmlp = heads.TFEcatmlp()

    PTlabelEmbeddingNetwork = heads.PTlabelEmbeddingNetwork()

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))


    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    # bkMlp
    head_feature_extractor = clf_features.residual_bottleneck_reshape(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    # head_feature_extractor = clf_features.residual_bottleneck_reshape_768(feature_dim=feature_dim,
    #                                                           num_blocks=head_feat_blocks, l2norm=head_feat_norm,
    #                                                           final_conv=final_conv, norm_scale=norm_scale,
    #                                                           out_dim=out_feature_dim)


    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)


    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet_PT(feature_extractor=backbone_net, head=head, head_layer=head_layer,
    bkMlp = bkMlp,
    JEPA_predictor_cls = JEPA_predictor_cls,
    JEPA_predictor_breg = JEPA_predictor_breg,
    cotracker_model = cotracker_model,
    SideNetwork_D = SideNetwork_D,
    SideNetwork_U = SideNetwork_U,
    PTrackAttentionModel = PTrackAttentionModel,
    PointEmbeddingNetwork = PointEmbeddingNetwork,
    TFEcatmlp = TFEcatmlp,
    PTlabelEmbeddingNetwork = PTlabelEmbeddingNetwork,
    )
    return net














