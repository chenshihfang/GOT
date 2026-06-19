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
###
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

import time

import ltr.data.processing_utils as prutils

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

CLIP_model_name = "empty"

use_clip = 1
use_open_clip = 0

use_autocast = 0

print("use_clip", use_clip)
print("use_autocast", use_autocast)

# # ''' CLIP

if use_clip or use_open_clip:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading CLIP")
if use_clip:
    print("import clip")
    import clip

    CLIP_model_name = 'ViT-L/14@336px'

    print("loading clip")
    clip_model, clip_model_preprocess = clip.load(CLIP_model_name, device=device)
    
    print("loading CLIP done")

# ToMPnet_CLIP ToMPnet
class ToMPnet_CLIP(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, 
            feature_extractor,
            head,
            head_layer,
            test_fuse,
            test_fuse_head,
            test_shrink,
            hint_test_feat_head,
            bkMlp, # DINO
                        ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))


        #### DINO

        self.bkMlp = bkMlp
        self.normMethod = 2
        print('self.normMethod', self.normMethod)
        #### DINO

        ### SRopt
        self.test_fuse = test_fuse
        self.test_fuse_head = test_fuse_head
        self.test_shrink = test_shrink
        self.hint_test_feat_head = hint_test_feat_head
        ######################################################################

        ### train
        self.wo_shr = 1
        print("self.wo_shr", self.wo_shr)
        
        ### train
        
        ### test
        self.infer = 1
        self.show = 0
        print("self.infer", self.infer) 
        if self.infer:

            self.sim_mean_early = 0 # 0
            self.wimask = 0
            if self.wimask:
                self.wimask_val = 0.0

            self.simTopN = 0
            self.topN = 5

            self.clip_float = 1 # what Interacting_with_CLIP.ipynb do

            self.hisclip_use_fir_only = 0 ### CLIP

            self.ini_mask_use_fir_only = 0 ### inimaks

            self.prune_highlight_center_only = 1

            self.cenArV = 0.5
            self.cenarea = 0 # 0 as cross

            self.adahightlight = 0
            self.hightlight = 1
            self.prune = 0
            torch.set_printoptions(precision=3, sci_mode=False)

            print("self.wimask", self.wimask)
            print("self.prune", self.prune)
            print("self.hightlight", self.hightlight)
            print("self.simTopN", self.simTopN)
            print("self.topN", self.topN)
            print("self.prune_highlight_center_only", self.prune_highlight_center_only)
            print("self.cenArV", self.cenArV)
            print("self.cenarea", self.cenarea)
            print("CLIP_model_name", CLIP_model_name)
            print("self.sim_mean_early", self.sim_mean_early)
            print("self.clip_float", self.clip_float)
            
            if use_open_clip:
                print("pretrained_data", pretrained_data)
        ### test

        ######################################################################

        self.show_test_train_in = 0
        self.show_pretrain = 0
        self.show_train_input = 0

        if self.show:
            self.show_test_train_in = 1
            self.show_pretrain = 1
            self.show_train_input = 0
        
        ### CLIP

        self.dclip_crop = 1 # crop

        self.bg_info = 0 # crop with bg only for train pipeline

        self.train_crop_feat = 1 # train with crop

        self.clip_bk_only = 0

    def extract_dino_features_spatial(self, images, mode = 1):

        model_name = "dinov2_vitl14"

        if model_name == "dinov2_vits14":
            feat_dim = 384 # vits14
        elif model_name == "dinov2_vitb14":
            feat_dim = 768 # vitb14
        elif model_name == "dinov2_vitl14":
            feat_dim = 1024 # vitl14
        elif model_name == "dinov2_vitg14":
            feat_dim = 1536 # vitg14

        image_shape = images.shape[-1]

        patch_size = self.feature_extractor.patch_size # patchsize=14
        patch_h  = image_shape//patch_size
        patch_w  = image_shape//patch_size

        with torch.no_grad():
            features_dict = self.feature_extractor.forward_features(images)
            features = features_dict['x_norm_patchtokens']

        total_features = features.reshape(-1, patch_h, patch_w, feat_dim) #4(*H*w, 1024)

        if self.normMethod == 1:
            total_features = self.feature_extractor.norm(total_features)
        elif self.normMethod == 2:
            total_features /= total_features.norm(dim=-1, keepdim=True)

        total_features = total_features.permute(0,3,1,2) # [B, 16, 16 ,1536] # torch.Size([1, 18, 18, 1536])

        if mode == 1:
            total_features = self.bkMlp(total_features)
        elif mode == 2:
            total_features = self.bkMlp(total_features)

        total_features = F.adaptive_avg_pool2d(total_features, 27)

        return total_features

    def featcoors_2_crops(self, pre_boxwh_list, prob_map_small, test_imgs, target_candidate_coords, target_candidate_scores, maxCanNum):

        # [y, x]
        maxima_num = target_candidate_coords.shape[0]

        res_w, res_h = test_imgs.shape[-2], test_imgs.shape[-1]
        fea_w, fea_h = prob_map_small.shape[-2], prob_map_small.shape[-1]
        scale_w, scale_h = res_w/fea_w, res_h/fea_h

        crop_ims = []

        maxima_num = min(maxCanNum, maxima_num)

        for i in range(0, maxima_num):

            [pre_w, pre_h] = pre_boxwh_list[i]

            cen_x = int((target_candidate_coords[i][1].item())*scale_w)
            cen_y = int((target_candidate_coords[i][0].item())*scale_h)

            x, y, w, h = np.floor(cen_x - (pre_w/2)), np.floor(cen_y - (pre_h/2)), np.ceil(pre_w), np.ceil(pre_h)

            x = max(0, x)
            y = max(0, y)

            crop_im = self.im_2_crop_tensor(test_imgs, torch.FloatTensor([x,y,w,h]))

            crop_ims.append(crop_im)

            if self.show:

                cv_crop_im = self.tensorim_2_cvim(crop_im, norm=1, RGB2BGR=0) 
                cv2.imshow(str(i) + "cv_crop_im", cv_crop_im)

        crop_ims_tensor = torch.cat([torch.unsqueeze(dclip_train_imgs_, 0) for dclip_train_imgs_ in crop_ims], 0)

        ### parallel batch
        crop_clip_feat = self.CLIP_classical_model(crop_ims_tensor)

        return crop_clip_feat, crop_ims


    def hilight_feat(self, test_imgs, dclip_train_imgs, pre_boxwh_list, prob_map_small, target_candidate_coords, target_candidate_scores, history_crop_clip_feat, simvThr, maxCanNum):

            if self.show:
                cv_prob_map_small_ini = self.tensorim_2_cvim(prob_map_small[0].detach(), norm=0, RGB2BGR=1)
                cv2.namedWindow("cv_prob_map_small_ini", cv2.WINDOW_NORMAL)
                cv2.imshow("cv_prob_map_small_ini", cv_prob_map_small_ini)
                
            if target_candidate_coords.shape[0] != 0:
                
                crop_clip_feat, crop_ims = self.featcoors_2_crops(pre_boxwh_list, prob_map_small, test_imgs, target_candidate_coords, target_candidate_scores, maxCanNum)

                similarity = self.clip_sim(history_crop_clip_feat, crop_clip_feat)

                if self.sim_mean_early == 0:
                    similarity = torch.mean(similarity, 1)

                similarity_mean_soft = similarity


                if self.wimask:
                    prob_map_small = torch.ones_like(prob_map_small)
                    prob_map_small = prob_map_small * self.wimask_val # inc 01
                
                for idx, val in enumerate(similarity_mean_soft):
 
                    sim_value = val.item()

                    if sim_value > simvThr:
                        if self.hightlight:
                            prob_map_val = 1

                            if self.adahightlight:
                                prob_map_max = torch.max(prob_map_small)
                                prob_map_val = prob_map_max - (prob_map_max*0.02*idx)
                        else:
                            continue
                    else:
                        if similarity_mean_soft.shape[0] <=2:
                            continue
                        if self.prune:
                            prob_map_val = torch.min(prob_map_small)
                        else:
                            continue

                    sms_arg_coor = target_candidate_coords[idx]

                    w, h = pre_boxwh_list[idx]

                    ### move following to left for hight-prune
                    y_idx = sms_arg_coor[0].item()
                    x_idx = sms_arg_coor[1].item() 

                    if not self.prune_highlight_center_only:
                        mask_x, mask_y = x_idx, y_idx
                        mask_w, mask_h = 3, 3
                        mask_start_x, mask_start_y = max(0, mask_x - mask_w//2), max(0, mask_y - mask_h//2)
                        mask_end_x, mask_end_y = mask_start_x + mask_w, mask_start_y + mask_h

                        # square
                        if self.cenarea:
                            prob_map_small[0][0][mask_start_y:mask_end_y, mask_start_x:mask_end_x] = self.cenArV / 2

                        # cross
                        prob_map_small[0][0][y_idx-1, x_idx] = self.cenArV
                        prob_map_small[0][0][min(y_idx+1, 17), x_idx] = self.cenArV
                        prob_map_small[0][0][y_idx, x_idx-1] = self.cenArV
                        prob_map_small[0][0][y_idx, min(x_idx+1, 17)] = self.cenArV

                    prob_map_small[0][0][y_idx, x_idx] = prob_map_val

            else:
                crop_clip_feat = torch.empty((1,1))
                crop_ims = torch.empty((1,1))

            # input()

            if self.show_pretrain:

                cv_prob_map_small = self.tensorim_2_cvim(prob_map_small[0].detach(), norm=0, RGB2BGR=1)

                prob_map = F.interpolate(prob_map_small, scale_factor=352/22, mode='nearest')  #

                prob_map = prob_map.expand(-1, 3, -1, -1) # 3 1 288 288 -> # 3 3 288 288

                ### mask
                test_imgs_ = self.resize_5d_tensor(test_imgs, size = 352)
                mask_test_imgs = test_imgs_ + prob_map

                cv2.imshow("mask_test_imgs", self.tensorim_2_cvim(mask_test_imgs[0].detach(), norm=1, RGB2BGR=0))
                cv_prob_map = self.tensorim_2_cvim(prob_map[0].detach(), norm=0, RGB2BGR=1)
                cv2.imshow("prob_map", cv_prob_map)

            if self.show_test_train_in:
                cv2.waitKey(1)

            return prob_map_small, crop_clip_feat, crop_ims

    def SRoptModel(self, test_imgs, test_feat_enc, dclip_train_imgs, *args, **kwargs):

        if self.show_test_train_in:

            cv_dclip_train_imgs1 = self.tensorim_2_cvim(dclip_train_imgs[0][0].detach(), norm=1, RGB2BGR=0)
            cv_dclip_train_imgs2 = self.tensorim_2_cvim(dclip_train_imgs[0][1].detach(), norm=1, RGB2BGR=0)
            cv2.imshow("cv_dclip_train_imgs1", cv_dclip_train_imgs1)
            cv2.imshow("cv_dclip_train_imgs2", cv_dclip_train_imgs2)

        if self.infer:

            if self.ini_mask_use_fir_only:
                [train_feat, test_feat] = test_feat_enc
                train_feat[1] = train_feat[0]
                test_feat_enc = [train_feat, test_feat]

            prob_map, prob_map_small = self.test_feat_2_prob_map(test_feat_enc)
            prob_map = torch.unsqueeze(prob_map, 0)
            prob_map_small = torch.unsqueeze(torch.squeeze(prob_map_small, 1), 0)

            return prob_map_small # infer

        else:

            prob_map, prob_map_small = self.test_feat_2_prob_map(test_feat_enc)
            prob_map = torch.unsqueeze(prob_map, 0)
            prob_map_small = torch.unsqueeze(torch.squeeze(prob_map_small, 1), 0)
            mask_test_imgs = test_imgs * prob_map # no use

            return mask_test_imgs, prob_map_small # train

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


    def resize_5d_tensor(self, dclip_test_imgs, size = 512):

        tmp_dclip_test_imgs = dclip_test_imgs
        if dclip_test_imgs.dim()==4:
            tmp_dclip_test_imgs = torch.unsqueeze(tmp_dclip_test_imgs, 0)

        dclip_test_imgs_reshape = tmp_dclip_test_imgs.reshape(-1, *tmp_dclip_test_imgs.shape[-3:])

        test_imgs_resize = F.interpolate(dclip_test_imgs_reshape, size=(size, size), mode='bilinear')

        test_imgs_resize_reshape = test_imgs_resize.reshape(*tmp_dclip_test_imgs.shape[0:2], *test_imgs_resize.shape[-3:])

        if dclip_test_imgs.dim()==4:
            test_imgs_resize_reshape = torch.squeeze(test_imgs_resize_reshape, 0)

        return test_imgs_resize_reshape

    def test_feat_2_prob_map(self, test_feat_enc, *args, **kwargs):

        ### test_shrink
        # w shrink
        if not self.wo_shr:
            test_feat_enc = self.test_shrink(test_feat_enc)

        test_feat_merge = self.test_fuse(test_feat_enc)


        test_feat_enc_clone, prob_map_small = self.test_fuse_head(test_feat_merge)

        ## SRopt sigmoid
        test_feat_enc_shrink_sig = test_feat_enc_clone

        prob_map = test_feat_enc_shrink_sig.expand(-1, 3, -1, -1) # 3 1 288 288 -> # 3 3 288 288

        return prob_map, prob_map_small

    def im_2_crop_tensor(self, tmp_imm_0, gt):

        # back to shape
        im_shape = tmp_imm_0.shape[-1]

        x, y, w, h = gt.int()
        imm_crop = fn.crop(tmp_imm_0, y, x, h, w)

        if imm_crop.dim() != 4:
            imm_crop_usqz = torch.unsqueeze(imm_crop, 0)
        else:
            imm_crop_usqz = imm_crop


        imm_crop_resize = F.interpolate(imm_crop_usqz, size=(im_shape, im_shape), mode='bilinear')
        imm_crop_resize_sqz = torch.squeeze(imm_crop_resize, 0)

        return imm_crop_resize_sqz


    def CLIP_classical_model(self, im):

        if isinstance(im, str):
            text = im

            if use_clip:
                text_token = clip.tokenize([text]).to(device)
            elif use_open_clip:
                text_token = tokenizer([text]).to(device)

            with torch.no_grad():
                text_features = clip_model.encode_text(text_token)

                if self.clip_float:
                    return text_features.float()
                else:
                    return text_features

        else:

            if im.shape[-1] != 224:

                lst_224 = [ 'RN50' , 'RN101' , 'ViT-B/32' , 'ViT-B/16' , 'ViT-L/14', 'ViT-bigG-14', \
                            'ViT-g-14',]

                if CLIP_model_name in lst_224:
                    size_ = 224
                elif CLIP_model_name == 'RN50x4':
                    size_ = 288
                elif CLIP_model_name == 'RN50x16':
                    size_ = 384
                elif CLIP_model_name == 'RN50x64':
                    size_ = 448
                elif CLIP_model_name == 'ViT-L/14@336px':
                    size_ = 336

                im_resize = self.resize_5d_tensor(im, size = size_)

            with torch.no_grad():
                if im.dim()==5:
                    image_features = clip_model.encode_image(im_resize[0])
                else:
                    image_features = clip_model.encode_image(im_resize)
                
            if self.clip_float:
                return image_features.float()
            else:
                return image_features

    def clip_sim(self, history_crop_clip_feat, predict_im_clip_feat):

        ref_feats_ = history_crop_clip_feat
        maskSR_feats_ = predict_im_clip_feat

        ref_feats_ /= ref_feats_.norm(dim=-1, keepdim=True)
        maskSR_feats_ /= maskSR_feats_.norm(dim=-1, keepdim=True)

        similarity = (100.0 * maskSR_feats_ @ ref_feats_.T)
        if self.sim_mean_early == 1:
            similarity = torch.mean(similarity, 1)

        similarity = similarity.softmax(dim=0)

        return similarity

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

        dclip_train_imgs = train_imgs.clone()
        if self.dclip_crop:
            for idxii, imii in enumerate(dclip_train_imgs):
                for idxi, imi in enumerate(imii):

                    if self.bg_info:
                        try:
                            [x,y,w,h] = train_bb[idxii][idxi]
                            bb = [x-0.25*w,y-0.25*h,w*1.5,h*1.5]
                            bb = [max(0, ele) for ele in bb]
                            dclip_train_imgs[idxii][idxi] = self.im_2_crop_tensor(imi, torch.FloatTensor(bb))
                        except:
                            dclip_train_imgs[idxii][idxi] = self.im_2_crop_tensor(imi, train_bb[idxii][idxi])
                    else:
                        dclip_train_imgs[idxii][idxi] = self.im_2_crop_tensor(imi, train_bb[idxii][idxi])

                    if self.show_train_input:
                        cv2.imshow(str(idxii)+str(idxi)+"imm_crop", self.tensorim_2_cvim(dclip_train_imgs[idxii][idxi], norm=1, RGB2BGR=1))

            if self.show_train_input:
                for idxii, imii in enumerate(test_imgs):
                    for idxi, imi in enumerate(imii):
                        # cv2.imshow(str(idxi)+"immtest", self.tensorim_2_cvim(dclip_test_imgs[idxii][idxi], norm=1))
                        cv2.imshow(str(idxii)+str(idxi)+"immtest", self.tensorim_2_cvim(test_imgs[idxii][idxi], norm=1, RGB2BGR=1))
           
        # Extract backbone features
        train_ims = train_imgs.reshape(-1, *train_imgs.shape[-3:])
        test_ims = test_imgs.reshape(-1, *test_imgs.shape[-3:])
        train_ims_reshape = self.resize_5d_tensor(train_ims, size = 378)
        test_ims_reshape = self.resize_5d_tensor(test_ims, size = 378)
        train_feat_head = self.extract_dino_features_spatial(train_ims_reshape, mode = 1)
        test_feat_head = self.extract_dino_features_spatial(test_ims_reshape, mode = 1)

        #################################### SRopt

        ### SRopt input feat
        if self.train_crop_feat:

            ### wo filter_initializer
            dclip_train_imgs = train_imgs.reshape(-1, *dclip_train_imgs.shape[-3:])
            dclip_train_imgs = self.resize_5d_tensor(dclip_train_imgs, size = 378)

            dc_train_feat_head = self.extract_dino_features_spatial(dclip_train_imgs, mode = 2)
            ### wo filter_initializer
            
            opt_train_feat_head = dc_train_feat_head.clone()
        else:
            opt_train_feat_head = train_feat_head.clone()
            
        opt_test_feat_head = test_feat_head.clone()

        dclip_test_imgs_res, prob_map_small = self.SRoptModel(test_imgs, [opt_train_feat_head, opt_test_feat_head], dclip_train_imgs)

        ### train fuse w test
        prob_map_small = prob_map_small.to(test_feat_head.device)
        prob_map_small_swp = prob_map_small.permute(1,0,2,3)

        # forward part

        ### hint_test_feat_head
        prob_map_exp = prob_map_small_swp.expand(-1, test_feat_head.shape[1], -1, -1)
        test_feat_head = self.hint_test_feat_head(test_feat_head, prob_map_exp)

        ### train fuse w test

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb, *args, **kwargs)

        return test_scores, bbox_preds, prob_map_small #, hint_bbox_preds

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

# tompnet50_CLIP tompnet50
@model_constructor
def tompnet50_CLIP(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
              final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
              num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):

    # Backbone

    backbone_net = backbones.dinov2("dinov2_vitl14")

    bkMlp = heads.bkMlp()

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck_reshape(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)


    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    ### SRopt
    test_fuse = heads.test_fuse()

    test_fuse_head = heads.test_fuse_head()

    test_shrink = heads.test_shrink()

    hint_test_feat_head = heads.hint_test_feat_head()

    # ToMP network
    net = ToMPnet_CLIP(
    feature_extractor=backbone_net, 
    head=head, 
    head_layer=head_layer,
    test_fuse = test_fuse,
    test_fuse_head = test_fuse_head,
    test_shrink = test_shrink,
    hint_test_feat_head = hint_test_feat_head,
    bkMlp = bkMlp,
    )
    return net


class ToMPnet(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer,
        bkMlp
        ):
        super().__init__()

        self.bkMlp = bkMlp

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

        self.normMethod = 2

        print('self.normMethod', self.normMethod)

        self.show = 0

    def extract_dino_features_spatial(self, images):

        model_name = "dinov2_vitl14"

        if model_name == "dinov2_vits14":
            feat_dim = 384 # vits14
        elif model_name == "dinov2_vitb14":
            feat_dim = 768 # vitb14
        elif model_name == "dinov2_vitl14":
            feat_dim = 1024 # vitl14
        elif model_name == "dinov2_vitg14":
            feat_dim = 1536 # vitg14

        image_shape = images.shape[-1]

        patch_size = self.feature_extractor.patch_size # patchsize=14

        patch_h  = image_shape//patch_size
        patch_w  = image_shape//patch_size

        if use_autocast:
            with torch.no_grad(), autocast():
                features_dict = self.feature_extractor.forward_features(images)
                features = features_dict['x_norm_patchtokens']
        else:
            with torch.no_grad():
                features_dict = self.feature_extractor.forward_features(images)
                features = features_dict['x_norm_patchtokens']

        total_features = features.reshape(-1, patch_h, patch_w, feat_dim) #4(*H*w, 1024)

        if self.normMethod == 1:
            total_features = self.feature_extractor.norm(total_features)
        elif self.normMethod == 2:
            total_features /= total_features.norm(dim=-1, keepdim=True)

        total_features = total_features.permute(0,3,1,2) # [B, 16, 16 ,1536] # torch.Size([1, 18, 18, 1536])

        total_features = self.bkMlp(total_features)

        total_features = F.adaptive_avg_pool2d(total_features, 27)

        return total_features


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


    def resize_5d_tensor(self, dclip_test_imgs, size = 512):

        tmp_dclip_test_imgs = dclip_test_imgs
        if dclip_test_imgs.dim()==4:
            tmp_dclip_test_imgs = torch.unsqueeze(tmp_dclip_test_imgs, 0)

        dclip_test_imgs_reshape = tmp_dclip_test_imgs.reshape(-1, *tmp_dclip_test_imgs.shape[-3:])

        test_imgs_resize = F.interpolate(dclip_test_imgs_reshape, size=(size, size), mode='bilinear')

        test_imgs_resize_reshape = test_imgs_resize.reshape(*tmp_dclip_test_imgs.shape[0:2], *test_imgs_resize.shape[-3:])

        if dclip_test_imgs.dim()==4:
            test_imgs_resize_reshape = torch.squeeze(test_imgs_resize_reshape, 0)

        return test_imgs_resize_reshape


    def im_2_crop_tensor(self, tmp_imm_0, gt):

        im_shape = 288

        x, y, w, h = gt.int()
        imm_crop = fn.crop(tmp_imm_0, y, x, h, w)

        if imm_crop.dim() != 4:
            imm_crop_usqz = torch.unsqueeze(imm_crop, 0)
        else:
            imm_crop_usqz = imm_crop

        imm_crop_resize = F.interpolate(imm_crop_usqz, size=(im_shape, im_shape), mode='bilinear')
        imm_crop_resize_sqz = torch.squeeze(imm_crop_resize, 0)

        return imm_crop_resize_sqz
        

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


        train_ims_reshape = self.resize_5d_tensor(train_ims, size = 378)
        test_ims_reshape = self.resize_5d_tensor(test_ims, size = 378)

        train_feat_head = self.extract_dino_features_spatial(train_ims_reshape)
        test_feat_head = self.extract_dino_features_spatial(test_ims_reshape)

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb, *args, **kwargs)

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
    backbone_net = backbones.dinov2("dinov2_vitl14")

    bkMlp = heads.bkMlp()
    
    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck_reshape(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)


    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer,
    bkMlp = bkMlp
    )
    return net




















