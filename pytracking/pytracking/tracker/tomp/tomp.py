from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
from ltr.models.layers import activation

from collections import OrderedDict

import numpy as np
from collections import defaultdict

import sys
import os
cpu_num = 8  # Num of CPUs you want to use
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

import ltr.data.processing_utils as prutils

import cv2
import torchvision.transforms.functional as fn

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

class ToMP(BaseTracker):
    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        self.initialize_features()
        self.net = self.params.net
        tic = time.time()

        self.show = 0

        try:
            print("info['dataset']", info['dataset'])
            self.object_class = info['object_class']
            self.seq_name = info['name']
            print("self.object_class", self.object_class)
            print("self.seq_name", self.seq_name)
        except:
            pass

            
        if self.params.use_SRmask:

            self.clip_sim_upt = 1 
            try:
                self.clip_sim_upt = self.params.clip_sim_upt
                print("self.clip_sim_upt = ", self.clip_sim_upt)
            except:
                print("self.clip_sim_upt = ", self.clip_sim_upt)
                
            self.newimSimThr = 85 # 85
            try:
                self.newimSimThr = self.params.newimSimThr
                print("self.newimSimThr = self.params.newimSimThr")
            except:
                print("self.newimSimThr = ", self.newimSimThr)

            self.hcanTh = 0.05 #  0.05
            self.hcanKs = 3 # 3
            self.usefixPrewh = 0
            self.show_distractor = 0

            self.test_datasets = 1 # if VOT set as 0

            if self.net.show:
                self.test_datasets = 0

            self.wSimIm = 1
            self.hintImImOnly = 0
            self.TraMemwCLIP = 0
            self.replace_other_prompt = 0
            self.simvThr = self.params.simvThr
            self.maxCanNum = self.params.maxCanNum
            self.upt_criteria = 1

            print("self.hcanTh", self.hcanTh)
            print("self.hcanKs", self.hcanKs)
            print("self.usefixPrewh", self.usefixPrewh)
            print("self.wSimIm", self.wSimIm)
            print("self.hintImImOnly", self.hintImImOnly)
            print("self.test_datasets", self.test_datasets)
            print("self.TraMemwCLIP", self.TraMemwCLIP)
            print("self.replace_other_prompt", self.replace_other_prompt)
            print("search_area_scale", self.params.search_area_scale)
            print("target_not_found_threshold", self.params.target_not_found_threshold)
            print("self.newimSimThr", self.newimSimThr)
            print("self.maxCanNum", self.maxCanNum)
            print("self.simvThr", self.simvThr)

            # self.w_clip_sim = 1
            self.hint_test_feat_head = 1
            self.hint_test_score = 0
            self.skip_fir_4_frames = 1

            if self.test_datasets:
                try :
                    print("info['dataset']", info['dataset'])
                    self.object_class = info['object_class']
                    self.seq_name = info['name']

                    if self.replace_other_prompt and self.object_class == "other":
                        self.object_class = self.seq_name

                    print("self.object_class",  self.object_class)
                    print("self.seq_name", self.seq_name)
                except:
                    pass
                
            self.init_template_clip_fea = torch.empty((1,1))
            self.init_text_clip_fea = torch.empty((1,1))
            self.init_text_blur_clip_fea = torch.empty((1,1))
            self.init_text_object_clip_fea = torch.empty((1,1))

            self.pre_bbox_preds = torch.empty((1,1))
            self.pre_sample_coords = torch.empty((1,1))
            self.pre_score_loc = torch.empty((1,1))
            self.pre_scores_raw = torch.empty((1,1))

            self.distractor_crop_clip_fea = torch.empty((1,1))
            self.pre_w, self.pre_h = 5, 5
            self.history_crop_clip_feat = torch.empty((1,1))
            self.previous_template_bank = torch.empty((2,3), dtype=torch.int64)
            self.init_template = torch.empty((2,3), dtype=torch.int64)

            # all learned weight will load from pretrained checkpoint in model, not in init self.net
            ### clip


        self.dclip_train_imgs = torch.empty((2, 3), dtype=torch.int64)
        print("self.params.use_SRmask", self.params.use_SRmask)

        im = numpy_to_torch(image)
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
        self.target_sz = torch.Tensor([state[3], state[2]])
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else f' {self.object_id}'

        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            stride = self.params.get('feature_stride', 32)
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz
        tfs = self.params.get('train_feature_size', 18)
        stride = self.params.get('feature_stride', 16)
        self.train_img_sample_sz = torch.Tensor([tfs * stride, tfs * stride])

        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale = np.sqrt(search_area) / self.img_sample_sz.prod().sqrt()
        self.base_target_sz = self.target_sz / self.target_scale
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        init_backbone_feat = self.generate_init_samples(im)
        self.init_classifier(init_backbone_feat)

        self.logging_dict = defaultdict(list)
        self.target_scales = []
        self.target_not_found_counter = 0
        self.cls_weights_avg = None

        out = {'time': time.time() - tic}
        return out

        

    def clip_bbox_to_image_area(self, bbox, image, minwidth=10, minheight=10):
        H, W = image.shape[:2]
        x1 = max(0, min(bbox[0], W - minwidth))
        y1 = max(0, min(bbox[1], H - minheight))
        x2 = max(x1 + minwidth, min(bbox[0] + bbox[2], W))
        y2 = max(y1 + minheight, min(bbox[1] + bbox[3], H))
        return torch.Tensor([x1, y1, x2 - x1, y2 - y1])

    def encode_bbox(self, bbox):
        stride = self.params.get('feature_stride')
        output_sz = self.params.get('image_sample_size')

        shifts_x = torch.arange(
            0, output_sz, step=stride,
            dtype=torch.float32, device=bbox.device
        )
        shifts_y = torch.arange(
            0, output_sz, step=stride,
            dtype=torch.float32, device=bbox.device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        xs, ys = locations[:, 0], locations[:, 1]

        xyxy = torch.stack([bbox[:, 0], bbox[:, 1], bbox[:, 0] + bbox[:, 2],
                            bbox[:, 1] + bbox[:, 3]], dim=1)

        l = xs[:, None] - xyxy[:, 0][None]
        t = ys[:, None] - xyxy[:, 1][None]
        r = xyxy[:, 2][None] - xs[:, None]
        b = xyxy[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2).reshape(-1, 4)

        reg_targets_per_im = reg_targets_per_im / output_sz

        sz = output_sz // stride
        nb = bbox.shape[0]
        reg_targets_per_im = reg_targets_per_im.reshape(sz, sz, nb, 4).permute(2, 3, 0, 1)

        return reg_targets_per_im

    def get_sam_pos(self, scores_raw):

        target_candidate_coords, target_candidate_scores = prutils.find_local_maxima(scores_raw.squeeze(), th=self.hcanTh, ks=self.hcanKs)
        # print("target_candidate_coords yx", target_candidate_coords)
        sam_pos = target_candidate_coords.float()

        return sam_pos

    def track(self, image, info: dict = None) -> dict:

        # print("track info", info)

        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        if self.frame_num % 20 == 0:
            print(self.frame_num)
        # sys.stdout.write(str(self.frame_num))

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)
        
        # Extract classification features
        test_x = self.get_backbone_head_feat(backbone_feat)

        ### CLIP
        test_feat_head = test_x

        im_patches = im_patches.cuda()
        
        if self.net.show:
            test_im_A = self.net.resize_5d_tensor(im_patches, size = 378)
            test_im_B = self.net.tensorim_2_cvim(test_im_A[0].detach(), norm=1, RGB2BGR=0)
            cv2.imshow("test_im", test_im_B)

        if self.dclip_train_imgs.dim() == 5 and self.params.use_SRmask:
            with torch.no_grad():
                
                ### SRopt input feat
                if self.net.train_crop_feat:

                    dc_train_feat = self.extract_backbone_features_tiny(self.dclip_train_imgs.reshape(-1, *self.dclip_train_imgs.shape[-3:]))
                    dc_train_feat_head = self.net.get_backbone_head_feat(dc_train_feat)

                    opt_train_feat_head = dc_train_feat_head.clone()
                else:
                    opt_train_feat_head = train_feat_head.clone()
                    
                opt_test_feat_head = test_feat_head.clone()

                tensor_test_im = self.net.resize_5d_tensor(im_patches, size = 378)

                cv_prob_map = self.net.SRoptModel(tensor_test_im, [opt_train_feat_head, opt_test_feat_head], self.dclip_train_imgs)


                if self.params.refine and self.history_crop_clip_feat.shape[-1] != 1:

                    target_candidate_coords, target_candidate_scores = prutils.find_local_maxima(cv_prob_map.squeeze(), th=self.hcanTh, ks=self.hcanKs)

                    if self.pre_bbox_preds.dim() != 2:

                        pre_boxwh_list = []

                        for target_candidate_coord in target_candidate_coords:

                            score_loc = target_candidate_coord

                            bbox_raw = self.direct_bbox_regression(self.pre_bbox_preds, self.pre_sample_coords, score_loc, cv_prob_map)
                            bbox = self.clip_bbox_to_image_area(bbox_raw, image)
                            # print("bbox 1", bbox, bbox[0])

                            pre_boxwh_list.append([int(bbox[2].item()), int(bbox[3].item())])

                    ### fix hw
                    if self.usefixPrewh == 1:
                        for idx, ele in enumerate(pre_boxwh_list):
                            # print("ele", ele)
                            pre_boxwh_list[idx] = [self.pre_w, self.pre_h]

                    latest_history_crop_clip_feat = self.history_crop_clip_feat
   

                    hilight_cv_prob_map, _, _ = self.net.hilight_feat(tensor_test_im, self.dclip_train_imgs, pre_boxwh_list, cv_prob_map, \
                                        target_candidate_coords, target_candidate_scores, latest_history_crop_clip_feat, self.simvThr, self.maxCanNum)

                    prob_map_exp = hilight_cv_prob_map.expand(-1, test_feat_head.shape[1], -1, -1)
                else:
                    prob_map_exp = cv_prob_map.expand(-1, test_feat_head.shape[1], -1, -1)

                if self.hint_test_feat_head:
                    test_x = self.net.hint_test_feat_head(test_x, prob_map_exp)

        ### CLIP


        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw, bbox_preds = self.classify_target(test_x)

        ### CLIP
        if self.show:
            cv2.namedWindow("scores_raw", cv2.WINDOW_NORMAL) # search region tensor
            cv2.imshow("scores_raw", self.net.tensorim_2_cvim(torch.squeeze(scores_raw).expand(3, -1, -1), norm = 0, RGB2BGR=1))
        ###

        translation_vec, scale_ind, s, flag, score_loc = self.localize_target(scores_raw, sample_pos, sample_scales)

        bbox_raw = self.direct_bbox_regression(bbox_preds, sample_coords, score_loc, scores_raw)
        bbox = self.clip_bbox_to_image_area(bbox_raw, image)

        self.pre_bbox_preds, self.pre_sample_coords, self.pre_score_loc, self.pre_scores_raw = \
                                bbox_preds, sample_coords, score_loc, scores_raw

        if flag != 'not_found':
            self.pos = bbox[:2].flip(0) + bbox[2:].flip(0)/2  # [y + h/2, x + w/2]
            self.target_sz = bbox[2:].flip(0)
            self.target_scale = torch.sqrt(self.target_sz.prod() / self.base_target_sz.prod())
            self.target_scales.append(self.target_scale)
        else:
            if self.params.get('search_area_rescaling_at_occlusion', False):
                self.search_area_rescaling()

        # ------- UPDATE ------- #


        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False) and scores_raw.max() > self.params.get('conf_ths', 0.0):
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...] # train feat

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])
            train_y = self.get_label_function(self.pos, sample_pos[scale_ind, :], sample_scales[scale_ind]).to(
                self.params.device)

            # Update the classifier model
            self.update_memory(TensorList([train_x]), train_y, target_box, learning_rate)

        score_map = s[scale_ind, ...]

        # print("score_map.shape", score_map.shape) # 18 18
        if self.net.show:
            heat_show(score_map, "score_map")

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

        # Visualize and set debug info
        self.search_area_box = torch.cat((sample_coords[0,[1,0]], sample_coords[0,[3,2]] - sample_coords[0,[1,0]] - 1))

        if self.params.get('output_not_found_box', False):
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        ### clip
        # print("output_state", output_state) # cur coor
        if self.params.use_SRmask:
            if self.params.bg_info:
                try:
                    [x,y,w,h] = output_state
                    bb = [x-0.25*w,y-0.25*h,w*1.5,h*1.5]
                except:
                    bb = output_state
            else:
                bb = output_state

            bb = [max(0, ele) for ele in bb]

            self.pre_w, self.pre_h = bb[2], bb[3]
            
            im = im.cuda()

            tensor_crop_im, cv_crop_im = self.infer_im_2_crop_tensor(im, torch.FloatTensor(bb), in_RGB2BGR=1, im_size=352, norm = 0)

            tensor_crop_im = self.extra_norm2(tensor_crop_im) ###

            if self.previous_template_bank.dim() == 2:
                self.init_template = tensor_crop_im
                self.previous_template_bank = tensor_crop_im

                if self.clip_sim_upt:
                    self.init_template_clip_fea =  self.net.CLIP_classical_model(self.init_template.detach())

                if not self.params.refine:
                    self.dclip_train_imgs = torch.unsqueeze(torch.cat([self.init_template, self.init_template]), 0)

            else: 
                if self.skip_fir_4_frames:
                    if self.frame_num == 4:
                        self.init_template = tensor_crop_im
                        self.previous_template_bank = tensor_crop_im

                        if self.clip_sim_upt:
                            self.init_template_clip_fea =  self.net.CLIP_classical_model(self.init_template.detach())

                        self.dclip_train_imgs = torch.unsqueeze(torch.cat([self.init_template, self.init_template]), 0)

                if update_flag:
                    if self.clip_sim_upt:
                        with torch.no_grad():

                            self.upt_criteria = 0

                            # preim_newim
                            history_crop_clip_feat_im = torch.cat([torch.unsqueeze(self.net.CLIP_classical_model(dclip_train_imgs_.detach()), 0) \
                                                                        for dclip_train_imgs_ in [
                                                                            torch.unsqueeze(self.previous_template_bank[-1],0),
                                                                            tensor_crop_im
                                                                            ]], 0)                            

                            # ini_preim_newim
                            init_template_clip_fea_sqz = torch.unsqueeze(self.init_template_clip_fea, 0)
                            history_crop_clip_feat = torch.cat((init_template_clip_fea_sqz, history_crop_clip_feat_im), 0)  
                            history_crop_clip_feat_ = history_crop_clip_feat.clone()

                            history_crop_clip_feat_ = torch.squeeze(history_crop_clip_feat_, 1)

                            history_crop_clip_feat_ /= history_crop_clip_feat_.norm(dim=-1, keepdim=True)
                            similarity = (100.0 * history_crop_clip_feat_ @ history_crop_clip_feat_.T)

                            if self.history_crop_clip_feat.shape[-1] == 1:
                                self.history_crop_clip_feat = torch.cat([history_crop_clip_feat[0],history_crop_clip_feat[-1]], 0)

                            if similarity[0][-1].item() >= similarity[1][-1].item() and similarity[0][-1].item() >= self.newimSimThr:
                                self.upt_criteria = 1
                            else:
                                self.upt_criteria = 0

                            if self.clip_sim_upt == 0:
                                self.upt_criteria = 1

                            if self.upt_criteria == 1:
                                self.history_crop_clip_feat = torch.cat([history_crop_clip_feat[0],history_crop_clip_feat[-1]], 0)
                                self.previous_template_bank = torch.cat([self.init_template, tensor_crop_im])
                                self.dclip_train_imgs = self.previous_template_bank
                                self.dclip_train_imgs = torch.unsqueeze(self.dclip_train_imgs, 0)

            output_state = np.clip(output_state, 0, 9999)

        out = {'target_bbox': output_state,
               'object_presence_score': score_map.max().cpu().item()}

        if self.visdom is not None:
            self.visualize_raw_results(score_map)

        if  self.net.show:
            # cv2.waitKey(1)
            cv2.waitKey(0)

        return out

    def visualize_raw_results(self, score_map):
        self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
        self.logging_dict['max_score'].append(score_map.max())
        self.visdom.register(torch.tensor(self.logging_dict['max_score']), 'lineplot', 3, 'Max Score')
        self.debug_info['max_score'] = score_map.max().item()
        self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

    def direct_bbox_regression(self, bbox_preds, sample_coords, score_loc, scores_raw):
        shifts_x = torch.arange(
            0, self.img_sample_sz[0], step=16,
            dtype=torch.float32
        )
        shifts_y = torch.arange(
            0, self.img_sample_sz[1], step=16,
            dtype=torch.float32
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + 16 // 2
        xs, ys = locations[:, 0], locations[:, 1]
        s1, s2 = scores_raw.shape[2:]
        xs = xs.reshape(s1, s2)
        ys = ys.reshape(s1, s2)

        ltrb = bbox_preds.permute(0,1,3,4,2)[0,0].cpu() * self.train_img_sample_sz[[0, 1, 0, 1]]
        xs1 = xs - ltrb[:, :, 0]
        xs2 = xs + ltrb[:, :, 2]
        ys1 = ys - ltrb[:, :, 1]
        ys2 = ys + ltrb[:, :, 3]
        sl = score_loc.int()

        x1 = xs1[sl[0], sl[1]] / self.img_sample_sz[1] * (sample_coords[0, 3] - sample_coords[0, 1]) + sample_coords[0, 1]
        y1 = ys1[sl[0], sl[1]] / self.img_sample_sz[0] * (sample_coords[0, 2] - sample_coords[0, 0]) + sample_coords[0, 0]
        x2 = xs2[sl[0], sl[1]] / self.img_sample_sz[1] * (sample_coords[0, 3] - sample_coords[0, 1]) + sample_coords[0, 1]
        y2 = ys2[sl[0], sl[1]] / self.img_sample_sz[0] * (sample_coords[0, 2] - sample_coords[0, 0]) + sample_coords[0, 0]
        w = x2 - x1
        h = y2 - y1

        return torch.Tensor([x1, y1, w, h])

    def search_area_rescaling(self):
        if len(self.target_scales) > 0:
            min_scales, max_scales, max_history = 2, 30, 60
            self.target_not_found_counter += 1
            num_scales = max(min_scales, min(max_scales, self.target_not_found_counter))
            target_scales = torch.tensor(self.target_scales)[-max_history:]
            target_scales = target_scales[target_scales >= target_scales[-1]]  # only boxes that are bigger than the `not found`
            target_scales = target_scales[-num_scales:]  # look as many samples into past as not found endures.
            self.target_scale = torch.mean(target_scales) # average bigger boxes from the past

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            train_samples = self.training_samples[0][:self.num_stored_samples[0], ...]
            target_labels = self.target_labels[0][:self.num_stored_samples[0], ...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0], :]

            test_feat = self.net.head.extract_head_feat(sample_x)
            train_feat = self.net.head.extract_head_feat(train_samples)

            train_ltrb = self.encode_bbox(target_boxes)
            cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc = \
                self.net.head.get_filter_and_features_in_parallel(train_feat, test_feat, num_gth_frames=self.num_gth_frames,
                                                                  train_label=target_labels, train_ltrb_target=train_ltrb)
            # fuse encoder and decoder features to one feature map
            target_scores = self.net.head.classifier(cls_test_feat_enc, cls_weights)
            # compute the final prediction using the output module
            bbox_preds = self.net.head.bb_regressor(bbreg_test_feat_enc, bbreg_weights)

        return target_scores, bbox_preds

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            ### in
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None, max_disp

    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found', max_disp1
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative', max_disp2
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1

        return translation_vec1, scale_ind, scores_hn, 'normal', max_disp1

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):

        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():

            im_patches = self.extra_norm2(im_patches)
            im_patches = self.net.resize_5d_tensor(im_patches, size = 378)
            im_patches = im_patches.cuda()
            backbone_feat = OrderedDict()

            backbone_feat['layer3'] = self.net.extract_dino_features_spatial(im_patches)

        return backbone_feat, patch_coords, im_patches

    def extract_backbone_features_tiny(self, im: torch.Tensor):

        with torch.no_grad():

            im_patches = im

            im_patches = self.net.resize_5d_tensor(im_patches, size = 378)

            im_patches = im_patches.cuda()

            backbone_feat = OrderedDict()

            backbone_feat['layer3'] = self.net.extract_dino_features_spatial(im_patches)

        return backbone_feat


    def get_backbone_head_feat(self, backbone_feat):
        with torch.no_grad():
            return self.net.get_backbone_head_feat(backbone_feat)

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        # Extract initial backbone features
        with torch.no_grad():

            im_patches = self.extra_norm2(im_patches) ###

            im_patches = self.net.resize_5d_tensor(im_patches, size = 378)

            im_patches = im_patches.cuda()

            init_backbone_feat = OrderedDict()

            init_backbone_feat['layer3'] = self.net.extract_dino_features_spatial(im_patches)


        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_target_labels(self, train_x: TensorList):
        self.target_labels = TensorList([x.new_zeros(self.params.sample_memory_size, 1,
                                                     x.shape[2] + (int(self.kernel_size[0].item()) + 1) % 2,
                                                     x.shape[3] + (int(self.kernel_size[1].item()) + 1) % 2)
                                         for x in train_x])
        # Output sigma factor
        output_sigma_factor = self.params.get('output_sigma_factor', 1/4)
        self.sigma = (self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Center pos in normalized img_coords
        target_center_norm = (self.pos - self.init_sample_pos) / (self.init_sample_scale * self.img_support_sz)

        for target, x in zip(self.target_labels, train_x):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center_pos = self.feature_sz * target_center_norm + 0.5 * ksz_even
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * self.feature_sz
                target[i, 0, ...] = dcf.label_function_spatial(self.feature_sz, self.sigma, sample_center, end_pad=ksz_even)

        return self.target_labels[0][:train_x[0].shape[0]]

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        for y_memory, y, ind in zip(self.target_labels, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, pos, sample_pos, sample_scale):
        train_y = TensorList()
        target_center_norm = (pos - sample_pos) / (sample_scale * self.img_support_sz)

        for sig, sz, ksz in zip([self.sigma], [self.feature_sz], [self.kernel_size]):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center = sz * target_center_norm + 0.5*ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))

        return train_y

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_backbone_head_feat(init_backbone_feat)
        # x = init_backbone_feat

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = getattr(self.net.head.filter_predictor, 'filter_size', 1)
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Get target labels for the different augmentations
        self.init_target_labels(TensorList([x]))

        self.num_gth_frames = target_boxes.shape[0]

        if hasattr(self.net.head.filter_predictor, 'num_gth_frames'):
            self.net.head.filter_predictor.num_gth_frames = self.num_gth_frames

        self.init_memory(TensorList([x]))

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
                self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')


    ### CLIP
    def infer_im_2_crop_tensor(self, tmp_imm_0, gt, in_RGB2BGR=0, im_size=288, norm = 1):

        x, y, w, h = gt.int()
        tensor_crop = fn.crop(tmp_imm_0, y, x, h, w)

        tensor_crop_resize_ = F.interpolate(tensor_crop, size=(im_size, im_size), mode='bilinear')

        cv_tensor_crop_resize_ = torch.squeeze(tensor_crop_resize_, 0)

        cv_tensor_crop_resize_ = self.net.tensorim_2_cvim(cv_tensor_crop_resize_, norm = norm, RGB2BGR=in_RGB2BGR)

        return tensor_crop_resize_, cv_tensor_crop_resize_

    def extra_norm(self, im_patches):
        
        im_patches = im_patches/255

        return im_patches

    def extra_norm2(self, im_patches):
        
        im_patches = im_patches/255

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        _mean = torch.Tensor(mean).view(-1, 1, 1).to(im_patches.device)
        _std = torch.Tensor(std).view(-1, 1, 1).to(im_patches.device)

        im_patches -= _mean
        im_patches /= _std
        
        return im_patches

    def extra_norm3(self, im_patches):
        
        im_patches = im_patches/255

        im_patches = im_patches[:, [2, 1, 0], :, :]

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        _mean = torch.Tensor(mean).view(-1, 1, 1).to(im_patches.device)
        _std = torch.Tensor(std).view(-1, 1, 1).to(im_patches.device)

        im_patches -= _mean.cuda()
        im_patches /= _std.cuda()

        return im_patches


    ### CLIP