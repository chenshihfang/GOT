# This is the GOT-Edit inference protocol that is compatible with VGGT/DA3.
# This code was created by a human developer, 
# refined with AI assistance (ChatGPT) for clarity and cleanup before release, 
# and validated through human evaluation.

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
# from heatmap import heat_show
import numpy as np
from collections import OrderedDict, deque, defaultdict
import sys
import os
cpu_num = 8
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
import ltr.data.processing_utils as prutils
import cv2
import torchvision.transforms.functional as fn
import torch.backends.cuda
import torch.nn as nn

# DinoPatch = 18
DinoPatch = 27

DinoStride = 14

import multiprocessing as mp
use_deepspeed = False
if not mp.current_process().daemon:
    try:
        import deepspeed
        use_deepspeed = True
        print(f'[Deepspeed imported]')
    except:
        print(f'[Deepspeed disabled]')
        use_deepspeed = False
print('use_deepspeed', use_deepspeed)
from contextlib import contextmanager
use_tf32 = True
print('use_tf32', use_tf32)
use_compile = True
if use_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
use_set_float32_matmul_precision = False

@contextmanager
def measure_time(name='block'):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    print(f'{name} took {(end - start) * 1000:.3f} ms')

class ToMP(BaseTracker):
    multiobj_mode = 'parallel'

    def __init__(self, params):
        super(ToMP, self).__init__(params)
        self.second_ref_history = None
        self.confidence_history = None
        self._dino_stream = None
        self._vggt_stream = None

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def _pick_patch_max_scale_change_required_shrink(self, im: torch.Tensor, scales, sz: torch.Tensor):
        """
        Use this function to enable auto-tuning for search-region rescaling, 
        but it is still acceptable to use it without that for similar performance.
        Choose patch_max_scale_change from {2.0, 2.5} using a required-shrink gate.
        """
        mode = self.params.get('border_mode', 'replicate')
        if mode not in ['inside', 'inside_major']:
            return self.params.get('patch_max_scale_change', None)
        low = float(self.params.get('patch_max_scale_change_low', 2.0))
        high = float(self.params.get('patch_max_scale_change_high', 2.5))
        thr = float(self.params.get('patch_max_scale_change_gate_thr', 2.0))
        im_sz = torch.tensor([im.shape[2], im.shape[3]], device=im.device, dtype=torch.float32)
        if isinstance(scales, (int, float)):
            s_max = float(scales)
        else:
            s_max = float(torch.as_tensor(scales).max().item())
        sample_sz = sz.to(im.device).float() * s_max
        shrink_vec = (sample_sz / im_sz).clamp(min=1.0)
        required_shrink = float(shrink_vec.max().item()) if mode == 'inside' else float(shrink_vec.min().item())
        return high if required_shrink > thr else low

    def initialize(self, image, info: dict) -> dict:
        print('DinoPatch infer', DinoPatch)
        if use_set_float32_matmul_precision:
            torch.set_float32_matmul_precision('high')
        print('use_set_float32_matmul_precision', use_set_float32_matmul_precision)
        self.par_dino_vggt = True
        print('self.par_dino_vggt ', self.par_dino_vggt)
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'
        self.initialize_features()
        if self.params.device.startswith('cuda') and torch.cuda.is_available():
            if self._dino_stream is None:
                self._dino_stream = torch.cuda.Stream()
            if self._vggt_stream is None:
                self._vggt_stream = torch.cuda.Stream()
        if not use_compile:
            self.net = self.params.net
        else:
            self.net_wrapper = self.params.net
            base = self.params.net.net
            base.eval()
            if use_deepspeed:
                ds_engine = deepspeed.init_inference(model=base, dtype=torch.bfloat16, mp_size=1, replace_with_kernel_inject=False)
                self.net = ds_engine
                self.net.eval()
                print(f'[Deepspeed applied]')
            else:
                self.net = torch.compile(base, mode='reduce-overhead', fullgraph=False)
                print(f'[torch.compile applied]')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tic = time.perf_counter()
        self.show = 0
        try:
            print("info['dataset']", info['dataset'])
            self.object_class = info['object_class']
            self.seq_name = info['name']
            print('self.object_class', self.object_class)
            print('self.seq_name', self.seq_name)
        except:
            pass
        im = numpy_to_torch(image)
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
        self.target_sz = torch.Tensor([state[3], state[2]])
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz
        tfs = self.params.get('train_feature_size', DinoPatch)
        stride = self.params.get('feature_stride', 16)
        self.train_img_sample_sz = torch.Tensor([tfs * stride, tfs * stride])
        self.im_patches_mem = torch.zeros(self.params.sample_memory_size + 1, 1, 3, DinoPatch * stride, DinoPatch * stride, device='cuda')
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale = math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()
        self.base_target_sz = self.target_sz / self.target_scale
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)
        init_backbone_feat = self.generate_init_samples(im)
        self.init_classifier(init_backbone_feat)
        self.logging_dict = defaultdict(list)
        self.target_scales = []
        self.target_not_found_counter = 0
        self.cls_weights_avg = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out = {'time': time.perf_counter() - tic}
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
        shifts_x = torch.arange(0, output_sz, step=stride, dtype=torch.float32, device=bbox.device)
        shifts_y = torch.arange(0, output_sz, step=stride, dtype=torch.float32, device=bbox.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        xs, ys = (locations[:, 0], locations[:, 1])
        xyxy = torch.stack([bbox[:, 0], bbox[:, 1], bbox[:, 0] + bbox[:, 2], bbox[:, 1] + bbox[:, 3]], dim=1)
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

    def track(self, image, info: dict=None) -> dict:
        self.debug_info = {}
        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num
        if self.frame_num % 50 == 0:
            print(self.frame_num)
        im = numpy_to_torch(image)
        if not self.par_dino_vggt:
            backbone_feat, sample_coords, im_patches, im_patches_wores, vggt_dpt_feats_ = self.extract_backbone_features(im, self.get_centered_sample_pos(), self.target_scale * self.params.scale_factors, self.img_sample_sz)
        else:
            backbone_feat, sample_coords, im_patches, im_patches_wores, vggt_dpt_feats_ = self.extract_backbone_features_par(im, self.get_centered_sample_pos(), self.target_scale * self.params.scale_factors, self.img_sample_sz)
        test_x = self.get_backbone_head_feat(backbone_feat)
        sample_pos, sample_scales = self.get_sample_location(sample_coords)
        scores_raw, bbox_preds = self.classify_target(test_x, vggt_dpt_feats_)
        if self.net.show:
            cv2.namedWindow('scores_raw', cv2.WINDOW_NORMAL)
            cv2.imshow('scores_raw', self.net.tensorim_2_cvim(torch.squeeze(scores_raw).expand(3, -1, -1), norm=0, RGB2BGR=1))
            heat_show(scores_raw[0][0], 'heat_show_scores_raw')
        translation_vec, scale_ind, s, flag, score_loc = self.localize_target(scores_raw, sample_pos, sample_scales)
        bbox_raw = self.direct_bbox_regression(bbox_preds, sample_coords, score_loc, scores_raw)
        bbox = self.clip_bbox_to_image_area(bbox_raw, image)
        if flag != 'not_found':
            self.pos = bbox[:2].flip(0) + bbox[2:].flip(0) / 2
            self.target_sz = bbox[2:].flip(0)
            self.target_scale = torch.sqrt(self.target_sz.prod() / self.base_target_sz.prod())
            self.target_scales.append(self.target_scale)
        elif self.params.get('search_area_rescaling_at_occlusion', False):
            self.search_area_rescaling()
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = flag == 'hard_negative'
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None
        if update_flag and self.params.get('update_classifier', False) and (scores_raw.max() > self.params.get('conf_ths', 0.0)):
            train_x_feat_current = test_x[scale_ind:scale_ind + 1, ...]
            target_box_current = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind, :], sample_scales[scale_ind])
            train_y_label_current = self.get_label_function(self.pos, sample_pos[scale_ind, :], sample_scales[scale_ind]).to(self.params.device)
            learning_rate_update = self.params.get('hard_negative_learning_rate', None) if hard_negative else None
            self.update_memory(TensorList([train_x_feat_current]), train_y_label_current, target_box_current, learning_rate_update)
            current_patch_selected_scale = im_patches
            self.im_patches_mem[1], self.im_patches_mem[2] = (im_patches, im_patches)
        score_map = s[scale_ind, ...]
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))
        self.search_area_box = torch.cat((sample_coords[0, [1, 0]], sample_coords[0, [3, 2]] - sample_coords[0, [1, 0]] - 1))
        if self.params.get('output_not_found_box', False):
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()
        out = {'target_bbox': output_state, 'object_presence_score': score_map.max().cpu().item()}
        if self.visdom is not None:
            self.visualize_raw_results(score_map)
        if self.net.show:
            Im_w_box = self.draw_im_rect(im, output_state)
            cv2.namedWindow('Im_w_box', cv2.WINDOW_NORMAL)
            cv2.imshow('Im_w_box', Im_w_box)
            cv2.waitKey(0)
        return out

    def visualize_raw_results(self, score_map):
        self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
        self.logging_dict['max_score'].append(score_map.max())
        self.visdom.register(torch.tensor(self.logging_dict['max_score']), 'lineplot', 3, 'Max Score')
        self.debug_info['max_score'] = score_map.max().item()
        self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

    def direct_bbox_regression(self, bbox_preds, sample_coords, score_loc, scores_raw):
        shifts_x = torch.arange(0, self.img_sample_sz[0], step=self.params.get('feature_stride', 16), dtype=torch.float32)
        shifts_y = torch.arange(0, self.img_sample_sz[1], step=self.params.get('feature_stride', 16), dtype=torch.float32)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + self.params.get('feature_stride', 16) // 2
        xs, ys = (locations[:, 0], locations[:, 1])
        s1, s2 = scores_raw.shape[2:]
        xs = xs.reshape(s1, s2)
        ys = ys.reshape(s1, s2)
        ltrb = bbox_preds.permute(0, 1, 3, 4, 2)[0, 0].cpu() * self.train_img_sample_sz[[0, 1, 0, 1]]
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
            min_scales, max_scales, max_history = (2, 30, 60)
            self.target_not_found_counter += 1
            num_scales = max(min_scales, min(max_scales, self.target_not_found_counter))
            target_scales = torch.tensor(self.target_scales)[-max_history:]
            target_scales = target_scales[target_scales >= target_scales[-1]]
            target_scales = target_scales[-num_scales:]
            self.target_scale = torch.mean(target_scales)

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5 * (sample_coord[:, :2] + sample_coord[:, 2:] - 1)
        sample_scales = ((sample_coord[:, 2:] - sample_coord[:, :2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return (sample_pos, sample_scales)

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + (self.feature_sz + self.kernel_size) % 2 * self.target_scale * self.img_support_sz / (2 * self.feature_sz)

    def classify_target(self, sample_x: TensorList, vggt_dpt_feats_from_track):
        """Classify target by applying the DiMP filter.
        sample_x: Backbone DiNO features of the current test frame (num_scales, C_dino_raw, H_feat_raw, W_feat_raw).
                  These are expected to be the output of self.net.extract_dino_features_spatial_intermediate_layers.
        vggt_dpt_feats_from_track: Combined VGGT DPT features for reference and current frames.
                                     Expected shape (S_total, C_feat_vggt, H_feat, W_feat),
                                     where S_total includes (fixed_ref, dynamic_ref, current_frame_patch).
        """
        with torch.no_grad():
            train_samples_raw_dino = self.training_samples[0][:self.num_stored_samples[0], ...]
            target_labels = self.target_labels[0][:self.num_stored_samples[0], ...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0], :]
            train_ltrb = self.encode_bbox(target_boxes)
            test_feat_head_2d = self.net.head.extract_head_feat(sample_x)
            train_feat_head_2d = self.net.head.extract_head_feat(train_samples_raw_dino)
            if vggt_dpt_feats_from_track is not None:
                num_train_refs_for_vggt = self.num_stored_samples[0]
                train_vggt_dpt_feats_raw = vggt_dpt_feats_from_track[:self.params.sample_memory_size]
                test_vggt_dpt_feats_raw = vggt_dpt_feats_from_track[self.params.sample_memory_size:]
                train_vggt_dpt_feats_mlp_in = train_vggt_dpt_feats_raw.reshape(-1, *train_vggt_dpt_feats_raw.shape[-3:])
                test_vggt_dpt_feats_mlp_in = test_vggt_dpt_feats_raw.reshape(-1, *test_vggt_dpt_feats_raw.shape[-3:])
                if self.net.use_geo_type == 'VGGT':
                    test_vggt_dpt_feats_head_ = self.net.vggtDPTfeatMlp_head(test_vggt_dpt_feats_mlp_in.contiguous())
                    train_vggt_dpt_feats_head_ = self.net.vggtDPTfeatMlp_head(train_vggt_dpt_feats_mlp_in.contiguous())
                else:
                    test_vggt_dpt_feats_head_ = test_vggt_dpt_feats_mlp_in.contiguous()
                    train_vggt_dpt_feats_head_ = train_vggt_dpt_feats_mlp_in.contiguous()
                train_feat_for_gate = train_feat_head_2d.unsqueeze(1)
                train_vggt_for_gate = train_vggt_dpt_feats_head_.unsqueeze(1)
                test_feat_for_gate = test_feat_head_2d.unsqueeze(1)
                test_vggt_for_gate = test_vggt_dpt_feats_head_.unsqueeze(1)
                if train_feat_for_gate.shape[0] != self.params.sample_memory_size:
                    train_feat_for_gate = train_feat_for_gate.expand(self.params.sample_memory_size, -1, -1, -1, -1)
                train_feat_vggt, test_feat_vggt = self.net.head._safe_fp32_call(lambda a, b, c, d, u3, si: self.net.DiNO_VGGT_Gate(a, b, c, d, u3, si), 'DiNO_VGGT_Gate', train_feat_for_gate, train_vggt_for_gate.to(train_feat_for_gate.device), test_feat_for_gate, test_vggt_for_gate.to(test_feat_for_gate.device), True, False)
                final_train_feat_vggt = train_feat_vggt.squeeze(1)
                final_test_feat_vggt = test_feat_vggt.squeeze(1)
            if self.num_stored_samples[0] > 1:
                cls_filter, bbreg_filter, cls_test_feat_enc, bbreg_test_feat_enc = self.net.head.get_filter_and_features_in_parallel(train_feat_head_2d, test_feat_head_2d, num_gth_frames=self.num_gth_frames, train_label=target_labels, train_ltrb_target=train_ltrb)
                cls_filter_vggt, breg_filter_vggt, cls_test_feat_enc_vggt, bbreg_test_feat_enc_vggt = self.net.head.get_filter_and_features_in_parallel(final_train_feat_vggt, final_test_feat_vggt, num_gth_frames=self.num_gth_frames, train_label=target_labels, train_ltrb_target=train_ltrb)
            else:
                cls_filter, bbreg_filter, cls_test_feat_enc, bbreg_test_feat_enc = self.net.head.get_filter_and_features_in_parallel(train_feat_head_2d[0].unsqueeze(0), test_feat_head_2d, num_gth_frames=self.num_gth_frames, train_label=target_labels[0].unsqueeze(0), train_ltrb_target=train_ltrb[0].unsqueeze(0))
                cls_filter_vggt, breg_filter_vggt, cls_test_feat_enc_vggt, bbreg_test_feat_enc_vggt = self.net.head.get_filter_and_features_in_parallel(final_train_feat_vggt[0].unsqueeze(0), final_test_feat_vggt, num_gth_frames=self.num_gth_frames, train_label=target_labels[0].unsqueeze(0), train_ltrb_target=train_ltrb[0].unsqueeze(0))
            if self.params.get('wpcls', True):
                cls_filter = self.net.head._safe_fp32_call(self.net.JEPA_predictor_cls, 'JEPA_predictor_cls', cls_filter)
            if self.params.get('wpbreg', True):
                bbreg_filter = self.net.head._safe_fp32_call(self.net.JEPA_predictor_breg, 'JEPA_predictor_breg', bbreg_filter)
            if not self.net.head.splitCfilter:
                cls_filter_vggt = self.net.head._safe_fp32_call(self.net.JEPA_predictor_cls, 'JEPA_predictor_cls(vggt)', cls_filter_vggt)
            cls_filter_vggt = self.net.head._safe_fp32_call(self.net.WvggtLinearCls, 'WvggtLinearCls', cls_filter_vggt)
            breg_filter_vggt = self.net.head._safe_fp32_call(self.net.JEPA_predictor_breg, 'JEPA_predictor_breg(vggt)', breg_filter_vggt)
            refiner = self.net.head.AlphaEditRefiner_mix()
            with torch.cuda.amp.autocast(enabled=False):
                if not torch.isfinite(cls_test_feat_enc).all():
                    refined_cls_filter = cls_filter.float()
                else:
                    refined_cls_filter = refiner.refine_weights_with_alphaedit(k0=cls_test_feat_enc.float().contiguous(), w=cls_filter.float(), w_vggt_delta=cls_filter_vggt.float(), hparams=self.net.head.AlphaEditHyperparams)
            refined_cls_filter = self.net.head._sanitize(refined_cls_filter, 'refined_cls_filter', verbose=False)
            cls_test_feat_enc_vggt = self.net.head._sanitize(cls_test_feat_enc_vggt, 'test_feat_enc_vggt', verbose=False)
            bbreg_test_feat_enc_vggt = self.net.head._sanitize(bbreg_test_feat_enc_vggt, 'bbreg_test_feat_enc_vggt', verbose=False)
            target_scores = self.net.head.classifier(cls_test_feat_enc_vggt, refined_cls_filter)
            bbox_preds = self.net.head.bb_regressor(bbreg_test_feat_enc_vggt, breg_filter_vggt)
            target_scores = self.net.head._sanitize(target_scores, 'target_scores(out)', verbose=False)
            bbox_preds = self.net.head._sanitize(bbox_preds, 'bbox_preds(out)', verbose=False)
            return (target_scores, bbox_preds)

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
            kernel = scores.new_ones(1, 1, score_filter_ksz, score_filter_ksz)
            scores = F.conv2d(scores.view(-1, 1, *scores.shape[-2:]), kernel, padding=score_filter_ksz // 2).view(scores.shape)
        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1) / 2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind, ...].float().cpu().view(-1)
        target_disp = max_disp - score_center
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]
        return (translation_vec, scale_ind, scores, None, max_disp)

    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""
        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1) / 2
        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window
        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale
        if max_score1.item() < self.params.target_not_found_threshold:
            return (translation_vec1, scale_ind, scores_hn, 'not_found', max_disp1)
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return (translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1)
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return (translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1)
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)
        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right] = 0
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale
        prev_target_vec = (self.pos - sample_pos[scale_ind, :]) / (self.img_support_sz / output_sz * sample_scale)
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1 - prev_target_vec) ** 2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2 - prev_target_vec) ** 2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2
            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return (translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1)
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return (translation_vec2, scale_ind, scores_hn, 'hard_negative', max_disp2)
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return (translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1)
            return (translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1)
        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return (translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1)
        return (translation_vec1, scale_ind, scores_hn, 'normal', max_disp1)

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        dyn_msc = self._pick_patch_max_scale_change_required_shrink(im, scales, sz)
        im_patches_current_raw, patch_coords = sample_patch_multiscale(im, pos, scales, sz, mode=self.params.get('border_mode', 'replicate'), max_scale_change=dyn_msc)
        with torch.no_grad():
            im_patches_current_normalized = self.extra_norm2(im_patches_current_raw)
            im_patches_current_resized = self.net.resize_5d_tensor(im_patches_current_normalized, size=DinoPatch * DinoStride)
            im_patches_current_resized = im_patches_current_resized.cuda()
            backbone_feat_current = OrderedDict()
            with torch.inference_mode():
                backbone_feat_current['layer3'] = self.net.extract_dino_features_spatial_intermediate_layers(im_patches_current_resized)
            vggt_dpt_feats_output = None
            current_patch_single_scale_for_vggt = im_patches_current_resized[0:1]
            current_patch_single_scale_for_vggt_b = current_patch_single_scale_for_vggt.unsqueeze(1)
            im_patches_for_vggt_cat_seqdim = torch.cat((self.im_patches_mem[0:self.params.sample_memory_size], current_patch_single_scale_for_vggt_b), dim=0)
            im_patches_in_for_vggt_batchdim = im_patches_for_vggt_cat_seqdim.permute(1, 0, 2, 3, 4)
            with torch.inference_mode():
                if self.net.use_geo_type == 'VGGT':
                    vggt_dpt_feats_raw = self.net.extract_dino_features_spatial_intermediate_layers_vggt_cat(im_patches_in_for_vggt_batchdim)
                else:
                    vggt_dpt_feats_raw = self.net.extract_da3_dpt_features_intermediate_cat(im_patches_in_for_vggt_batchdim)
            vggt_dpt_feats_output = vggt_dpt_feats_raw.squeeze(1)
            return (backbone_feat_current, patch_coords, im_patches_current_resized, im_patches_current_raw.cuda(), vggt_dpt_feats_output)

    def extract_backbone_features_par(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        dyn_msc = self._pick_patch_max_scale_change_required_shrink(im, scales, sz)
        im_patches_current_raw, patch_coords = sample_patch_multiscale(
            im,
            pos,
            scales,
            sz,
            mode=self.params.get('border_mode', 'replicate'),
            max_scale_change=dyn_msc,
        )

        with torch.no_grad():
            im_patches_current_normalized = self.extra_norm2(im_patches_current_raw)
            im_patches_current_resized = self.net.resize_5d_tensor(
                im_patches_current_normalized,
                size=DinoPatch * DinoStride,
            )
            im_patches_current_resized = im_patches_current_resized.cuda(non_blocking=True)

            backbone_feat_current = OrderedDict()
            current_patch_single_scale_for_vggt = im_patches_current_resized[0:1]
            current_patch_single_scale_for_vggt_b = current_patch_single_scale_for_vggt.unsqueeze(1)
            im_patches_for_vggt_cat_seqdim = torch.cat(
                (self.im_patches_mem[0:self.params.sample_memory_size], current_patch_single_scale_for_vggt_b),
                dim=0,
            )
            im_patches_in_for_vggt_batchdim = im_patches_for_vggt_cat_seqdim.permute(1, 0, 2, 3, 4)

            current_stream = torch.cuda.current_stream()
            dino_stream = self._dino_stream if self._dino_stream is not None else current_stream
            vggt_stream = self._vggt_stream if self._vggt_stream is not None else current_stream

            dino_stream.wait_stream(current_stream)
            vggt_stream.wait_stream(current_stream)

            with torch.cuda.stream(dino_stream):
                with torch.inference_mode():
                    backbone_feat_current['layer3'] = self.net.extract_dino_features_spatial_intermediate_layers(
                        im_patches_current_resized
                    )

            with torch.cuda.stream(vggt_stream):
                with torch.inference_mode():
                    if self.net.use_geo_type == 'VGGT':
                        vggt_dpt_feats_raw = self.net.extract_dino_features_spatial_intermediate_layers_vggt_cat(
                            im_patches_in_for_vggt_batchdim
                        )
                    else:
                        vggt_dpt_feats_raw = self.net.extract_da3_dpt_features_intermediate_cat(
                            im_patches_in_for_vggt_batchdim
                        )
                    vggt_dpt_feats_output = vggt_dpt_feats_raw.squeeze(1)

            current_stream.wait_stream(dino_stream)
            current_stream.wait_stream(vggt_stream)

            backbone_feat_current['layer3'].record_stream(current_stream)
            vggt_dpt_feats_output.record_stream(current_stream)
            im_patches_current_resized.record_stream(current_stream)

            return (
                backbone_feat_current,
                patch_coords,
                im_patches_current_resized,
                im_patches_current_raw,
                vggt_dpt_feats_output,
            )

    def get_backbone_head_feat(self, backbone_feat):
        with torch.no_grad():
            return self.net.get_backbone_head_feat(backbone_feat)

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples. (path during init)"""
        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = sample_sz.float() / im_sz
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = sample_sz.float() / shrink_factor
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = -((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)
        self.init_sample_pos = self.pos.round()
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]
        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz / 2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)
        with torch.no_grad():
            im_patches_normalized = self.extra_norm2(im_patches)
            im_patches_resized = self.net.resize_5d_tensor(im_patches_normalized, size=DinoPatch * DinoStride)
            im_patches_resized = im_patches_resized.cuda()
            if not hasattr(self, 'im_patches_mem') or self.im_patches_mem is None:
                self.im_patches_mem = torch.zeros(self.params.sample_memory_size + 1, 1, 3, DinoPatch * DinoStride, DinoPatch * DinoStride, device='cuda')
            initial_patch_for_memory = im_patches_resized[0:1].unsqueeze(1)
            self.im_patches_mem = initial_patch_for_memory.repeat(self.im_patches_mem.shape[0], 1, 1, 1, 1)
            init_dino_feats = self.net.extract_dino_features_spatial_intermediate_layers(im_patches_resized)
            init_backbone_feat = OrderedDict()
            init_backbone_feat['layer3'] = init_dino_feats
            return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0], :] = init_target_boxes
        return init_target_boxes

    def init_target_labels(self, train_x: TensorList):
        self.target_labels = TensorList([x.new_zeros(self.params.sample_memory_size, 1, x.shape[2] + (int(self.kernel_size[0].item()) + 1) % 2, x.shape[3] + (int(self.kernel_size[1].item()) + 1) % 2) for x in train_x])
        output_sigma_factor = self.params.get('output_sigma_factor', 1 / 4)
        self.sigma = (self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)
        target_center_norm = (self.pos - self.init_sample_pos) / (self.init_sample_scale * self.img_support_sz)
        for target, x in zip(self.target_labels, train_x):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center_pos = self.feature_sz * target_center_norm + 0.5 * ksz_even
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * self.feature_sz
                target[i, 0, ...] = dcf.label_function_spatial(self.feature_sz, self.sigma, sample_center, end_pad=ksz_even)
        return self.target_labels[0][:train_x[0].shape[0]]

    def init_memory(self, train_x: TensorList):
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw
        self.training_samples = TensorList([x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])
        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0], ...] = x

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, target_box, learning_rate=None):
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x
        for y_memory, y, ind in zip(self.target_labels, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y
        self.target_boxes[replace_ind[0], :] = target_box
        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate=None):
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
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind
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
            center = sz * target_center_norm + 0.5 * ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))
        return train_y

    def update_state(self, new_pos, new_scale=None):
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale
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
        x = self.get_backbone_head_feat(init_backbone_feat)
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1] * num)
            x = torch.cat([x, F.dropout2d(x[0:1, ...].expand(num, -1, -1, -1), p=prob, training=True)])
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = getattr(self.net.head.filter_predictor, 'filter_size', 1)
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1) % 2
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz * self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)
        target_boxes = self.init_target_boxes()
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

    def infer_im_2_crop_tensor(self, tmp_imm_0, gt, in_RGB2BGR=0, im_size=DinoPatch * DinoStride, norm=1):
        x, y, w, h = gt.int()
        tensor_crop = fn.crop(tmp_imm_0, y, x, h, w)
        tensor_crop_resize_ = F.interpolate(tensor_crop, size=(im_size, im_size), mode='bilinear')
        cv_tensor_crop_resize_ = torch.squeeze(tensor_crop_resize_, 0)
        cv_tensor_crop_resize_ = self.net.tensorim_2_cvim(cv_tensor_crop_resize_, norm=norm, RGB2BGR=in_RGB2BGR)
        return (tensor_crop_resize_, cv_tensor_crop_resize_)

    def extra_norm(self, im_patches):
        im_patches = im_patches / 255
        return im_patches

    def extra_norm2(self, im_patches):
        im_patches = im_patches / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        _mean = torch.Tensor(mean).view(-1, 1, 1).to(im_patches.device)
        _std = torch.Tensor(std).view(-1, 1, 1).to(im_patches.device)
        im_patches -= _mean
        im_patches /= _std
        return im_patches

    def extra_norm3(self, im_patches):
        im_patches = im_patches / 255
        im_patches = im_patches[:, [2, 1, 0], :, :]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        _mean = torch.Tensor(mean).view(-1, 1, 1).to(im_patches.device)
        _std = torch.Tensor(std).view(-1, 1, 1).to(im_patches.device)
        im_patches -= _mean.cuda()
        im_patches /= _std.cuda()
        return im_patches

    def draw_im_rect(self, im, output_state):
        im_np = im.squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
        im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
        x, y, w, h = map(int, output_state)
        cv2.rectangle(im_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return im_np
