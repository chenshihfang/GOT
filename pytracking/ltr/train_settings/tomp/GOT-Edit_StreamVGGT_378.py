# -*- coding: utf-8 -*-
import os
import torch
import torch.optim as optim
import torch.distributed as dist

from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq, VastTrack
from ltr.data import processing, sampler
from ltr.data import LTRLoader

from ltr.models.tracking import tompnet_JEPAp_vggt
import ltr.models.loss as ltr_losses
import ltr.actors.tracking as actors
from ltr.trainers.ltr_trainer_dsA_magav1 import LTRTrainer_magav1  # DS/Native-DDP aware trainer

from ltr.models.loss.bbr_loss import GIoULoss
import ltr.admin.loading as network_loading

from ltr.models.backbone.peft_dora_vggt import inject_dora_into_backbone, collect_adapter_params

import deepspeed
print("DS version:", getattr(deepspeed, "__version__", "unknown"))
print("Torch:", torch.__version__)

# === DoRA injection: *global-attention only* (Linear layers inside global attn) ===
from peft import LoraConfig, TaskType, get_peft_model
import torch.nn as nn
import re

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python ltr/run_training_dsA.py tomp GOT-Edit_StreamVGGT_378

# ===== Backend selector =====
#   LTR_BACKEND=ddp  -> native PyTorch DDP
#   LTR_BACKEND=ds0  -> DeepSpeed ZeRO-0 (DS's DDP)

# CUDA_VISIBLE_DEVICES=0,1 python ltr/run_training_dsA.py tomp test

# BACKEND = os.environ.get("LTR_BACKEND", "ddp").lower().strip()  # "ddp" or "ds0"
BACKEND = "ds0"

# ===== DS knobs (when BACKEND=ds0) =====
DEEPSPEED_ZERO_STAGE = 0
DEEPSPEED_FP16 = False
DEEPSPEED_BF16 = False

# Global AC toggle
USE_ACTIVATION_CHECKPOINTING = True

# === Simple selector for checkpointing implementation ===
#   "torch"     → torch.utils.checkpoint (deterministic RNG)
#   "deepspeed" → DeepSpeed activation checkpointing (best-effort configure)
CKPT_IMPL = "torch"   # change to "torch" to use Torch AC

# ===== Gradient clipping ( boolean) =====
# True  → use DeepSpeed's internal gradient clipping in engine.step()
# False → use PyTorch clip_grad_norm_ once in the trainer before step()
# USE_DS_GRAD_CLIP = True
USE_DS_GRAD_CLIP = False
GRAD_CLIP_NORM = 0.8  # set <= 0 to disable clipping entirely

# ===== Training knobs =====

GPU_num = 8
PER_GPU_BATCH_SIZE = 8

NUM_WORKERS = 4

EFFECTIVE_BATCH_SIZE = GPU_num * PER_GPU_BATCH_SIZE

NUM_EPOCHS = 25
TRAIN_SAMPLES_PER_EPOCH = 200000   # global target; sharded per rank
VAL_SAMPLES_PER_EPOCH   = 10000
VAL_EPOCH_INTERVAL = 1

TARGET_FILTER_SZ = 1
FEATURE_SZ = 27
SEARCH_AREA_FACTOR = 5.0
OUTPUT_SIGMA_FACTOR = 1/4
CENTER_JITTER_FACTOR = {'train': 0., 'test': 4.5}
SCALE_JITTER_FACTOR  = {'train': 0., 'test': 0.5}
CROP_TYPE = 'inside_major'
MAX_SCALE_CHANGE = 1.5
MAX_GAP  = 200
NUM_TRAIN_FRAMES = 2
NUM_TEST_FRAMES  = 1
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
USE_TEST_FRAME_ENCODING = True
FROZEN_BACKBONE_LAYERS = ['conv1', 'bn1', 'layer1', 'layer2']
FREEZE_BACKBONE_BN_LAYERS = True

WEIGHT_GIOU = 1.0
WEIGHT_CLF  = 100.0
HINGE_THRESHOLD = 0.05
NORMALIZED_BBREG_COORDS = True
CENTER_SAMPLING_RADIUS  = 1.0

BASE_LRS = {
    "JEPA_predictor_cls": 1e-4,
    "JEPA_predictor_breg": 1e-4,
    "WvggtLinearCls": 1e-4,
    "vggtDPTfeatMlp": 1e-4,
    "vggtDPTfeatMlp_head": 1e-4,
    "DiNO_VGGT_Gate": 1e-4,
    "bkMlp": 1e-4,
    "head": 1e-4,
}

WEIGHT_DECAY = 1e-4
LR_MILESTONES = [10, 15, 20, 25]

LR_GAMMA = 0.2

# DP-trained checkpoint (same path used in DP config)
PRETRAIN_PATH = "/data1/sfchen94/pytrackingcsf/pytracking/checkpoints/ltr/tomp/path_to_tompL_378/pretrain_model.pth.tar" 

# ===== utils =====
def _dist_ready():
    return dist.is_available() and dist.is_initialized()

def _rank():
    if _dist_ready():
        return dist.get_rank()
    return 0

def print0(*a, **k):
    if _rank() == 0:
        print(*a, **k)

def _world_size():
    if _dist_ready():
        return dist.get_world_size()
    env_ws = os.environ.get("WORLD_SIZE")
    if env_ws:
        try:
            return max(1, int(env_ws))
        except Exception:
            pass
    devs = torch.cuda.device_count() or 1
    return max(1, devs)

def get_root(m):
    return m.module if hasattr(m, "module") else m


# ===== main =====
def run(settings):
    settings.description = 'ToMP50 (Native DDP or DS-ZeRO0) with DP-style init + parity fixes'

    # --- backend: choose native DDP or DS-ZeRO0 ---
    settings.use_deepspeed = (BACKEND == "ds0")
    settings.deepspeed_zero_stage = int(DEEPSPEED_ZERO_STAGE) if settings.use_deepspeed else 0
    settings.deepspeed_fp16 = DEEPSPEED_FP16
    settings.deepspeed_bf16 = DEEPSPEED_BF16
    settings.USE_ACTIVATION_CHECKPOINTING = USE_ACTIVATION_CHECKPOINTING
    settings.ckpt_impl = CKPT_IMPL  # <- pass down simple selector

    # expose clipping choice to trainer/base
    settings.gradient_clipping = float(GRAD_CLIP_NORM)
    settings.use_deepspeed_grad_clip = bool(USE_DS_GRAD_CLIP) and (settings.gradient_clipping > 0.0)
    settings.use_torch_grad_clip = (not settings.use_deepspeed_grad_clip) and (settings.gradient_clipping > 0.0)

    # Validation: rank-0 only (DP parity)
    settings.val_mode = "rank0"
    # settings.val_mode = "distributed"
    settings.print_interval = 1

    # --- core ---
    settings.batch_size = PER_GPU_BATCH_SIZE
    settings.num_workers = NUM_WORKERS
    settings.multi_gpu = True  # Trainer handles DDP/DS wrapping

    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std  = [0.229, 0.224, 0.225]

    settings.search_area_factor = SEARCH_AREA_FACTOR
    settings.output_sigma_factor = OUTPUT_SIGMA_FACTOR
    settings.target_filter_sz    = TARGET_FILTER_SZ
    settings.feature_sz = FEATURE_SZ
    settings.output_sz  = settings.feature_sz * 16
    settings.center_jitter_factor = CENTER_JITTER_FACTOR
    settings.scale_jitter_factor  = SCALE_JITTER_FACTOR
    settings.hinge_threshold = HINGE_THRESHOLD
    settings.num_train_frames = NUM_TRAIN_FRAMES
    settings.num_test_frames  = NUM_TEST_FRAMES
    settings.num_encoder_layers = NUM_ENCODER_LAYERS
    settings.num_decoder_layers = NUM_DECODER_LAYERS
    settings.frozen_backbone_layers   = FROZEN_BACKBONE_LAYERS
    settings.freeze_backbone_bn_layers = FREEZE_BACKBONE_BN_LAYERS

    settings.crop_type = CROP_TYPE
    settings.max_scale_change = MAX_SCALE_CHANGE
    settings.max_gap = MAX_GAP

    ws = _world_size()
    # shard training per rank so global == DP
    settings.train_samples_per_epoch = max(1, TRAIN_SAMPLES_PER_EPOCH // ws)
    # keep **full** validation budget when rank-0 validates (parity)
    settings.val_samples_per_epoch = (
        max(1, VAL_SAMPLES_PER_EPOCH)
        if str(getattr(settings, 'val_mode', 'rank0')).lower() == 'rank0'
        else max(1, VAL_SAMPLES_PER_EPOCH // ws)
    )

    settings.val_epoch_interval = VAL_EPOCH_INTERVAL
    settings.num_epochs = NUM_EPOCHS

    settings.weight_giou = WEIGHT_GIOU
    settings.weight_clf  = WEIGHT_CLF
    settings.normalized_bbreg_coords = NORMALIZED_BBREG_COORDS
    settings.center_sampling_radius  = CENTER_SAMPLING_RADIUS
    settings.use_test_frame_encoding = USE_TEST_FRAME_ENCODING

    # ------ manual GA (keep DS GAS=1) ------
    manual_GAS = max(1, EFFECTIVE_BATCH_SIZE // max(PER_GPU_BATCH_SIZE * ws, 1))
    settings.train_micro_batch_size_per_gpu = PER_GPU_BATCH_SIZE
    settings.gradient_accumulation_steps = manual_GAS
    settings.manual_gradient_accumulation_steps = manual_GAS

    # DS config (only when BACKEND=ds0). Respect clipping selection.
    if settings.use_deepspeed:
        ds_clip = float(settings.gradient_clipping if settings.use_deepspeed_grad_clip else 0.0)
        settings.deepspeed_config = {
            "train_micro_batch_size_per_gpu": max(1, PER_GPU_BATCH_SIZE),
            "gradient_accumulation_steps": 1,          # manual GA in trainer
            "zero_optimization": {"stage": int(settings.deepspeed_zero_stage)},
            "gradient_clipping": ds_clip,              # on only when using DS clip
            # "fp16": {"enabled": bool(DEEPSPEED_FP16)},
            # "bf16": {"enabled": bool(DEEPSPEED_BF16)},
        }

    # --- datasets & transforms ---
    import ltr.data.transforms as tfm
    lasot_train       = Lasot(settings.env.lasot_dir, split='train')
    got10k_train      = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=None)
    coco_train        = MSCOCOSeq(settings.env.coco17_dir)
    vasttrack_train = VastTrack(settings.env.vasttrack_dir, split='train')

    got10k_val        = Got10k(settings.env.got10k_dir, split='votval')

    transform_joint = tfm.Transform(
        tfm.ToGrayscale(probability=0.05),
        tfm.RandomHorizontalFlip(probability=0.5)
    )
    transform_train = tfm.Transform(
        tfm.ToTensorAndJitter(0.2),
        tfm.RandomHorizontalFlip(probability=0.5),
        tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)
    )
    transform_val = tfm.Transform(
        tfm.ToTensor(),
        tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)
    )

    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    label_params = {'feature_sz': settings.feature_sz,
                    'sigma_factor': output_sigma,
                    'kernel_sz': settings.target_filter_sz}

    data_processing_train = processing.LTRBDenseRegressionProcessing(
        search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        crop_type=settings.crop_type,
        max_scale_change=settings.max_scale_change,
        mode='sequence',
        label_function_params=label_params,
        transform=transform_train,
        joint_transform=transform_joint,
        use_normalized_coords=settings.normalized_bbreg_coords,
        center_sampling_radius=settings.center_sampling_radius
    )
    data_processing_val = processing.LTRBDenseRegressionProcessing(
        search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        crop_type=settings.crop_type,
        max_scale_change=settings.max_scale_change,
        mode='sequence',
        label_function_params=label_params,
        transform=transform_val,
        joint_transform=transform_joint,
        use_normalized_coords=settings.normalized_bbreg_coords,
        center_sampling_radius=settings.center_sampling_radius
    )

    dataset_train = sampler.DiMPSampler([vasttrack_train, lasot_train, got10k_train, trackingnet_train, coco_train], [1, 1, 1, 1, 1],
        samples_per_epoch=settings.train_samples_per_epoch,
        max_gap=settings.max_gap,
        num_test_frames=settings.num_test_frames,
        num_train_frames=settings.num_train_frames,
        processing=data_processing_train
    )
    loader_train = LTRLoader('train', dataset_train, training=True,
                             batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    dataset_val = sampler.DiMPSampler(
        [got10k_val], [1],
        samples_per_epoch=settings.val_samples_per_epoch,
        max_gap=settings.max_gap,
        num_test_frames=settings.num_test_frames,
        num_train_frames=settings.num_train_frames,
        processing=data_processing_val
    )

    # keep per-GPU batch for rank-0 validation
    val_bs = settings.batch_size
    loader_val = LTRLoader('val', dataset_val, training=False,
                           batch_size=val_bs, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True,
                           epoch_interval=settings.val_epoch_interval, stack_dim=1)

    # --- model ---
    net = tompnet_JEPAp_vggt.tompnet50(
        filter_size=settings.target_filter_sz, backbone_pretrained=True, head_feat_blocks=0,
        head_feat_norm=True, final_conv=True, out_feature_dim=256, feature_sz=settings.feature_sz,
        frozen_backbone_layers=settings.frozen_backbone_layers,
        num_encoder_layers=settings.num_encoder_layers,
        num_decoder_layers=settings.num_decoder_layers,
        use_test_frame_encoding=settings.use_test_frame_encoding,
        use_activation_checkpointing=USE_ACTIVATION_CHECKPOINTING,
        ckpt_impl=settings.ckpt_impl,
    )

    root = get_root(net)
    # === Pretrained weights (exact DP-style) ===
    if PRETRAIN_PATH:
        JEPAc_net, _ = network_loading.load_network(checkpoint=PRETRAIN_PATH)  # DP structure
        root.feature_extractor.load_state_dict(JEPAc_net.feature_extractor.state_dict())
        root.bkMlp.load_state_dict(JEPAc_net.bkMlp.state_dict())
        root.head.load_state_dict(JEPAc_net.head.state_dict())
        root.JEPA_predictor_cls.load_state_dict(JEPAc_net.JEPA_predictor_cls.state_dict())
        root.JEPA_predictor_breg.load_state_dict(JEPAc_net.JEPA_predictor_breg.state_dict())
        root.WvggtLinearCls.load_state_dict(JEPAc_net.JEPA_predictor_cls.state_dict())

    net.feature_extractor_VGGT = inject_dora_into_backbone(
        net.feature_extractor_VGGT,
        r=16,
        alpha=16,
        dropout=0.0,
        freeze_base=True,
        verbose=True,
    )

    # Trainables: freeze only the backbone feature_extractor
    # === One-pass grad control ===
    # 1) freeze everything
    for p in net.parameters():
        p.requires_grad_(False)

    # 2) enable heads  want to train
    for m in [net.JEPA_predictor_cls, net.JEPA_predictor_breg, net.WvggtLinearCls,
            net.vggtDPTfeatMlp, net.vggtDPTfeatMlp_head, net.DiNO_VGGT_Gate, net.bkMlp, net.head]:
        for p in m.parameters():
            p.requires_grad_(True)

    # 3) enable DoRA adapters only in aggregator.global_blocks attention (qkv/proj)
    for n, p in net.feature_extractor_VGGT.named_parameters():
        is_adapter = ("lora_" in n) or ("dora_" in n)
        in_global  = ".aggregator.global_blocks." in n
        in_attn    = ".attn." in n
        p.requires_grad_(is_adapter and in_global and in_attn)




    objective = {
        'giou': GIoULoss(),
        'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold),
    }
    loss_weight = {'giou': settings.weight_giou, 'test_clf': settings.weight_clf}
    actor = actors.ToMPActor(net=net, objective=objective, loss_weight=loss_weight)

    # --- optimizer groups ---
    _root = get_root(actor.net)
    raw_groups = [
        {'params': list(_root.JEPA_predictor_cls.parameters()),  'lr': BASE_LRS["JEPA_predictor_cls"]},
        {'params': list(_root.JEPA_predictor_breg.parameters()), 'lr': BASE_LRS["JEPA_predictor_breg"]},
        {'params': list(_root.WvggtLinearCls.parameters()),      'lr': BASE_LRS["WvggtLinearCls"]},
        {'params': list(_root.vggtDPTfeatMlp.parameters()),      'lr': BASE_LRS["vggtDPTfeatMlp"]},
        {'params': list(_root.vggtDPTfeatMlp_head.parameters()), 'lr': BASE_LRS["vggtDPTfeatMlp_head"]},
        {'params': list(_root.DiNO_VGGT_Gate.parameters()),      'lr': BASE_LRS["DiNO_VGGT_Gate"]},
        {'params': list(_root.bkMlp.parameters()),          'lr': BASE_LRS["bkMlp"]},
        {'params': list(_root.head.parameters()),                'lr': BASE_LRS["head"]},
    ]

    adapter_params = collect_adapter_params(_root.feature_extractor_VGGT)
    if len(adapter_params) > 0:
        dora_group = {
            'params': adapter_params,
            'lr': BASE_LRS.get("DoRA_adapters", 2e-5),
            'weight_decay': 0.0
        }
        optimizer_params = [dora_group] + [g for g in raw_groups if len(g['params']) > 0]
    else:
        optimizer_params = [g for g in raw_groups if len(g['params']) > 0]

    optimizer = optim.AdamW(optimizer_params, weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)

    settings.dora_save_mode = "merged"       # or "adapters" or "both" or "none"
    settings.dora_save_every_epoch = True   # True if want each epoch
    settings.dora_save_tag = "MERGED_INFER_ONLY"  # suffix for the merged file name

    # --- trainer ---
    trainer = LTRTrainer_magav1(
        actor, [loader_train, loader_val], optimizer, settings, lr_scheduler,
        freeze_backbone_bn_layers=settings.freeze_backbone_bn_layers,
        use_GradScaler=False
    )

    if _rank() == 0:
        print0(f"[Training setup] backend={BACKEND}, world_size={_world_size()}, per_gpu_batch={PER_GPU_BATCH_SIZE}, "
               f"manual_GAS={manual_GAS}, effective_global_batch={PER_GPU_BATCH_SIZE * _world_size() * manual_GAS}, "
               f"train_samples_per_epoch_global={TRAIN_SAMPLES_PER_EPOCH}, "
               f"per_rank_train_samples={settings.train_samples_per_epoch}")
        print0(f"[Grad clip] DS={'ON' if settings.use_deepspeed_grad_clip else 'OFF'}, "
               f"torch={'ON' if settings.use_torch_grad_clip else 'OFF'}, "
               f"max_norm={settings.gradient_clipping}")

    print("[check] DoRA params: F",
        sum(p.numel() for p in adapter_params) if len(adapter_params) else 0)

    try:
        _root.feature_extractor_VGGT.print_trainable_parameters()
    except Exception:
        pass

    # --- Count total and trainable parameters ---
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print0(f"Total parameters: {total_params / 1e6:.2f} M")
    print0(f"Trainable parameters: {trainable_params / 1e6:.2f} M")
    print0(f"Trainable ratio: {trainable_params / total_params * 100:.2f} %")

    # input()
    print0("Starting training ...")
    trainer.train(settings.num_epochs, load_latest=True, fail_safe=True)
