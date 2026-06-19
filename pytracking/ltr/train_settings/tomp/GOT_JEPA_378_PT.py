import torch.optim as optim
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import tompnet_PT
import ltr.models.loss as ltr_losses
import ltr.actors.tracking as actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.models.loss.bbr_loss import GIoULoss

import ltr.admin.loading as network_loading
import os

def run(settings):
    settings.description = 'ToMP50'

    settings.batch_size = 13*4
    settings.num_workers = 8

    settings.multi_gpu = True

    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 1
    settings.feature_sz = 27
    settings.output_sz = settings.feature_sz * 16

    settings.center_jitter_factor = {'train': 0.5, 'test': 0.5}
    settings.scale_jitter_factor = {'train': 0.05, 'test': 0.05}

    settings.hinge_threshold = 0.05

    settings.num_train_frames = 8*2
    settings.num_test_frames = 8

    settings.num_encoder_layers = 6
    settings.num_decoder_layers = 6
    settings.frozen_backbone_layers = ['conv1', 'bn1', 'layer1', 'layer2']
    settings.freeze_backbone_bn_layers = True

    settings.crop_type = 'inside_major'
    settings.max_scale_change = 1.5
    settings.max_gap = 1

    settings.train_samples_per_epoch = 100000
    settings.val_samples_per_epoch = 10000

    settings.val_epoch_interval = 1

    settings.num_epochs = 60

    settings.weight_giou = 1.0
    settings.weight_clf = 200.0

    settings.weight_giouPT = 0.50
    settings.weight_clfPT = 100.0

    settings.normalized_bbreg_coords = True
    settings.center_sampling_radius = 1.0
    settings.use_test_frame_encoding = True  # Set to True to use the same as in the paper but is less stable to train.

    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=None)

    coco_train = MSCOCOSeq(settings.env.coco17_dir)

    got10k_val = Got10k(settings.env.got10k_dir, split='votval')


    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),

                                    )

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),

                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}

    data_processing_train = processing.LTRBDenseRegressionProcessing(search_area_factor=settings.search_area_factor,
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
                                                                     center_sampling_radius=settings.center_sampling_radius)

    data_processing_val = processing.LTRBDenseRegressionProcessing(search_area_factor=settings.search_area_factor,
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
                                                                   center_sampling_radius=settings.center_sampling_radius)

    dataset_train = sampler.PTSampler([lasot_train, got10k_train, trackingnet_train, coco_train], [1, 1, 1, 1],
                                        samples_per_epoch=settings.train_samples_per_epoch, max_gap=settings.max_gap,
                                        num_test_frames=settings.num_test_frames, num_train_frames=settings.num_train_frames,
                                        processing=data_processing_train, frame_step=1)


    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    dataset_val = sampler.PTSampler([got10k_val], [1], samples_per_epoch=settings.val_samples_per_epoch,
                                      max_gap=settings.max_gap, num_test_frames=settings.num_test_frames,
                                      num_train_frames=settings.num_train_frames, processing=data_processing_val, frame_step=1)


    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=settings.val_epoch_interval, stack_dim=1)

    JEPAc_weights_path = "got_jepa_finetune.tar"

    JEPAc_net, _ = network_loading.load_network(checkpoint=JEPAc_weights_path)


    net = tompnet_PT.tompnet50_PT(filter_size=settings.target_filter_sz, backbone_pretrained=True, head_feat_blocks=0,
                            head_feat_norm=True, final_conv=True, out_feature_dim=256, feature_sz=settings.feature_sz,
                            frozen_backbone_layers=settings.frozen_backbone_layers,
                            num_encoder_layers=settings.num_encoder_layers,
                            num_decoder_layers=settings.num_decoder_layers,
                            use_test_frame_encoding=settings.use_test_frame_encoding)


    net.feature_extractor.load_state_dict(JEPAc_net.feature_extractor.state_dict())
    net.bkMlp.load_state_dict(JEPAc_net.bkMlp.state_dict())

    net.head.load_state_dict(JEPAc_net.head.state_dict())

    net.JEPA_predictor_cls.load_state_dict(JEPAc_net.JEPA_predictor_cls.state_dict())
    net.JEPA_predictor_breg.load_state_dict(JEPAc_net.JEPA_predictor_breg.state_dict())


    for p in net.parameters():
        p.requires_grad_(False)

    for p in net.SideNetwork_D.parameters():
        p.requires_grad_(True)

    for p in net.SideNetwork_U.parameters():
        p.requires_grad_(True)

    for p in net.PTrackAttentionModel.parameters():
        p.requires_grad_(True)

    for p in net.PointEmbeddingNetwork.parameters():
        p.requires_grad_(True)

    for p in net.TFEcatmlp.parameters():
        p.requires_grad_(True)

    for p in net.PTlabelEmbeddingNetwork.parameters():
        p.requires_grad_(True)


    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = {'giou': GIoULoss(), 'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold)}

    loss_weight = {'giou': settings.weight_giou, 'test_clf': settings.weight_clf, 'giouPT': settings.weight_giouPT, 'test_clfPT': settings.weight_clfPT}

    actor = actors.ToMPActor_PTcur(net=net, objective=objective, loss_weight=loss_weight)

    optimizer = optim.AdamW([

        {'params': actor.net.SideNetwork_D.parameters(), 'lr': 1e-4},
        {'params': actor.net.SideNetwork_U.parameters(), 'lr': 1e-4},
        {'params': actor.net.PTrackAttentionModel.parameters(), 'lr': 1e-4},

        {'params': actor.net.PointEmbeddingNetwork.parameters(), 'lr': 1e-4},

        {'params': actor.net.TFEcatmlp.parameters(), 'lr': 1e-4},
        {'params': actor.net.PTlabelEmbeddingNetwork.parameters(), 'lr': 1e-4},

    ], lr=2e-4, weight_decay=0.0001)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler,
                         freeze_backbone_bn_layers=settings.freeze_backbone_bn_layers)


    trainer.train(settings.num_epochs, load_latest=True, fail_safe=True)
