from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone
 
def parameters():
    params = TrackerParams()


    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.train_feature_size = 27
    params.feature_stride = 16
    params.image_sample_size = params.train_feature_size*params.feature_stride
    params.search_area_scale = 5
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 2
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = False
    params.augmentation = {}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True
    params.conf_ths = 0.9
    params.search_area_rescaling_at_occlusion = True

    ### PT
    params.wPT = 1
    params.track_vis_prob_thres = 0.85
    params.frames_step = 8
    params.TFwPTT = 1
    params.wbbreg = 0
    params.iniLabelPad = 1
    params.havenouptcon = 0
    params.ocpad = 0
    params.point_constraint = 1
    params.TrackS = "Last"
    #
    params.iniFramePad = 1
    params.check_score_visflag = 1
    params.point_constraintall = 0
    params.mix_PT_ToMP_thres = 1
    params.visThresIn = 1
    params.wpcls = 1
    params.wpbreg = 1
    params.sameTFE = 0
    ### PT

    params.net = NetWithBackbone(net_path="GOT_JEPA_378.tar", use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'

    params.use_gt_box = True
    params.plot_iou = True

    return params



