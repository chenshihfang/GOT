import math
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
# from torchvision.models.resnet import model_urls
from .base import Backbone
import torch.nn.functional as F


import copy
import os
import os.path as osp

# from ..transformer.heads import Transformer, LayerNorm
# from .vit_encoder import build_vit_encoder

import urllib
# import mmcv
from functools import partial
# from mmcv.runner import load_checkpoint
import torch
import itertools


print_misaligned = 1

try:
    from cotracker2.co_tracker.cotracker.predictor import CoTrackerPredictor
    from vggt.models.vggt import VGGT
    # from FastVGGT.vggt.models.vggt import VGGT
    print("imported VGGT.")
except:
    print("Could not import VGGT.")
    pass

import os
import sys
import traceback



def dinov2_DA3(checkpoint_path="depth-anything/da3-large", device="cuda"):
    """
    Load Depth Anything 3 base model from Hugging Face hub.

    Returns a DepthAnything3 object (wrapper with .inference(...)).
    """

    try:
        from depth_anything_3.api import DepthAnything3
        print("import DepthAnything3 OK")
    except Exception as e:
        print("import DepthAnything3 fail:", e)
        DepthAnything3 = None


    if DepthAnything3 is None:
        raise ImportError("DepthAnything3 not available, install depth-anything-3 and check PYTHONPATH.")

    print("Loading Depth Anything 3 model:", checkpoint_path)
    model = DepthAnything3.from_pretrained(checkpoint_path)
    model = model.to(device=device)
    model.eval()
    
    # input()
    return model



def dinov2_StreamVGGT(checkpoint_path="/data/pytrackingcsf/pytracking/ltr/StreamVGGT/src/streamvggt/ckpt/checkpoints.pth", device='cuda'):
    """
    Loads the StreamVGGT model from a local checkpoint file.
    """

    try:
        # This is the only import that should work, based on your symlink
        from streamvggt.models.streamvggt import StreamVGGT
        import streamvggt
        
        print("--- SUCCESS ---")
        print("imported StreamVGGT.")
        # This will print the exact file path Python is using
        print("Imported from:", streamvggt.__file__ if hasattr(streamvggt, "__file__") else streamvggt.__path__)
        
    except Exception as e:
        print("--- FAILED ---")
        print("Could not import StreamVGGT. See traceback below:")
        # This will print the *exact* error (e.g., "No module named X", etc.)
        traceback.print_exc()
        StreamVGGT = None

    # input()

    # 3) Import and print where it came from
    try:
        from streamvggt.models.streamvggt import StreamVGGT
        import streamvggt
        print("imported StreamVGGT from:", streamvggt.__file__ if hasattr(streamvggt, "__file__") else streamvggt.__path__)
    except Exception as e:
        print("Could not import StreamVGGT.")
        traceback.print_exc()
        StreamVGGT = None


    if StreamVGGT is None:
        raise ImportError("StreamVGGT class not loaded.")
    
    print("Loading StreamVGGT model from local checkpoint...")

    # 1. Instantiate the raw StreamVGGT model
    # NOTE: If StreamVGGT() requires non-default args, you must add them here.
    # Assuming the default constructor is correct:
    model = StreamVGGT()
    
    # 2. Load the checkpoint state dict from the file path
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 3. Load the state dict into the model
    # (Following your snippet, assuming the .pth file is the state dict)
    model.load_state_dict(checkpoint, strict=True)
    
    # 4. Move to device and set to eval mode
    # The ToMPnet .train() or .eval() calls will override this,
    # but .eval() is a safe default for a pre-trained backbone.
    model.to(device).eval()
    
    print("StreamVGGT model loaded successfully from local checkpoint.")
    # input()
    return model


def dinov2_VGGT(checkpoint_path="facebook/VGGT-1B", device='cuda'):

    dinov2 = VGGT.from_pretrained(checkpoint_path).to(device)
    return dinov2



# def dinov2_VGGT(checkpoint_path="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt", device='cuda'): # 11.1 ~ 12.2

#     model = VGGT()
#     model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_path))
#     dinov2 = model.to(device)

#     return dinov2


# def dinov2_VGGT():

#     return VGGT()


def cotracker_predictor(checkpoint_path="ltr/cotracker2/co_tracker/checkpoints/cotracker2.pth", device='cuda'):
    cotracker = CoTrackerPredictor(checkpoint=checkpoint_path).to(device)
    return cotracker



def VIT(SEARCH_SIZE, ENCODER_TYPE, ENCODER_PRETRAIN_TYPE, train_encoder):

    encoder = build_vit_encoder(SEARCH_SIZE, ENCODER_TYPE, ENCODER_PRETRAIN_TYPE, train_encoder)

    return encoder

def dinov2(model_name = "dinov2_vitl14"):

    dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
    return dinov2


def dinov3(dino_name):
    # Map model name to its weight file name (if you ever want local defaults)
    model_weights = {
        "dinov3_vitl16": "dinov3_vitl16_lvd1689m.pth",
        "dinov3_convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m.pth",
    }
    
    # Check if model is supported
    if dino_name not in model_weights:
        raise ValueError(f"Unsupported DINOv3 model: {dino_name}")

    REPO_DIR = "/data/pytrackingcsf/dinov3/"

    print(f"Loading {dino_name} with weights from remote URL.")
    model = torch.hub.load(
        repo_or_dir=REPO_DIR,
        model=dino_name,
        source='local',
        weights="https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiMjRveTIwczV0d2RqNTcwcjVxdWFpbW0zIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0MDY3MzZ9fX1dfQ__&Signature=JN1t-OncFtkNjeoe0m7X%7ErfXvGR9gUuEUlaQc1bamJ7M%7ETV7LKXFZVoeRLxJRnfqL1vN8iZIlG-ENUAieQgvPiabTKIJdOxsy3SkavFhjPjuwSOHMdLV3B49lAVVDWR1rpkOUlCnHOGQJDVHTk0IB27qzFGuamPi%7EwyaYO565AiVtZ%7E9Dobmen5RmW5vF2O557GhasIS0e9oR7zjqj%7EOmtSjXWtLr5bzTW4-PQad9WCWvBo6KaV%7EsoLmq1ugDzLupXdC5Xeyb2DFQbA0A4kmD1F4gL4PUzY50ubn90V1wqvj0UXGLZe2doVWi0IV2PADte89o0M7dra2p6rwrXllEw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=2433063113757620"
    )

    model.to("cuda")
    model.eval()
    return model

def dinov3_(dino_name):
    # This dictionary maps a model name to its weight file name
    model_weights = {
        "dinov3_vitl16": "dinov3_vitl16_lvd1689m.pth", # Example file name
        "dinov3_convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m.pth",
        # Add other models and their weight file names here
    }
    
    # Check if the requested model is supported
    if dino_name not in model_weights:
        raise ValueError(f"Unsupported DINOv3 model: {dino_name}")

    MODEL_WEIGHTS_FILE = model_weights[dino_name]
    WEIGHT_PATH = "/data/pytrackingcsf/dinov3_vitl16.pth"
    REPO_DIR = "/data/pytrackingcsf/dinov3/"

    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"Model weights not found at: {WEIGHT_PATH}")

    # --- MODIFICATION STARTS HERE ---
    
    # 1. Use torch.hub.load to build the model architecture ONLY.
    #    Do NOT pass the 'weights' argument here directly.
    print(f"Loading {dino_name} architecture from local hub.")
    model = torch.hub.load(
        repo_or_dir=REPO_DIR,
        model=dino_name,  # This should be the function name in hubconf.py
        source='local',
        weights="https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiMjRveTIwczV0d2RqNTcwcjVxdWFpbW0zIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0MDY3MzZ9fX1dfQ__&Signature=JN1t-OncFtkNjeoe0m7X%7ErfXvGR9gUuEUlaQc1bamJ7M%7ETV7LKXFZVoeRLxJRnfqL1vN8iZIlG-ENUAieQgvPiabTKIJdOxsy3SkavFhjPjuwSOHMdLV3B49lAVVDWR1rpkOUlCnHOGQJDVHTk0IB27qzFGuamPi%7EwyaYO565AiVtZ%7E9Dobmen5RmW5vF2O557GhasIS0e9oR7zjqj%7EOmtSjXWtLr5bzTW4-PQad9WCWvBo6KaV%7EsoLmq1ugDzLupXdC5Xeyb2DFQbA0A4kmD1F4gL4PUzY50ubn90V1wqvj0UXGLZe2doVWi0IV2PADte89o0M7dra2p6rwrXllEw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=2433063113757620"
        # Remove 'weights=WEIGHT_PATH' from this call
    )

    # 2. Manually load the state dictionary from your local .pth file
    print(f"Loading weights from: {WEIGHT_PATH}")
    # It's good practice to load to CPU first to avoid CUDA memory issues during loading
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu") 
    model.load_state_dict(state_dict) # Apply the loaded weights to the model
    
    # --- MODIFICATION ENDS HERE ---

    model.to("cuda") # Move model to GPU after loading weights
    model.eval()
    
    return model


# from dinov3.hub.backbones import dinov3_vitl16 as load_dinov3_vitl16
# from dinov3.models.vision_transformer import DinoVisionTransformer

# def dinov3(model_name="dinov3_vitl16"):
#     # Define the model parameters based on the specific DINOv3 model
#     # For ViT-L/16, these are common parameters. You may need to check the
#     # dinov3 source code for the exact values.
#     # Note: These values might need adjustment based on the DINOv3 version.
#     model_params = {
#         'patch_size': 16,
#         'embed_dim': 1024,
#         'depth': 24,
#         'num_heads': 16,
#         'mlp_ratio': 4,
#         'num_register_tokens': 4,
#         'interpolate_mode': 'bicubic'
#     }

#     # Create the model instance directly without the hub
#     model = DinoVisionTransformer(**model_params)
    
#     # Load your local weights
#     weights_DIR = "/data/pytrackingcsf/dinov3_vitl16.pth"
#     checkpoint = torch.load(weights_DIR, map_location='cpu')

#     # Load the state dictionary from the checkpoint file
#     # DINOv3 checkpoints often store the state dict under a 'teacher' key

#     # Try loading the 'student' key instead of 'teacher'
#     # if 'student' in checkpoint:
#     #     state_dict = checkpoint['student']
#     # elif 'teacher' in checkpoint:
#     #     state_dict = checkpoint['teacher']
#     # else:
#     #     # Fallback to the entire checkpoint if no specific key is found
#     #     state_dict = checkpoint
#     state_dict = checkpoint
#     # model.load_state_dict(state_dict, strict=False)
#     model.load_state_dict(state_dict, strict=True)
    
#     # Move the model to the GPU
#     model.cuda()
    
#     return model


# def dinov3(model_name="dinov3_vitl16"):
#     # This function is now simplified and directly uses the installed package.
#     # The `model_name` argument might become redundant if you're only loading one specific model.
#     weights_DIR = "/data/pytrackingcsf/dinov3_vitl16.pth"
    
#     # Load the model directly from the installed package
#     dinov3_model = load_dinov3_vitl16(weights=weights_DIR)
    
#     # Move to GPU
#     dinov3_model.cuda()
    
#     return dinov3_model

# def dinov3(model_name = "dinov3_vitl16"):

#     # examples of available DINOv3 models:
#     # MODEL_DINOV3_VITS = "dinov3_vits16"
#     # MODEL_DINOV3_VITSP = "dinov3_vits16plus"
#     # MODEL_DINOV3_VITB = "dinov3_vitb16"
#     # MODEL_DINOV3_VITL = "dinov3_vitl16"
#     # MODEL_DINOV3_VITHP = "dinov3_vith16plus"
#     # MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

#     # wget -O dinov3_vitl16.pth ""

#     REPO_DIR = "/data/pytrackingcsf/dinov3/"
#     weights_DIR = "/data/pytrackingcsf/dinov3_vitl16.pth"
    

#     # DINOv3 ViT models pretrained on web images
#     # dinov3 = torch.hub.load(REPO_DIR, model_name, source='local', weights=weights_DIR)

#     dinov3 = torch.hub.load(
#         repo_or_dir="facebookresearch/dinov3",
#         model="dinov3_vitl16",
#         weights="/data/pytrackingcsf/dinov3_vitl16.pth",
#     )

#     # dinov3 = torch.hub.load('facebookresearch/dinov3', model_name)


#     dinov3.cuda()

#     return dinov3


def vjepa2_hub(model_name='vjepa2_vit_large', out_layers=None):
    """
    Loads a V-JEPA model, configuring it to output features
    from specific intermediate layers.
    """
    # The hub entry returns (model, preprocessor).
    # Pass out_layers directly to the model constructor via torch.hub.load
    model, _ = torch.hub.load(
        'facebookresearch/vjepa2',
        model_name,
        out_layers=out_layers,
        force_reload=False
    )
    return model

def radio():

    model_version = "radio_v2.1"
    # model_version = "radio_v2"
    # model_version = "e-radio_v2"

    model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)

    # model.cuda().eval()

    return model


# 3. Define utility functions
def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

# 2. Define utility classes
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    # @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)
    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )
    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    return depther
    
def dinov2_Depth(backbone_model, backbone_scale):

    # backbone_model.eval()
    backbone_model.cuda()
    # 5. Load configuration
    HEAD_DATASET = "nyu"
    HEAD_TYPE = "dpt"
    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


    BACKBONE_SIZE = backbone_scale
    backbone_archs = {"small": "vits14", "base": "vitb14", "large": "vitl14", "giant": "vitg14"}
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"
    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    model = create_depther(cfg, backbone_model=backbone_model, backbone_size=backbone_name, head_type=HEAD_TYPE)
    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.cuda()

    # print("model dpt\n", model)
    # input()

    # print("model dpt DPTHead:\n", model.decode_head)
    # input()

    # output = model.decode_head(input_tensor)
    # print("output decode_head:\n", output)
    # input()


    return model

    
def resnet50(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck, [3, 4, 6, 3], output_layers, **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        except: 
            pass

    # model = torchvision.models.resnet50() # note: this could be any model
    # model = torch.compile(model) # <- magic happens!
    
    return model

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.use_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):
    """ ResNet network module. Allows extracting specific feature blocks."""
    def __init__(self, block, layers, output_layers, num_classes=1000, inplanes=64, dilation_factor=1, frozen_layers=()):
        self.inplanes = inplanes
        super(ResNet, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]
        self.layer1 = self._make_layer(block, inplanes, layers[0], dilation=max(dilation_factor//8, 1))
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=stride[0], dilation=max(dilation_factor//4, 1))
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=stride[1], dilation=max(dilation_factor//2, 1))
        self.layer4 = self._make_layer(block, inplanes*8, layers[3], stride=stride[2], dilation=dilation_factor)

        out_feature_strides = {'conv1': 4, 'layer1': 4, 'layer2': 4*stride[0], 'layer3': 4*stride[0]*stride[1],
                               'layer4': 4*stride[0]*stride[1]*stride[2]}

        # TODO better way?
        if isinstance(self.layer1[0], BasicBlock):
            out_feature_channels = {'conv1': inplanes, 'layer1': inplanes, 'layer2': inplanes*2, 'layer3': inplanes*4,
                               'layer4': inplanes*8}
        elif isinstance(self.layer1[0], Bottleneck):
            base_num_channels = 4 * inplanes
            out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2,
                                    'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        else:
            raise Exception('block not supported')

        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(inplanes*8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        if layer is None:
            return self._out_feature_strides
        else:
            return self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        if layer is None:
            return self._out_feature_channels
        else:
            return self._out_feature_channels[layer]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        x = self.layer4(x)

        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self._add_output_and_check('fc', x, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


def resnet_baby(output_layers=None, pretrained=False, inplanes=16, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, inplanes=inplanes, **kwargs)

    if pretrained:
        raise NotImplementedError
    return model


def resnet18(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model




def resnet101(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))


    model = ResNet(Bottleneck, [3, 4, 23, 3], output_layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model




from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

try:
    from timm.models.layers import drop, drop_path, trunc_normal_
except:
    pass


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        # print("x.shape NCHW", x.shape) # torch.Size([2, 2048, 16, 16])
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        # print("x.shape (HW)NC", x.shape) # torch.Size([256, 2, 2048])
        # print("x.mean(dim=0, keepdim=True).shape", x.mean(dim=0, keepdim=True).shape) # torch.Size([1, 2, 2048])

        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        # print("x.shape cat", x.shape) # torch.Size([257, 2, 2048])

        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        # print("x.shape MHSA", x.shape) # torch.Size([257, 2, 1024])

        x = x.permute(1, 2, 0)

        # print("x.shape MHSA permute", x.shape) # torch.Size([2, 1024, 257])

        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)

        # print("global_feat.shape", global_feat.shape) # torch.Size([2, 1024])
        # print("feature_map.shape", feature_map.shape) # torch.Size([2, 1024, 16, 16])
        # exit()

        return global_feat, feature_map
