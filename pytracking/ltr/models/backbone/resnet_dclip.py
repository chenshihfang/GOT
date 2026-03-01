import math
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from .base import Backbone
import torch.nn.functional as F

# === DenseCLIP
# pip install -U openmim
# mim install mmengine
# mim install mmcv-full
from mmcv.runner import init_dist
import torch.distributed as dist
import copy
import os
import os.path as osp
from ..transformer.heads import Transformer, LayerNorm

print_misaligned = 1

CLIPResNetWithAttention_pretrained = "/home/sfchen94/dclip/DenseCLIP/segmentation/pretrained/RN50.pt"
# CLIPResNetWithAttention_pretrained = "/home/sfchen94/dclip/DenseCLIP/segmentation/pretrained/RN101.pt"
# CLIPResNetWithAttention_pretrained = "/home/sfchen94/dclip/DenseCLIP/segmentation/pretrained/RN50x16.pt"
# CLIPResNetWithAttention_pretrained = "/home/sfchen94/dclip/DenseCLIP/segmentation/pretrained/RN50x64.pt"

CLIPTextContextEncoder_pretrained = "/home/sfchen94/dclip/DenseCLIP/segmentation/pretrained/RN50.pt"

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
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

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



# === DenseCLIP

def resnet_dclip_backbone():
    """Constructs a ResNet-DCLIP model.
    """

    model_res_dclip = CLIPResNetWithAttention()

    # input()
    # multi_gpu = 1
    # if multi_gpu:
    #     model_res_dclip = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_res_dclip)
    # print("model_res_dclip ", model_res_dclip)
    # input()
    return model_res_dclip

# def vit_dclip_backbone():
#     """Constructs a Vit-DCLIP model.
#     """
#     model_vit_dclip = CLIPVisionTransformer()

#     return model_vit_dclip

def resnet_dclip_text_encoder():
    """A
    """

    model_CLIPTextContextEncoder = CLIPTextContextEncoder()

    return model_CLIPTextContextEncoder

# from collections import OrderedDict
# from torch import nn
# import math

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from timm.models.layers import drop, drop_path, trunc_normal_

### for CLIPResNetWithAttention
class Bottleneck_dclip(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck_dclip.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

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
### for CLIPResNetWithAttention

class CLIPResNetWithAttention(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers=[3, 4, 6, 3], output_dim=1024, input_resolution=512, width=64, pretrained=None, **kwargs): # 50
    # def __init__(self, layers=[3, 4, 6, 3], output_dim=1024, input_resolution=576, width=64, pretrained=None, **kwargs): # 50

    # def __init__(self, layers=[3, 4, 23, 3], output_dim=512, input_resolution=512, width=64, pretrained=None, **kwargs): # 101
    # def __init__(self, layers=[6, 8, 18, 8], output_dim=768, input_resolution=512, width=96, pretrained=None, **kwargs): # 50x16
    # def __init__(self, layers=[3, 15, 36, 10], output_dim=1024, input_resolution=512, width=128, pretrained=None, **kwargs): # 50x64

        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers # Bottleneck
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension

        # 50 101
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)
        # self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        
        pretrained = CLIPResNetWithAttention_pretrained
        
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]
                    if 'positional_embedding' in new_k:
                        if self.attnpool.positional_embedding.shape != state_dict[new_k].shape:
                            if print_misaligned:
                                print(f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.attnpool.positional_embedding.shape}')
                            cls_pos = state_dict[new_k][0:1, :]
                            H = W = self.input_resolution // 32
                            old_h = int(math.sqrt(state_dict[new_k][1:,].shape[0]))
                            # print("shape", state_dict[new_k][1:,].shape) # 50 and 101: 49, 2048 # 50x64 196 4096
                            # print("old_h", old_h) # 50 and 101 7 # 50x64 14
                            # print("cls_pos.shape[1]", cls_pos.shape[1]) # 50 and 101 2048 # 50x64 4096
                            # input()
                            spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, old_h, old_h, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                            spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            state_dict[new_k] = positional_embedding
                            # print("self.attnpool.positional_embedding.shape", self.attnpool.positional_embedding.shape) # 50 and 101: 257 2048
                            # print("state_dict[new_k].shape", state_dict[new_k].shape)
                            # input()
                            assert self.attnpool.positional_embedding.shape == state_dict[new_k].shape

            u, w = self.load_state_dict(state_dict, False)

            if print_misaligned:
                print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck_dclip(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck_dclip.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck_dclip(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)


        x = stem(x)
        outs = []
        x = self.layer1(x)
        outs.append(x)


        x = self.layer2(x)
        # print("x2.shape", x.shape) # 3 512 64 64
        outs.append(x)

        x = self.layer3(x)
        outs.append(x)
        # print("x3.shape", x.shape) # 3 1024 32 32
        # input()

        x = self.layer4(x)
        outs.append(x)
        # print("x4.shape", x.shape) # 3 2048 16 16
        # input()

        x_global, x_local = self.attnpool(x)
        outs.append([x_global, x_local])

        # print("CLIPResNetWithAttention end")
        # input()

        return tuple(outs)



class CLIPTextContextEncoder(nn.Module):
    def __init__(self, 

                #  context_length=11, # 36
                #  context_length=14, # 84 48
                # context_length=8, # 24
                context_length=10, # 82 44
                #  context_length=13
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained

        pretrained = CLIPTextContextEncoder_pretrained

        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        if print_misaligned:
                            print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)

            if print_misaligned:
                print(u, w, 'are misaligned params in text encoder')

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context):
        
        # after tokenize -> text
        # print("text", text)
        # input()
        # print("text.shape", text.shape) # torch.Size([3, 5])

        x_text = self.token_embedding(text)  # n_clas, n_text, C

        # print("x_text", x_text)
        # print("x_text.shape", x_text.shape) # torch.Size([3, 5, 512])
        # print("context.shape", context.shape) # torch.Size([1, 8, 512])
        # input()

        K, N1, C = x_text.shape
        B, N2, C = context.shape

        eos_indx = text.argmax(dim=-1) + N2
        eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

        x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
        context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

        # print("x_text.shape ", x_text.shape) # torch.Size([1, 3, 5, 512])
        # print("context.shape ", context.shape) # torch.Size([1, 3, 8, 512])

        # print("x_text[:,:,0:1].shape ", x_text[:,:,0:1].shape) # torch.Size([1, 3, 1, 512])
        # print("x_text[:, :, 1:].shape ", x_text[:, :, 1:].shape) # torch.Size([1, 3, 4, 512])

        x = torch.cat([x_text[:,:,0:1], context, x_text[:, :, 1:]], dim=2).reshape(B*K, N1+N2, C)

        # print("x.shape ", x.shape) # torch.Size([5, 13, 512]) # torch.Size([3, 13, 512]) 

        # print("x ", x)
        #  [-3.5915e-03, -6.1417e-03, -1.7891e-03,  ..., -8.2245e-03,
        #   -5.1758e-01, -4.4656e-04],
        #  [-3.4882e-02, -4.3182e-03, -1.0056e-02,  ...,  1.0078e-02,
        #    7.0038e-03, -2.5925e-02],

        # print("x_text[:,:,0:1]", x_text[:,:,0:1])
        # print("x_text[:, :, 1:]", x_text[:, :, 1:])
        # input()

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(B, K, self.embed_dim)
        return x

####################