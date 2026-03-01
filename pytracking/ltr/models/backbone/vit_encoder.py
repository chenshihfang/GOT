"""
Encoder modules: we use ViT for the encoder.
"""

from torch import nn
from .utils.misc import is_main_process

from .vit_models import vit as vit_module

class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
        super().__init__()
        open_blocks = open_layers[2:]
        open_items = open_layers[0:2]
        for name, parameter in encoder.named_parameters():

            if not train_encoder:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze == True:
                    parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !

        self.body = encoder
        self.num_channels = num_channels

    def forward(self, images_list):
        xs = self.body(images_list)
        return xs


class Encoder(EncoderBase):
    """ViT encoder."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 pretrain_type: str,
                 search_size: int,
                 search_number: int,
                #  template_size: int,
                #  template_number: int,
                 open_layers: list,
                 cfg=None):

        ENCODER_DROP_PATH = 0
        
        ENCODER_USE_CHECKPOINT = False
        # ENCODER_USE_CHECKPOINT = True

        print("ENCODER_USE_CHECKPOINT", ENCODER_USE_CHECKPOINT)

        if "vit" in name.lower():
            encoder = getattr(vit_module, name)(pretrained=is_main_process(), pretrain_type=pretrain_type,
                                                       search_size=search_size, 
                                                    #    template_size=template_size,
                                                       search_number=search_number, 
                                                    #    template_number=template_number,
                                                       drop_path_rate=ENCODER_DROP_PATH,
                                                       use_checkpoint=ENCODER_USE_CHECKPOINT
                                                      )
            if "_base_" in name:
                num_channels = 768
            elif "_large_" in name:
                num_channels = 1024
            elif "_huge_" in name:
                num_channels = 1280
            else:
                num_channels = 768

        else:
            raise ValueError()
        super().__init__(encoder, train_encoder, open_layers, num_channels)



def build_vit_encoder(SEARCH_SIZE = 384, ENCODER_TYPE= "vit_base_patch16", ENCODER_PRETRAIN_TYPE= "mae", train_encoder = True):

    print("ENCODER_TYPE", ENCODER_TYPE)

    ENCODER_MULTIPLIER = 0.1  # encoder's LR = this factor * LR
    FREEZE_ENCODER = False # for freezing the parameters of encoder

    # SEARCH_SIZE = 256
    # SEARCH_SIZE = 288
    # SEARCH_SIZE = 324
    # SEARCH_SIZE = 361
    # SEARCH_SIZE = 384
    SEARCH_NUMBER = 1

    TEMPLATE_SIZE = SEARCH_NUMBER
    # TEMPLATE_NUMBER = 2
    ENCODER_OPEN = []

    # train_encoder = (ENCODER_MULTIPLIER > 0) and (FREEZE_ENCODER == False)

    print("train_encoder", train_encoder)
    encoder = Encoder(ENCODER_TYPE, train_encoder,
                      ENCODER_PRETRAIN_TYPE,
                      SEARCH_SIZE, SEARCH_NUMBER,
                    #   TEMPLATE_SIZE, 
                    #   TEMPLATE_NUMBER,
                      ENCODER_OPEN)
    return encoder