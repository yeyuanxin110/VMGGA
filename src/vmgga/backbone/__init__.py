from .resnet_fpn import (ResNetFPN_8_2, ResNetFPN_16_4, Dinov3,
                         ResNetFPN_8_2_without_dinov3, Dinov3Only_FPN_8_2)


def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            if config.get('dinov3_name') is not None:
                return ResNetFPN_8_2(config['resnetfpn'])
            else:
                return ResNetFPN_8_2_without_dinov3(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    else:
        raise ValueError(f"VMGGA.BACKBONE_TYPE {config['backbone_type']} not supported.")
