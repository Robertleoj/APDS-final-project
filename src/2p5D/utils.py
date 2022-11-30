
from models.UNet2p5D.Unet import Unet2p5D


def dice_all(mask1, mask2, mask_classes):
    dices = {}
    for name, v in mask_classes.items():
        dices[name] = dice(mask1, mask2, v)

    return dices


def model_from_config(config):
    return Unet2p5D(
        dim=config['slice_number'],
        n_classes=config['n_classes'],
        dim_mults=config['dim_mults'],
        attn_heads=config['attn_heads'],
        attn_head_dim=config['attn_head_dim'],
        n_res_blocks=config['n_res_blocks']
    )


def dice(mask1, mask2, val):
    mask1 = mask1 == val
    mask2 = mask2 == val

    TP = ((mask1 == 1) &  (mask2 == 1)).sum()

    denom = mask1.sum() + mask2.sum()

    return 2 * TP / denom

