
from models.UNet2p5D.Unet import Unet2p5D
from glob import glob
import os
import torch


def dice_all(mask1, mask2, mask_classes):
    dices = {}
    for name, v in mask_classes.items():
        dices[name] = dice(mask1, mask2, v)

    return dices


def model_from_config(config):
    return Unet2p5D(
        dim=3,
        n_classes=config['n_classes'],
        dim_mults=config['dim_mults'],
        attn_heads=config['attn_heads'],
        attn_head_dim=config['attn_head_dim'],
        n_res_blocks=config['n_res_blocks']
    )

def inverse_norm(x,config):
    low = config['clip_lower']
    high = config['clip_upper']

    return (x * (high - low) + low)


def dice(mask1, mask2, val):
    mask1 = mask1 == val
    mask2 = mask2 == val

    TP = ((mask1 == 1) &  (mask2 == 1)).sum()

    denom = mask1.sum() + mask2.sum()

    return (2 * TP / denom).item()

def get_newest_chekpoint(ckpts_path):

    ckpt_paths = glob(f"{ckpts_path}/*.ckpt")
    ckpt_tuples = []
    for path in ckpt_paths:
        splitted = path.split('_')
        epoch = int(splitted[-2])
        iter = int(splitted[-1].split('.')[0])

        ckpt_tuples.append((epoch, iter))

    epoch, iter = max(ckpt_tuples)
    return(epoch, iter)


def load_newest(ckpts_path, net, device='cpu'):

    epoch, iter = get_newest_chekpoint(ckpts_path)

    net = load_checkpoint(net, epoch, iter, ckpts_path)
    return net



def load_checkpoint(net, epoch, iter, ckpts_path, device="cpu"):
    path = f"{ckpts_path}/weights_{epoch}_{iter}.ckpt"
    if not os.path.exists(path):
        raise ValueError(f"Checkpoint ({epoch=}, {iter=}) does not exist.")
    
    net.load_state_dict(
        torch.load(path, map_location=device)
    )

    return net

