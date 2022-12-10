# Cell
from datasets.preprocessor import Preprocessor_2p5D
from datasets.utils import get_split_indices, liver_min_max
import torch
import gc
import os
from load_config import load_config 


# Cell
config = load_config()

# Cell
pre = Preprocessor_2p5D(config)

train_split, _, _, = get_split_indices('./datasets/split.json')


def get_liver_z_pct_bounds(seg):
    depth = seg.size(0)
    liver_start, liver_end = liver_min_max(seg)

    return (liver_start / depth).item(), (liver_end / depth).item()

# dp = pre.process(train_split[1])
# vol, seg = dp.full_vol, dp.full_seg

# Cell
# get_liver_z_pct_bounds(seg)

# Cell
from tqdm import tqdm

maxes = []
mins = []

for scan_idx in tqdm(train_split):
    dp = pre.process(scan_idx)
    _, seg = dp.full_vol, dp.full_seg

    dp = None

    gc.collect()

    mn, mx = get_liver_z_pct_bounds(seg)
    maxes.append(mx)
    mins.append(mn)

print(list(zip(mins, maxes)))
print(min(mins), max(maxes))

# Cell
# seg.shape

# Cell
# %matplotlib ipympl
# from plotting.plotting import make_visual
# make_visual(vol, seg)

# Cell



