import nibabel as nib
import numpy as np
from collections import namedtuple
from scipy import ndimage
import json
import torch
import shutil
import os

DataPoint = namedtuple('DataPoint', (
    'full_vol', 'full_seg',
    'slice_list', 'rem_vol', 'rem_seg'
))

def check_orientation(ct_image, ct_arr):
    """
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param ct_image: NIfTI file
    :param ct_arr: array file
    :return: array after flipping
    """
    x, y, z = nib.aff2axcodes(ct_image.affine)
    if x != 'R':
        ct_arr = torch.flip(ct_arr, dims=(0,))
    if y != 'P':
        ct_arr = torch.flip(ct_arr, dims=(1,))
    if z != 'S':
        ct_arr = torch.flip(ct_arr, dims=(2,))
    return ct_arr


def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = torch.tensor(array)
    array = check_orientation(ct_scan, array)
    return(array)

#assumes path goes to directory to clean
def nuke(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def resize_volume(img, xy=128):
    return resize(img, xy, 'nearest')

def resize_seg(img, xy=128):
    return resize(img, xy, 'nearest')

def resize(img: torch.Tensor, xy, mode=None):
    if img.shape[-1] == 0:
        
        return torch.zeros((xy, xy, 0))

    img = img.numpy()

    desired_width = xy
    desired_height = xy

    # Get current depth
    current_depth = img.shape[-1]
    desired_depth = current_depth

    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1, mode=mode)

    return torch.tensor(img)


def get_split_indices(split_path):
    
    with open(split_path, 'r') as f:
        indices_dict = json.load(f)

        return (
            indices_dict['train'], 
            indices_dict['val'], 
            indices_dict['test']
        )


def liver_min_max(seg: torch.Tensor):
    """
    scan is Z x H x W
    """
    indices = seg.max(-1).values.max(-1).values.nonzero()

    return indices.min(), indices.max()






