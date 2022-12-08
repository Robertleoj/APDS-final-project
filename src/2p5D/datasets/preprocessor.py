from .utils import (
    read_nii_scan, 
    DataPoint, 
    resize_seg, 
    resize_volume,
    read_nii
)

import torch


class Preprocessor_2p5D:
    def __init__(self, config):

        self.data_path = config['unzipped_path']
        self.clip_lower = config['clip_lower']
        self.clip_upper = config['clip_upper']

    def __normalize(self, x:torch.Tensor):
        # return (x - x.mean()) / x.std()
        x = x.clamp(self.clip_lower, self.clip_upper)
        x = (x - self.clip_lower) / (self.clip_upper - self.clip_lower)
        return x

    def get_seg(self, idx):
        """
        WARNING: 
        this does not return the mask in the correct orientation
        """

        seg_file = f"{self.data_path}/segmentation-{idx}.nii"
        return read_nii(seg_file)

    def get_scan_no_slice(self, idx):
        seg_file = f"{self.data_path}/segmentation-{idx}.nii"
        vol_file = f"{self.data_path}/volume-{idx}.nii"

        vol_arr, seg_arr = read_nii_scan(vol_file, seg_file)

        seg_arr = seg_arr.to(dtype=torch.uint8)

        vol_arr = self.__normalize(vol_arr).to(dtype=torch.float32)

        vol_arr = vol_arr.permute(2, 0, 1).flip((0,))
        seg_arr = seg_arr.permute(2, 0, 1).flip((0,))

        return vol_arr, seg_arr

      

    def process(self, scan_index):

        vol_arr, seg_arr = self.get_scan_no_slice(scan_index)
        
        full_vol_arr = vol_arr.clone()
        full_seg_arr = seg_arr.clone()

        vol_arr_slices = vol_arr.split(1)
        seg_arr_slices = seg_arr.split(1)


        # seg_arr_slices, seg_rem = self.__get_rem(seg_arr_slices)
        
        dp = DataPoint(
            full_vol=full_vol_arr,#full_vol_arr,
            full_seg=full_seg_arr,#full_seg_arr,
            slice_list=list(zip(vol_arr_slices, seg_arr_slices)),
        )

        return dp

    