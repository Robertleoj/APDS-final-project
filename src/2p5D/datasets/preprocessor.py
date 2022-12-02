from .utils import read_nii, DataPoint, resize_seg, resize_volume
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

    def process(self, scan_index):
        
        seg_file = f"{self.data_path}/segmentation-{scan_index}.nii"
        vol_file = f"{self.data_path}/volume-{scan_index}.nii"
        
        seg_arr = read_nii(seg_file).to(dtype=torch.long)
        vol_arr = read_nii(vol_file).to(dtype=torch.float)


        # print(seg_arr.shape)
        # print(vol_arr.shape)


        vol_arr = self.__normalize(vol_arr)


        full_vol_arr = vol_arr.permute(2, 0, 1).clone()
        full_seg_arr = seg_arr.permute(2, 0, 1).clone()


        vol_arr = vol_arr.to(dtype=torch.float32)


        vol_arr = vol_arr.permute(2, 0, 1).flip((0,))
        seg_arr = seg_arr.permute(2, 0, 1).flip((0,))

        vol_arr_slices = vol_arr.split(1)
        seg_arr_slices = seg_arr.split(1)


        # seg_arr_slices, seg_rem = self.__get_rem(seg_arr_slices)
        
        dp = DataPoint(
            full_vol=full_vol_arr,#full_vol_arr,
            full_seg=full_seg_arr,#full_seg_arr,
            slice_list=list(zip(vol_arr_slices, seg_arr_slices)),
        )

        return dp

# turn into tensors
# normalize
# handle differing sizes

    