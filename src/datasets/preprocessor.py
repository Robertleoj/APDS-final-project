from .utils import read_nii, DataPoint
import torch


class Preprocessor_2p5D:
    def __init__(self, config):

        self.data_path = config['unzipped_path']
        self.slice_number = config['slice_number']

    def __normalize(self, x):
        return (x - x.mean()) / x.std()

    def process(self, scan_index):
        
        seg_file = f"{self.data_path}/segmentation-{scan_index}.nii"
        vol_file = f"{self.data_path}/volume-{scan_index}.nii"
        
        seg_arr = torch.tensor(read_nii(seg_file))
        vol_arr = torch.tensor(read_nii(vol_file))

        full_vol_arr = vol_arr.clone()
        full_seg_arr = seg_arr.clone()

        vol_arr = self.__normalize(vol_arr)


        slice_list = []

        slice_idx = 0
        while vol_arr.shape[2] >= self.slice_number:
            
            tensor_idx = vol_arr.shape[2] - self.slice_number

            # slice_number = torch.Size(self.slice_number)

            vol_slice = vol_arr[:, :, -self.slice_number:]
            seg_slice = seg_arr[:, :, -self.slice_number:]

            vol_arr = vol_arr[:, :, :-self.slice_number]
            seg_arr = seg_arr[:, :, :-self.slice_number]

            slice_list.append((vol_slice, seg_slice, slice_idx, tensor_idx))

            slice_idx += 1

        dp = DataPoint(
            full_vol=full_vol_arr,
            full_seg=full_seg_arr,
            slice_list=slice_list,
            rem_vol=vol_arr,
            rem_seg=seg_arr
        )

        return dp

# turn into tensors
# normalize
# handle differing sizes

    