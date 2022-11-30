from .utils import read_nii, DataPoint, resize_seg, resize_volume
import torch


class Preprocessor_2p5D:
    def __init__(self, config):

        self.data_path = config['unzipped_path']
        self.slice_number = config['slice_number']
        self.img_size = config['img_size']
        self.clip_lower = config['clip_lower']
        self.clip_upper = config['clip_upper']

    def __normalize(self, x:torch.Tensor):
        # return (x - x.mean()) / x.std()
        return x.clamp(self.clip_lower, self.clip_upper)

    def process(self, scan_index):
        
        seg_file = f"{self.data_path}/segmentation-{scan_index}.nii"
        vol_file = f"{self.data_path}/volume-{scan_index}.nii"
        
        seg_arr = torch.tensor(read_nii(seg_file))
        vol_arr = read_nii(vol_file)


        # print(seg_arr.shape)
        # print(vol_arr.shape)


        # full_vol_arr = vol_arr.clone()
        # full_seg_arr = seg_arr.clone()


        vol_arr = self.__normalize(vol_arr)

        vol_arr = torch.tensor(vol_arr, dtype=torch.float32)



        slice_list = []

        slice_idx = 0
        while vol_arr.shape[2] >= self.slice_number:
            
            tensor_idx = vol_arr.shape[2] - self.slice_number

            # slice_number = torch.Size(self.slice_number)

            vol_slice = vol_arr[:, :, -self.slice_number:].clone()
            vol_slice:torch.Tensor = resize_volume(vol_slice, self.img_size)
            vol_slice = vol_slice.permute(2, 0, 1).contiguous()

            seg_slice = seg_arr[:, :, -self.slice_number:].clone()
            seg_slice:torch.Tensor = resize_seg(seg_slice, self.img_size)
            seg_slice = seg_slice.permute(2, 0, 1).contiguous()

            # if slice_idx == 0:
            vol_arr = vol_arr[:, :, :-self.slice_number]
            seg_arr = seg_arr[:, :, :-self.slice_number]

            slice_list.append((vol_slice, seg_slice, slice_idx, tensor_idx))

            slice_idx += 1

        
        dp = DataPoint(
            full_vol=None,#full_vol_arr,
            full_seg=None,#full_seg_arr,
            slice_list=slice_list,
            rem_vol=resize_volume(vol_arr.clone(), self.img_size).permute(2, 0, 1).contiguous(),
            rem_seg=resize_seg(seg_arr.clone(), self.img_size).permute(2, 0, 1).contiguous()
        )

        return dp

# turn into tensors
# normalize
# handle differing sizes

    