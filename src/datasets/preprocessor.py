from utils import read_nii


class Preprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def process(self, scan_index):
        
        seg_file = f"{self.data_path}/segmentation-{scan_index}.nii"
        vol_file = f"{self.data_path}/volume-{scan_index}.nii"
        
        seg_arr = read_nii(seg_file)
        vol_arr = read_nii(vol_file)

        return vol_arr, seg_arr
        

    