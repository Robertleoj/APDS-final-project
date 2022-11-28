import numpy as np
from ..load_config import load_config
from glob import glob
from cacher import Cacher



def get_scan_file_paths(data_path):
    return glob(f"{data_path}/*.nii")


class Data:
    def __init__(self, data_config):
        self.data_path = data_config['unzipped_path']
        self.cache_dir = data_config['cache_dir']

        if self.__check_cache():
            return
        else:
            self.__make_cache()

    def get_train(self):
        pass

    def get_val(self):
        pass

    def get_test(self):
        pass

    def __check_cache(self):
        return len(glob(f"{self.cache_dir}/**/*.pickle")) > 0

    def __make_cache(self):
        cacher = Cacher()

        train_indices, val_indices, test_indices = self.__make_split_indices()

        cacher.make_cache(
            train_indices=train_indices, 
            val_indices=val_indices, 
            test_indices=test_indices, 
            data_path=self.data_path
        )

    
    def __make_split_indices(self):
        indices = list(range(0, 131))
        

    

# def get_dsets():
#     """
#     Returns train, val, test datasets
#     """
#     pass



# Need to
# import scans
# open as array (using nib)
# turn into tensors
# normalize
# handle differing sizes
# cache
# train, test, val split
# make pytorch datasets
# make them importable directly

# have files containing indices for splits
# 


