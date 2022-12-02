import numpy as np
# from ..load_config import load_config
from glob import glob
from torch.utils.data import Dataset, DataLoader
from .cacher import Cacher
import torch
import random
import json
import os



def get_scan_file_paths(data_path):
    return glob(f"{data_path}/*.nii")

class Dataset_2p5D(Dataset):
    def __init__(self, cache_path):

        seg_fpaths = sorted(glob(f"{cache_path}/seg*[!rem].pickle"))
        vol_fpaths = sorted(glob(f"{cache_path}/vol*[!rem].pickle"))

        self.pairs = list(zip(vol_fpaths, seg_fpaths))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vol_path, seg_path = self.pairs[idx]

        vol = torch.load(vol_path)
        seg = torch.load(seg_path)

        return vol, seg


class Data:
    def __init__(self, config, nuke_cache=False):
        self.config = config

        if  not nuke_cache and self.__check_cache():
            return
        else:
            self.__make_cache()

    def get_train(self):
        return Dataset_2p5D(
            f"{self.config['cache_dir']}/train"
        )

    def get_val(self):
        return Dataset_2p5D(
            f"{self.config['cache_dir']}/val"
        )

    def get_test(self):
        return Dataset_2p5D(
            f"{self.config['cache_dir']}/test"
        )

    def __check_cache(self):
        cache_dir = self.config['cache_dir']
        return len(glob(f"{cache_dir}/**/*.pickle")) > 0

    def __make_cache(self):
        train_indices, val_indices, test_indices = self.__get_split_indices()

        cacher = Cacher(
            config=self.config,
            train_indices=train_indices, 
            val_indices=val_indices, 
            test_indices=test_indices, 
        )

        cacher.nuke()

        cacher.make_cache()

    def __get_split_indices(self):
        split_path = self.config['split_path']
        
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                indices_dict = json.load(f)

                return (
                    indices_dict['train'], 
                    indices_dict['val'], 
                    indices_dict['test']
                )
        else:
            return self.__make_split_indices()
        
    
    def __make_split_indices(self):
        random.seed()
        indices = list(range(0, 131))
        random.shuffle(indices)
        
        split_config = self.config['split']

        train_split = split_config['train']
        val_split = split_config['val']
        test_split = split_config['test']
        
        train_indices = indices[:train_split]
        val_indices = indices[train_split:train_split+val_split]
        test_indices = indices[train_split + val_split:]

        self.__save_indices(
            train_indices,
            val_indices,
            test_indices
        )

        return (train_indices, val_indices, test_indices)

    
    def __save_indices(
        self,
        train_indices,
        val_indices,
        test_indices
    ):
        ind_dict = {
            'test': test_indices,
            'val': val_indices,
            'train': train_indices
        }

        with open(self.config['split_path'], 'w') as f:
            json.dump(ind_dict, f, indent=4)







        


    

# def get_dsets():
#     """
#     Returns train, val, test datasets
#     """
#     pass



# Need to
# import scans

# cache
# train, test, val split
# make pytorch datasets
# make them importable directly

# have files containing indices for splits
# 

