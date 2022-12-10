import numpy as np
# from ..load_config import load_config
from glob import glob
from torch.utils.data import IterableDataset, DataLoader
from .cacher import Cacher
import torch
import random
import json
import os



def get_scan_file_paths(data_path):
    return glob(f"{data_path}/*.nii")

class Dataset_2p5D(IterableDataset):
    def __init__(self, cache_path, n_slices):



        self.either_side = n_slices // 2

        vol_paths = glob(f"{cache_path}/vol*.pickle")

        indices = set([
            int(p.split('_')[-2]) 
            for p in vol_paths
        ])

        self.tuples = {i: [] for i in indices}

        for idx in indices:
            paths = glob(f"{cache_path}/vol_{idx}_*.pickle")
            slice_indices = [
                int(p.split('_')[-1].split('.')[0]) 
                for p in paths
            ]

            slice_indices = sorted(slice_indices)[self.either_side:-self.either_side]

            self.tuples[idx].extend([
                (idx, slice_idx) 
                for slice_idx in slice_indices
            ])

        self.cache_path = cache_path

        self.tuples = [v for v in self.tuples.values()]


    def __iter__(self):

        while True:
            lst = random.choice(self.tuples)

            scan_idx, slice_idx = random.choice(lst)

            vol_paths = [
                f"{self.cache_path}/vol_{scan_idx}_{slice_idx + i}.pickle"
                for i in range(-self.either_side, self.either_side + 1)
            ]

            seg_path = f"{self.cache_path}/seg_{scan_idx}_{slice_idx}.pickle"

            vol = torch.concat(
                [torch.load(path) for path in vol_paths], dim=0
            )

            seg = torch.load(seg_path)

            yield vol.to(dtype=torch.float), seg.squeeze(0).to(dtype=torch.long)


class Data:
    def __init__(self, config, nuke_cache=False):
        self.config = config

        if  not nuke_cache and self.__check_cache():
            return
        else:
            self.__make_cache()

    def get_train(self):
        return Dataset_2p5D(
            f"{self.config['cache_dir']}/train", self.config['n_slices']
        )

    def get_val(self):
        return Dataset_2p5D(
            f"{self.config['cache_dir']}/val", self.config['n_slices']
        )

    def get_test(self):
        return Dataset_2p5D(
            f"{self.config['cache_dir']}/test", self.config['n_slices']
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

