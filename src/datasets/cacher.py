from .preprocessor import Preprocessor_2p5D
import os
import torch
from enum import Enum
from .utils import DataPoint
from tqdm import tqdm
import shutil
from .utils import nuke


class WhichSplit(Enum):
    train=1
    val=2
    test=3

    @classmethod
    def get_name(cls, instance):

        match instance:
            case WhichSplit.train:
                return 'train'
            case WhichSplit.val:
                return 'val'
            case WhichSplit.test:
                return 'test'

       

class Cacher:
    def __init__(self, *,
        config,
        train_indices, 
        val_indices, 
        test_indices, 
    ):
        
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.data_path = config['unzipped_path']
        self.slice_number = config['slice_number']
        self.cache_path = config['cache_dir']
        self.config = config
        

    def __get_fpath(self, *,
        which:WhichSplit, slice_idx, sample_idx,  is_seg
    ):

        match which:
            case WhichSplit.train:
                postfix = 'train'
            case WhichSplit.val:
                postfix = 'val'
            case WhichSplit.test:
                postfix = 'test'

        folder_path = self.cache_path + '/' + postfix 

        file_name = f"{'seg' if is_seg else 'rem' }_{sample_idx}_{slice_idx}.pickle"

        return folder_path + '/' + file_name

    def __save_sample(self, dp:DataPoint, which:WhichSplit, sample_idx: int):
        
        slice_list = reversed(dp.slice_list)
        
        for vol_slice, seg_slice, slice_index, tensor_idx in slice_list:

            seg_fpath = self.__get_fpath(
                which=which,
                slice_idx=slice_index,
                sample_idx=sample_idx,
                is_seg=True
            )

            vol_fpath = self.__get_fpath(
                which=which,
                slice_idx=slice_index,
                sample_idx=sample_idx,
                is_seg=False
            )

            torch.save(obj=vol_slice, f=vol_fpath)
            torch.save(obj=seg_slice, f=seg_fpath)

        segrem_fpath = self.__get_fpath(
            which=which,
            slice_idx='rem',
            sample_idx=sample_idx,
            is_seg=True
        )

        volrem_fpath = self.__get_fpath(
            which=which,
            slice_idx='rem',
            sample_idx=sample_idx,
            is_seg=True
        )

        torch.save(obj=dp.rem_seg, f=segrem_fpath)
        torch.save(obj=dp.rem_vol, f=volrem_fpath)


    def nuke(self):
        if input("deleting " + self.cache_path) == 'y':
            nuke(self.cache_path)
    
    def make_cache(self):
        # loop through train, val and test indices
        # for each, make preprocessor process the scan
        # put them into their corresponding folder

        preprocessor = Preprocessor_2p5D(self.config) # some args

        cache_dir = self.config['cache_dir']

        os.makedirs(f"{cache_dir}/train", exist_ok=True)
        os.makedirs(f"{cache_dir}/val", exist_ok=True)
        os.makedirs(f"{cache_dir}/test", exist_ok=True)

        for indices, kind in (
            (self.train_indices, WhichSplit.train),
            (self.val_indices, WhichSplit.val),
            (self.test_indices, WhichSplit.test)
        ):
            print(WhichSplit.get_name(kind))



            for i in tqdm(indices):
                dp = preprocessor.process(i)
                self.__save_sample(dp, kind, i)

            print()


