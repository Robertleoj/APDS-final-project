import nibabel as nib
import numpy as np
from collections import namedtuple
import shutil
import os

DataPoint = namedtuple('DataPoint', (
    'full_vol', 'full_seg',
    'slice_list', 'rem_vol', 'rem_seg'
))

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.array(array)
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
