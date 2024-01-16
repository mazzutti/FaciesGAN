from turtle import shape
import torch
from torch.utils.data import Dataset
import datasets.helper as helper
import numpy as np
 
class GslibDataset(Dataset):
    def __init__(self,transform,gslib_file):
        self.gslib_file = gslib_file
        self.transforms = transform
    def __getitem__(self, index):  
        gslib=helper.OpenGslibFile(self.gslib_file) 
        data = self.transforms(gslib)
        return data
    def __len__(self):
        # 返回长度
        return 1

class GaussianDataset(Dataset):
    def __init__(self,gslib_file):
        self.gslib_file = gslib_file
    def __getitem__(self, index):  
        gslib=helper.OpenGslibFile(self.gslib_file) 
        data=torch.from_numpy(np.resize(shape=(1,250,250))) 
        return data
    def __len__(self):
        # 返回长度
        return 1