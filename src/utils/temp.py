
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

def get_concat_npy(path,data_list,idx,is_train):
    if is_train==1:
        rgb_npy = np.load(os.path.join(path+'all_rgbs', data_list[idx][:-1]+'.npy'))
        flow_npy = np.load(os.path.join(path+'all_flows', data_list[idx][:-1]+'.npy'))
        concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
    else:
        name, frames, gts = data_list[idx].split(' ')[0], int(data_list[idx].split(' ')[1]), int(data_list[idx].split(' ')[2][:-1])
        rgb_npy = np.load(os.path.join(path+'all_rgbs', name + '.npy'))
        flow_npy = np.load(os.path.join(path+'all_flows', name + '.npy'))
    return concat_npy



root = '/DATA/ReworkSultani/og-mini/UCF/'
class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path=root):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            anom_list = os.path.join(path, 'train_anomaly.txt')
            anom_file_len = 0
            with open(anom_list, 'r') as f:
                anom_file_len = len(f.readlines())
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
                self.data_list = self.data_list[:anom_file_len]

        else:
            anom_list = os.path.join(path, 'test_anomaly.txt')
            anom_file_len = 0
            with open(anom_list, 'r') as f:
                anom_file_len = len(f.readlines())
            data_list = os.path.join(path, 'test_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:anom_file_len]
    def __len__(self):
        return len(self.data_list)

    def which_dataset():
        return root.split('/')[-2]




    def __getitem__(self, idx):
        if self.is_train == 1:
            concat_npy = get_concat_npy(self.path,self.data_list,idx,self.is_train)
            return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            concat_npy = get_concat_npy(self.path,self.data_list,idx,self.is_train)
            return concat_npy, gts, frames

class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path=root):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        if self.is_train == 1:
            concat_npy = get_concat_npy(self.path,self.data_list,idx,self.is_train)
            return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            concat_npy = get_concat_npy(self.path,self.data_list,idx,self.is_train)
            return concat_npy, gts, frames

if __name__ == '__main__':
    loader2 = Normal_Loader(is_train=0)
    print(len(loader2))
    #print(loader[1], loader2[1])
