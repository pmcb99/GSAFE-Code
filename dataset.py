from audioop import avg
from sklearn.metrics import euclidean_distances
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from sklearn.preprocessing import normalize


def max_min_normalise(x,maximum,minimum):
    new = np.subtract(x,minimum)
    new = np.divide(new,(maximum-minimum))
    return new

def get_concat_npy(path,data_list,idx,is_train,is_normal,setting):
    vid_name = data_list[idx]
    vid_name = vid_name.replace('.mp4','',1) 
    append_string = ''
    if is_train:
        rgb_npy = np.load(os.path.join(path+'all_rgbs', vid_name[:-1]+'.npy'))
        flow_npy = np.load(os.path.join(path+'all_flows', vid_name[:-1]+'.npy'))
        fg_npy = np.load(os.path.join('/DATA/ReworkSultani/fgmasked/UCF/all_rgbs', vid_name[:-1]+'_fgmask.npy'))
        bg_npy = np.load(os.path.join('/DATA/ReworkSultani/bgmasked/UCF/all_rgbs', vid_name[:-1]+'_bgmask.npy'))

        frame_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/frame_count', vid_name[:-1]+'.npy'))
        person_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/person_count', vid_name[:-1]+'.npy'))
        bb_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/bb_areas', vid_name[:-1]+'.npy'))
        interhuman_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/interhuman_distance', vid_name[:-1]+'.npy'))
        avg_interhuman_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/avg_interhuman_distance', vid_name[:-1]+'.npy'))
    else:
        if is_normal:
            name = vid_name.split(' ')[0]
            rgb_npy = np.load(os.path.join(path+'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(path+'all_flows', name + '.npy'))
            fg_npy = np.load(os.path.join('/DATA/ReworkSultani/fgmasked/UCF/all_rgbs', name + '_fgmask.npy'))
            bg_npy = np.load(os.path.join('/DATA/ReworkSultani/bgmasked/UCF/all_rgbs', name + '_bgmask.npy'))
            frame_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/frame_count', name + '.npy'))

            person_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/person_count', name + '.npy'))
            bb_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/bb_areas', name + '.npy'))
            interhuman_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/interhuman_distance', name + '.npy'))
            avg_interhuman_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/avg_interhuman_distance', name + '.npy'))
        else:
            name = vid_name.split('|')[0]
            rgb_npy = np.load(os.path.join(path+'all_rgbs', name +'.npy'))
            flow_npy = np.load(os.path.join(path+'all_flows', name + '.npy'))
            fg_npy = np.load(os.path.join('/DATA/ReworkSultani/fgmasked/UCF/all_rgbs', name + '_fgmask.npy'))
            bg_npy = np.load(os.path.join('/DATA/ReworkSultani/bgmasked/UCF/all_rgbs', name + '_bgmask.npy'))
            # frame_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/frame_count', name + '.npy'))

            person_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/person_count', name + '.npy'))
            bb_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/bb_areas', name + '.npy'))
            interhuman_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/interhuman_distance', name + '.npy'))
            avg_interhuman_npy = np.load(os.path.join('/DATA/ReworkSultani/small-cq/UCF/avg_interhuman_distance', name + '.npy'))

    flow_npy = np.concatenate([flow_npy, flow_npy], axis=1) #Only for R3D so dimensions match
    # concat_npy = np.concatenate([rgb_npy, rgb_npy], axis=1)
    if 'RGB-' in setting:
        concat_npy = np.concatenate([rgb_npy,rgb_npy],axis=1)
    elif 'RGB+FLOW' in setting:
        concat_npy = np.concatenate([rgb_npy,flow_npy],axis=1)
    elif '-FLOW-' in setting:
        concat_npy = np.concatenate([flow_npy,flow_npy],axis=1)
    elif 'HEADCOUNT' in setting:
        concat_npy = np.concatenate([person_npy,person_npy],axis=1)
        concat_npy = max_min_normalise(concat_npy,np.max(concat_npy),np.min(concat_npy))
        concat_npy = np.nan_to_num(concat_npy) 
    elif 'BBOX' in setting:
        concat_npy = np.concatenate([bb_npy,bb_npy],axis=1)
        concat_npy = max_min_normalise(concat_npy,np.max(concat_npy),np.min(concat_npy))
        concat_npy = np.nan_to_num(concat_npy) 
    elif 'TOTALDIST' in setting:
        concat_npy = np.concatenate([interhuman_npy,interhuman_npy],axis=1)
        concat_npy = max_min_normalise(concat_npy,np.max(concat_npy),np.min(concat_npy))
        concat_npy = np.nan_to_num(concat_npy) 
    elif 'RANDOM' in setting:
        concat_npy = np.random.rand(32,4096)
    elif 'ZERO' in setting:
        concat_npy = np.zeros((32,4096),dtype=np.float32)
    elif 'AVGDIST' in setting:
        person_npy = np.concatenate([person_npy,person_npy],axis=1)
        interhuman_npy = np.concatenate([interhuman_npy,interhuman_npy],axis=1)
        avgdist_npy = np.divide(interhuman_npy+1000,person_npy+1000)
        concat_npy = max_min_normalise(avgdist_npy,np.max(avgdist_npy),np.min(avgdist_npy))
        concat_npy = np.nan_to_num(concat_npy) 
    elif 'AVGBBOX' in setting:
        person_npy = np.concatenate([person_npy,person_npy],axis=1)
        bb_npy = np.concatenate([bb_npy,bb_npy],axis=1)
        avgbbox_npy = np.divide(bb_npy+1000,person_npy+1000)
        concat_npy = max_min_normalise(avgdist_npy,np.max(avgdist_npy),np.min(avgdist_npy))
        concat_npy = np.nan_to_num(concat_npy) 
    elif 'SEE' in setting:
        avgdist_npy = np.divide(interhuman_npy+1000,person_npy+1000)
        avgbbox_npy = np.divide(bb_npy+1000,person_npy+1000)
        avgcoeff_npy = np.divide(avgbbox_npy,avgdist_npy)
        concat_npy = max_min_normalise(avgcoeff_npy,np.max(avgdist_npy),np.min(avgdist_npy))
        concat_npy = np.nan_to_num(concat_npy) 
        concat_npy = np.concatenate([concat_npy,concat_npy],axis=1)
    elif 'SEE+RGB' in setting:
        avgdist_npy = np.divide(interhuman_npy+1000,person_npy+1000)
        avgbbox_npy = np.divide(bb_npy+1000,person_npy+1000)
        avgcoeff_npy = np.divide(avgbbox_npy,avgdist_npy)
        concat_npy = max_min_normalise(avgcoeff_npy,np.max(avgdist_npy),np.min(avgdist_npy))
        concat_npy = np.nan_to_num(concat_npy) 
        concat_npy = np.concatenate([concat_npy,rgb_npy],axis=1)
    elif 'EUCLID' in setting:
        euclid_npy = np.divide(bb_npy+1000,interhuman_npy+1000)
        concat_npy = max_min_normalise(euclid_npy,np.max(euclid_npy),np.min(euclid_npy))
        concat_npy = np.nan_to_num(concat_npy) 
        concat_npy = np.concatenate([concat_npy,concat_npy],axis=1)
    elif 'TRIO' in setting:
        nonanperson_npy = np.nan_to_num(person_npy+1000) 
        nonanbb_npy = np.nan_to_num(bb_npy+1000) 
        nonaninterhuman_npy = np.nan_to_num(interhuman_npy+1000) 
        normperson_npy = max_min_normalise(nonanperson_npy,np.max(nonanperson_npy),np.min(nonanperson_npy))
        normbb_npy = max_min_normalise(nonanbb_npy,np.max(nonanbb_npy),np.min(nonanbb_npy))
        norminterhuman_npy = max_min_normalise(nonaninterhuman_npy,np.max(nonaninterhuman_npy),np.min(nonaninterhuman_npy))
        trio_npy = np.concatenate([normperson_npy,norminterhuman_npy,normbb_npy],axis=1)
        normtrio = np.nan_to_num(trio_npy) 
        concat_npy = normtrio 
    elif 'FG' in setting:
        concat_npy = fg_npy
        concat_npy = np.concatenate([concat_npy,concat_npy],axis=1)
    elif 'BG' in setting:
        concat_npy = bg_npy
        concat_npy = np.concatenate([concat_npy,concat_npy],axis=1)
    # else:
    #     concat_npy = rgb_npy
    # concat_npy = np.divide(concat_npy,976504) for frames features
    concat_npy = concat_npy.astype('float32')
    return concat_npy



root = '/DATA/ReworkSultani/og-mini/UCF/'
class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path=root,setting='',classname=''):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.setting = setting
        self.classname = classname
        if self.is_train == 1:
            anom_list = os.path.join(path, 'train_anomaly.txt')
            anom_file_len = 0
            with open(anom_list, 'r') as f:
                anom_list = f.readlines()
                # anom_list = [v for v in anom_list if classname in v]
                anom_file_len = len(anom_list)
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
                self.data_list = self.data_list[:anom_file_len]

        else:
            anom_list = os.path.join(path, 'test_anomaly.txt')
            anom_file_len = 0
            with open(anom_list, 'r') as f:
                anom_list = f.readlines()
                anom_list = [v for v in anom_list if classname in v]
                anom_file_len = len(anom_list)
            data_list = os.path.join(path, 'train_normal.txt')
            data_list = os.path.join(path, 'test_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:anom_file_len]
    def __len__(self):
        return len(self.data_list)

    def which_dataset(self):
        return 
    def which_dataset():
        return root.split('/')[-2]

    def __getitem__(self, idx):
        if self.is_train == 1:
            concat_npy = get_concat_npy(self.path,self.data_list,idx,self.is_train,True,self.setting)
            return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            concat_npy = get_concat_npy(self.path,self.data_list,idx,self.is_train,True,self.setting)
            return concat_npy, gts, frames

class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path=root,setting='',classname=''):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.setting = setting
        self.classname = classname
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
                # self.data_list = [v for v in self.data_list if classname in v]
        else:
            data_list = os.path.join(path, 'test_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
                self.data_list = [v for v in self.data_list if classname in v]

    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        if self.is_train == 1:
            concat_npy = get_concat_npy(self.path,self.data_list,idx,self.is_train,False,self.setting)
            return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            concat_npy = get_concat_npy(self.path,self.data_list,idx,self.is_train,False,self.setting)
            return concat_npy, gts, frames



class CombinedLoader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path=root):
        super(CombinedLoader, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            anom_list = os.path.join(path, 'train_anomaly.txt')
            norm_list = os.path.join(path, 'train_normal.txt')
            anom_file_len = 0
            with open(anom_list, 'r') as f:
                anom_file_len = len(f.readlines())
            normal_list = os.path.join(path, 'train_normal.txt')
            with open(normal_list, 'r') as f:
                self.normal_list = f.readlines()
                self.normal_list = self.normal_list[:anom_file_len]

            anom_list = os.path.join(path, 'train_anomaly.txt')
            with open(anom_list, 'r') as f:
                self.anom_list = f.readlines()
                self.anom_list = self.anom_list[:anom_file_len]
            
            self.data_list = self.anom_list+self.normal_list
            print(len(self.data_list))

        else:
            anom_list = os.path.join(path, 'test_anomaly.txt')
            anom_file_len = 0
            with open(anom_list, 'r') as f:
                anom_file_len = len(f.readlines())
            normal_list = os.path.join(path, 'test_normal.txt')
            anom_list = os.path.join(path, 'test_normal.txt')
            with open(norm_list, 'r') as f:
                self.normal_list = f.readlines()
                random.shuffle(self.normal_list)
                self.normal_list = self.normal_list[:anom_file_len]
            with open(anom_list, 'r') as f:
                self.anom_list = f.readlines()
                random.shuffle(self.anom_list)
                self.anom_list = self.anom_list[:anom_file_len]
            self.data_list = self.anom_list+self.normal_list
    def __len__(self):
        return len(self.data_list)

    def which_dataset(self):
        return 
    def which_dataset():
        return root.split('/')[-2]

    def __getitem__(self, idx):
        if self.is_train == 1:
            concat_npy = get_concat_npy(self.path,self.data_list,idx,self.is_train,True)
            return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            concat_npy = get_concat_npy(self.path,self.data_list,idx,self.is_train,True)
            return concat_npy, gts, frames






if __name__ == '__main__':
    loader2 = Normal_Loader(is_train=0)
    print(len(loader2))
    #print(loader[1], loader2[1])


