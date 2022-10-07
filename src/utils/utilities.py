import os
import shutil
import glob
import re
import numpy as np

def copy_to_tmp(video_path,direc):
    orig_folder = video_path.split('/')[-2]
    dest = f'/DATA/{direc}/{orig_folder}_tmp/'+video_path.split('/')[-1]
    os.makedirs(f'/DATA/{direc}/{orig_folder}_tmp/',exist_ok=True)
    shutil.copyfile(video_path,dest)

def move_back(video_path,direc):
    class_folder = video_path.split('/')[-1][:3]
    dest = glob.glob(f'/DATA/{direc}/{class_folder}*/')[0]+video_path.split('/')[-1]
    os.rename(video_path,dest)

def get_class_name_from_video(video_name):
    if 'Normal' in video_name:
        return 'Normal_Videos_event'
    else:
        class_name = video_name.split('_')[0]
        return re.sub('\d+','',class_name)

def get_video_name(video_path):
    return video_path.split('/')[-1].split('.')[0]


def convert_txt_to_numpy(f):
  data = np.loadtxt(f)
  np.save(f.replace('txt','npy'),data)
  os.remove(f)

def removedotmp4(f):
    os.rename(f,f.replace('.mp4',''))

# fs = glob.glob('/DATA/labels/Normal_Videos_event/la*/*.txt',recursive=True)
# for f in fs:
#     convert_txt_to_numpy(f)

