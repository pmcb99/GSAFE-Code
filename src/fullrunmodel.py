# Run the entire ensemble from end to end using this code
import glob
import shutil
import re
import os
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(1,'/DATA/2022-mcm-pmb/src/FeatureExtraction')
sys.path.insert(1,'/DATA/ReworkSultani/')
sys.path.insert(1,'/DATA/2022-mcm-pmb/src/CreateYOLO')
sys.path.insert(1,'/DATA/2022-mcm-pmb/yolov5')
sys.path.insert(1,'/DATA/2022-mcm-pmb/src/utils')
import numpy as np
from yolo_functions import pytorch_yolo, pytorch_yolo_dir
import feat_extract as fe
from run_model import run_detection_model
from utilities import *

def get_list_videos_for_yolo_from_splits(split_directory):
    #Using filenames in split files in split dir, output yolo videos to output folder
    video_id_list = []
    with open(split_directory+'train_anomaly.txt','r') as f:
        split_videos = f.readlines()
        video_id_list += [video.split('/')[-1].replace('\n','') for video in split_videos if video != '\n']
    with open(split_directory+'train_normal.txt','r') as f:
        split_videos = f.readlines()
        video_id_list += [video.split('/')[-1].replace('\n','') for video in split_videos if video != '\n']
    with open(split_directory+'test_anomaly.txt','r') as f:
        split_videos = f.readlines()
        video_id_list += [video.split('/')[-1].split('|')[0] for video in split_videos if video != '\n']
    with open(split_directory+'test_normal.txt','r') as f:
        split_videos = f.readlines()
        video_id_list += [video.split('/')[-1].split(' ')[0] for video in split_videos if video != '\n']
    return video_id_list
import cv2
import shutil

import time
def create_yolo_videos(og_video_file_list):
    yolo_dir = '/DATA/runs_yolo_large/'
    #Local video files given in list, not global full direcs
    for input_video_file in tqdm(og_video_file_list):
        og_video_path = glob.glob('/DATA/OGVideoFolders/**/'+input_video_file,recursive=True)[0]
        video_id = og_video_path.split('/')[-1].split('.')[0]
        output_video_path = og_video_path.replace('OGVideoFolders','runs_yolo_large') 
        if os.path.exists(output_video_path)==False:
            print(f'{output_video_path} doesnt exist!')
            time.sleep(1)
            copy_to_tmp(og_video_path,direc='OGVideoFolders')
        else:
            in_vid = cv2.VideoCapture(og_video_path)
            in_vid_length = int(in_vid.get(cv2.CAP_PROP_FRAME_COUNT))
            out_vid = cv2.VideoCapture(output_video_path)
            out_vid_length = int(out_vid.get(cv2.CAP_PROP_FRAME_COUNT))
            if in_vid_length != out_vid_length:
                print(f'{out_vid_length} frames instead of {in_vid_length} frames...')
                print(f'Incomplete video.. regenerating {output_video_path}.')
                time.sleep(2)
                os.remove(output_video_path)
                copy_to_tmp(og_video_path,direc='OGVideoFolders')
        tmp_folders = glob.glob('/DATA/OGVideoFolders/*_tmp') 
        for f in tmp_folders:
            class_folder = f.split('/')[-1].replace('_tmp','')
            pytorch_yolo_dir(f,yolo_dir,class_folder)
            shutil.rmtree(f)


def main():


    combinations = [['R3D'],['RGB','FLOW','RGB+FLOW'],['K1','K4'],['UCFFull','CQFull','CQ-AnomalySet']]
    combs = ['R3D-RGB-K1','R3D-RGB-K4','R3D-BGMask-K1','R3D-BGMask-K4','R3D-FGMask-K1','R3D-FGMask-K4']
    combs = ['I3D-RGB+FLOW-K4','I3D-RGB-K4','I3D-FLOW-K1','I3D-FLOW-K4','I3D-RGB+FLOW-K1','I3D-RGB-K1']
    # combs = ['R3D-RGB+FLOW-K4','R3D-RGB-K4','R3D-RGB+FLOW-K1','R3D-RGB-K1']
    # combs = ['R3D-FGMask-K4','R3D-RGB-K4']

    output_folder = 'small-cq-i3dvsr3d'
    output_folder = 'ogfull-i3d'
    # output_folder = 'fgmasked'
    # output_folder = 'bgmasked'
    # output_folder = 'small-cq'
    
    log_wndb = False
    # for f in glob.glob('/DATA/ReworkSultani/small-cq-i3dvsr3d/UCF/all_rgbs/*/*.npy'):
    #     os.rename(f,f.replace('.mp4',''))

    video_files_from_splits = get_list_videos_for_yolo_from_splits(f'/DATA/ReworkSultani/{output_folder}/UCF/')
    #IF matrix multiply error, change to 1024, 2048 or 4096 depending on feature dims
    input_dim = 2048
    num_segs = 32
    # video_files_from_splits = glob.glob('/DATA/OGVideoFolders/**/*.mp4',recursive=True)
    # video_files_from_splits = glob.glob('/DATA/OGVideoFolders/**/*.mp4',recursive=True)
    # video_files_from_splits = [v.split('/')[-1] for v in video_files_from_splits]
    # # print(video_files_from_splits)
    # if 'og' not in output_folder:
    #     create_yolo_videos(video_files_from_splits)

    #Run the model on the previously generated features
    dataset_path = f'/DATA/ReworkSultani/{output_folder}/UCF/'
    for c in combs:
        run_detection_model(dataset_path,log_wndb=log_wndb,num_segs=num_segs,input_dim=input_dim,setting=c)
    # run_detection_model(dataset_path,log_wndb=log_wndb,num_segs=num_segs,input_dim=input_dim,setting='')

if __name__ == "__main__":
    main()


    # for f in glob.glob('/DATA/ReworkSultani/pvmini-r3d/UCF/all_flows/**/*.npy',recursive=True):
    #     os.rename(f,f.replace('.mp4',''))


