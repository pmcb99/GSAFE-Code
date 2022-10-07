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
    vid_dir = '/DATA/Matching/Originals/'
    yolo_dir = '/DATA/runs/'
    #Local video files given in list, not global full direcs
    for input_video_file in tqdm(og_video_file_list):
        og_video_path = glob.glob(f'{vid_dir}**/'+input_video_file,recursive=True)[0]
        print(og_video_path)
        video_id = og_video_path.split('/')[-1].split('.')[0]
        output_video_path = og_video_path.replace(f'{vid_dir}','runs') 
        if os.path.exists(output_video_path)==False:
            print(f'{output_video_path} doesnt exist!')
            time.sleep(1)
            copy_to_tmp(og_video_path,direc='Matching')
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
                copy_to_tmp(og_video_path,direc='Matching')
        tmp_folders = glob.glob('/DATA/Matching/*_tmp') 
        for f in tmp_folders:
            class_folder = f.split('/')[-1].replace('_tmp','')
            pytorch_yolo_dir(f,yolo_dir,class_folder)
            shutil.rmtree(f)
def main():
    video_files_from_splits = glob.glob('/DATA/Matching/Originals/**/*.mp4',recursive=True)
    print(video_files_from_splits)
    video_files_from_splits = [v.split('/')[-1] for v in video_files_from_splits]

    create_yolo_videos(video_files_from_splits)
if __name__ == "__main__":
    main()


    # for f in glob.glob('/DATA/ReworkSultani/pvmini-r3d/UCF/all_flows/**/*.npy',recursive=True):
    #     os.rename(f,f.replace('.mp4',''))


