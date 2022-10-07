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
from yolo_functions import pytorch_yolo, pytorch_yolo_dir
import feat_extract as fe
from run_model import run_detection_model

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

def copy_to_tmp(video_path):
    orig_folder = video_path.split('/')[-2]
    dest = f'/DATA/OGVideoFolders/{orig_folder}_tmp/'+video_path.split('/')[-1]
    os.makedirs(f'/DATA/OGVideoFolders/{orig_folder}_tmp/',exist_ok=True)
    shutil.copyfile(video_path,dest)

def move_back(video_path):
    class_folder = video_path.split('/')[-1][:3]
    dest = glob.glob(f'/DATA/OGVideoFolders/{class_folder}*/')[0]+video_path.split('/')[-1]
    os.rename(video_path,dest)
import time
def create_yolo_videos(og_video_file_list):
    yolo_dir = '/DATA/runs_yolo/'
    #Local video files given in list, not global full direcs
    for input_video_file in tqdm(og_video_file_list):
        og_video_path = glob.glob('/DATA/OGVideoFolders/**/'+input_video_file,recursive=True)[0]
        video_id = og_video_path.split('/')[-1].split('.')[0]
        output_video_path = og_video_path.replace('OGVideoFolders','runs_yolo') 
        if os.path.exists(output_video_path)==False:
            print(f'{output_video_path} doesnt exist!')
            time.sleep(1)
            copy_to_tmp(og_video_path)
        else:
            in_vid = cv2.VideoCapture(og_video_path)
            in_vid_length = int(in_vid.get(cv2.CAP_PROP_FRAME_COUNT))
            out_vid = cv2.VideoCapture(output_video_path)
            out_vid_length = int(out_vid.get(cv2.CAP_PROP_FRAME_COUNT))
            if in_vid_length != out_vid_length:
                print(f'{out_vid_length} frames instead of {in_vid_length} frames...')
                print(f'Incomplete video.. regenerating {output_video_path}.')
                time.sleep(3)
                os.remove(output_video_path)
                copy_to_tmp(og_video_path)
        tmp_folders = glob.glob('/DATA/OGVideoFolders/*_tmp') 
        for f in tmp_folders:
            class_folder = f.split('/')[-1].replace('_tmp','')
            pytorch_yolo_dir(f,yolo_dir,class_folder)
            shutil.rmtree(f)

def fast_create_yolo_videos(og_video_file_list):
    pass

def create_features(video_files_from_splits, num_segments, output_folder, use_gpu):
    print(video_files_from_splits)
    process_list = []
    for f in video_files_from_splits:
        if 'og' in output_folder:
            input_video_path = glob.glob(f'/DATA/OGVideoFolders/{f[:2]}*/'+f,recursive=True)[0]
            print(f'OG so input coming from {input_video_path}')
        else:
            input_video_path = glob.glob(f'/DATA/runs_yolo/{f[:2]}*/'+f,recursive=True)[0]
            print(f'PV so input coming from {input_video_path}')
        video_id = input_video_path.split('/')[-1]
        if 'Normal' in input_video_path:
            output_dir = f'/DATA/ReworkSultani/{output_folder}/UCF/all_rgbs/'+'Normal_Videos_event'
        else:
            output_dir = f'/DATA/ReworkSultani/{output_folder}/UCF/all_rgbs/'+re.sub('\d+','',input_video_path.split('/')[-1].split('_')[0])
        print(output_dir + '/'+ f+'.npy')
        if not os.path.exists(output_dir + '/'+ f+'.npy'):
            print(input_video_path)
            process_list.append(input_video_path)
    print(f'Processing {len(process_list)} videos')
    fe.feature_extraction_function(process_list,output_folder=output_folder,use_gpu=use_gpu,num_segments=num_segments)

def create_priv_features():



def main():

    # output_folder = 'ogmini-gluon32segs'
    output_folder = 'pv-oneshot'

    video_files_from_splits = get_list_videos_for_yolo_from_splits(f'/DATA/ReworkSultani/{output_folder}/UCF/')
    print(video_files_from_splits)
    num_segs = 16 if '16' in output_folder else 32
    input_dim = 4096 if 'gluon' in output_folder else 2048
    #IF matrix multiply error, change to 1024, 2048 or 4096 depending on feature dims
    input_dim = 2048
    video_files_from_splits = glob.glob('/DATA/OGVideoFolders/**/*.mp4',recursive=True)
    video_files_from_splits = [v.split('/')[-1] for v in video_files_from_splits]
            

    if 'og' not in output_folder:
        create_yolo_videos(video_files_from_splits)



    #MAKE SURE TO USE THE ENV
    # create_features(video_files_from_splits=video_files_from_splits,num_segments=num_segs,output_folder=output_folder,use_gpu=False)

    # process_list = glob.glob('/DATA/OGVideoFolders/**/*.mp4',recursive=True)
    # fe.feature_extraction_function(process_list,output_folder=output_folder,use_gpu=False,num_segments=16)

    #Run the model on the previously generated features
    dataset_path = f'/DATA/ReworkSultani/{output_folder}/UCF/'

    # run_detection_model(dataset_path,log_wndb=False,num_segs=num_segs,input_dim=input_dim)

if __name__ == "__main__":
    main()




