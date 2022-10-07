#Make features using extracted yolo data
import glob
import shutil
import re
import os
from tqdm import tqdm
import pandas as pd
import sys
from scipy import signal
from scipy.spatial import distance
from pathlib import Path
import cv2
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
    # with open(split_directory+'train_anomaly.txt','r') as f:
    #     split_videos = f.readlines()
    #     video_id_list += [video.split('/')[-1].replace('\n','') for video in split_videos if video != '\n']
    # with open(split_directory+'train_normal.txt','r') as f:
    #     split_videos = f.readlines()
    #     video_id_list += [video.split('/')[-1].replace('\n','') for video in split_videos if video != '\n']
    # with open(split_directory+'test_anomaly.txt','r') as f:
    #     split_videos = f.readlines()
    #     video_id_list += [video.split('/')[-1].split('|')[0] for video in split_videos if video != '\n']
    with open(split_directory+'left.txt','r') as f:
        split_videos = f.readlines()
        video_id_list += [video.split('/')[-1].split(' ')[0] for video in split_videos if video != '\n']
    return video_id_list

def make_frame_features(video_list,output_folder):
    '''Features with number of frames in each video'''
    frame_dict = np.load('/DATA/ReworkSultani/small-cq/UCF/frames.pkl',allow_pickle=True)
    max_frames = max(frame_dict.values())
    for video in video_list:
        video_name = video.split('.')[0]
        class_name = get_class_name_from_video(video_name)
        try:
            total_frames = frame_dict[video_name]
        except KeyError as k:
            video_name = video_name.replace('s_','s')
        total_frames = frame_dict[video_name]
        video_name = video.split('.')[0]
        feature = np.zeros([32,2048],dtype=np.float32)
        feature[:,:]=np.float(total_frames/max_frames)
        os.makedirs(f'/DATA/ReworkSultani/{output_folder}/UCF/frame_count/{class_name}',exist_ok=True)
        feature_output = f'/DATA/ReworkSultani/{output_folder}/UCF/frame_count/{class_name}/{video_name}.npy'
        np.save(feature_output,feature)

def get_frames(video_name):
    frame_dict = np.load('/DATA/ReworkSultani/small-cq/UCF/frames.pkl',allow_pickle=True)
    try:
        total_frames = frame_dict[video_name]
    except KeyError as k:
        video_name = video_name.replace('s_','s')
    total_frames = frame_dict[video_name]
    return total_frames


def make_person_features(video_name_list):
    '''Features with time evolution of number of persons per frame'''
    frame_dict = np.load('/DATA/ReworkSultani/small-cq/UCF/frames.pkl',allow_pickle=True)
    video_name_list = [v.split('.')[0] for v in video_name_list]

    for vid in video_name_list:
        alreadydone = ' '.join(glob.glob('/DATA/ReworkSultani/small-cq/UCF/person_count/**/*.npy',recursive=True))
        if vid not in alreadydone:
            print(vid)


    total_frames = frame_dict
    # for vid in video_name_list:
    #     feature_generator(vid)
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(feature_generator,video_name_list)
    return results

def save_feature(feature_npy,output_folder,feature_folder,video_name):
    class_name = get_class_name_from_video(video_name)
    os.makedirs(f'/DATA/ReworkSultani/{output_folder}/UCF/{feature_folder}/{class_name}',exist_ok=True)
    feature_output = f'/DATA/ReworkSultani/{output_folder}/UCF/{feature_folder}/{class_name}/{video_name}.npy'
    np.save(feature_output,feature_npy)
    print(f'Shape: {feature_npy.shape} for {video_name}')

from tqdm import tqdm

def feature_generator(video_name,output_folder='small-cq'):
    total_frames = get_frames(video_name)
    class_name = get_class_name_from_video(video_name)
    frames_direc = f'/DATA/labels/{class_name}/**/{video_name}*'
    # print(f'/DATA/ReworkSultani/small-cq/UCF/avg_interhuman_distance/{video_name}.npy')
    if os.path.exists(f'/DATA/ReworkSultani/small-cq/UCF/avg_interhuman_distance/{class_name}/{video_name}.npy'): 
        return 
    else:
        print(f'Working on {video_name}')
        people_count_per_frame_array = np.zeros(total_frames,dtype=np.float32)
        total_distance_between_points_array = np.zeros(total_frames,dtype=np.float32)
        sum_of_people_areas_array = np.zeros(total_frames,dtype=np.float32)
        for i in range(total_frames-1):
            yolo_frame = glob.glob(f'/DATA/labels/{class_name}/lab*/{video_name}_{i+1}.*.npy',recursive=True)
            if yolo_frame == []:
                people_count_per_frame_array[i] = 0.0
                sum_of_people_areas_array[i] = 0.0
                total_distance_between_points_array[i] = 0.0
            else:
                yolo_data = np.load(yolo_frame[0])
                persons_data = yolo_data[yolo_data[0]==0] if yolo_data.ndim == 1 else yolo_data[yolo_data[:,0]==0]
                people_count_per_frame_array[i] = np.float32(get_person_count(persons_data))
                sum_of_people_areas_array[i] = get_total_area(persons_data)
                total_distance_between_points_array[i] = get_total_distance(persons_data)

    people_count_per_frame_array = signal.resample(people_count_per_frame_array,32*2048)
    sum_of_people_areas_array = signal.resample(sum_of_people_areas_array,32*2048)
    total_distance_between_points_array = signal.resample(total_distance_between_points_array,32*2048)
    average_distance_between_people_array = np.divide(total_distance_between_points_array,people_count_per_frame_array)

    people_count_per_frame_array = np.nan_to_num(people_count_per_frame_array)
    sum_of_people_areas_array = np.nan_to_num(sum_of_people_areas_array)
    total_distance_between_points_array = np.nan_to_num(total_distance_between_points_array)
    average_distance_between_people_array = np.nan_to_num(average_distance_between_people_array)

    people_count_per_frame_array = np.reshape(people_count_per_frame_array,(32,2048))
    sum_of_people_areas_array = np.reshape(sum_of_people_areas_array,(32,2048))
    total_distance_between_points_array = np.reshape(total_distance_between_points_array,(32,2048))
    average_distance_between_people_array = np.reshape(average_distance_between_people_array,(32,2048))

    save_feature(people_count_per_frame_array,output_folder,'person_count',video_name)
    save_feature(sum_of_people_areas_array,output_folder,'bb_areas',video_name)
    save_feature(total_distance_between_points_array,output_folder,'interhuman_distance',video_name)
    save_feature(average_distance_between_people_array,output_folder,'avg_interhuman_distance',video_name)

def get_person_count(persons_data):
    '''Get total number of people in the frame'''
    num_people = persons_data.shape[0]
    return num_people

def get_total_distance(persons_data):
    '''Get total distance between all people'''
    total_distance = 0
    coords = []
    if persons_data.shape[0] in [0,1]:
        pass
    else:
        for row in persons_data:
            x_centre = row[1]
            y_centre = row[2]
            coords.append((x_centre,y_centre))
        coords = np.array(coords)
        total_distance = np.sum(distance.cdist(coords, coords, 'euclidean'))
    return total_distance

def get_total_area(persons_data):
    '''Get sum of individual areas of bounding boxes for people'''
    total_area = 0
    if persons_data.shape[0] in [0,1]:
        pass
    else:
        for row in persons_data:
            width = row[3]
            height = row[4]
            area = width*height
            total_area+=area
    return total_area

def main():
    output_folder = 'small-cq'
    frame_dict = np.load('/DATA/ReworkSultani/small-cq/UCF/frames.pkl',allow_pickle=True)
    video_files_from_splits = get_list_videos_for_yolo_from_splits(f'/DATA/ReworkSultani/{output_folder}/UCF/')
    # print(video_files_from_splits)
    # make_frame_features(video_files_from_splits,output_folder)
    # make_person_features(video_files_from_splits)
    print(max(frame_dict.values()))


if __name__ == "__main__":
    main()


    # for f in glob.glob('/DATA/ReworkSultani/pvmini-r3d/UCF/all_flows/**/*.npy',recursive=True):
    #     os.rename(f,f.replace('.mp4',''))


