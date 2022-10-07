#Use split files in given directory to generate yolo files. Needed for limited disk space.

import contextlib
import sys
sys.path.insert(1,'/DATA/2022-mcm-pmb/yolov5')
from detect import *
import numpy as np
import matplotlib.image as mpimg
import pickle as pickle
import cv2
import os
from os import listdir
from os.path import isfile, join
import glob
from sympy import root
from tqdm import tqdm

def extract_frames(video_path,dest):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        name = dest+video_path.split('/')[-1].replace('.avi','').replace('.mp4','')+"_Frame"+str(count+1)+'.png'
        #Optional segment line below
        # img = cv2.resize(image,(256,320))
        cv2.imwrite(name, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

def generate_video(frame_list, dest_file):
    '''Give list of frames, the destination video file, then framerate '''
    framerate = 30
    img = cv2.imread(frame_list[0])
    height, width, layers = img.shape
    size = (width,height)
    out = cv2.VideoWriter(dest_file,cv2.VideoWriter_fourcc(*'m','p','4','v'), framerate, size)
    for filename in frame_list:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        out.write(img)
    out.release()


def pytorch_yolo_dir(tmp_dir,yolo_dir,class_folder):
    run(
            weights=ROOT / 'yolov5l.pt',  # model.pt path(s)
            source=tmp_dir,  # file/dir/URL/glob, 0 for webcam
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=True,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=[0,1,2,3,5,7],  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project= yolo_dir,  # save results to project/name
            name=class_folder,  # save results to project/name
            exist_ok=True,  # existing project/name ok, do not increment
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=True,  # hide labels
            hide_conf=True,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    )

def pytorch_yolo(input_video_path, output_video_path):
    video_id = input_video_path.split('/')[-1].split('.')[0]
    run(
            weights=ROOT / 'yolov5l.pt',  # model.pt path(s)
            source=input_video_path,  # file/dir/URL/glob, 0 for webcam
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=True,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=[0,1,2,3,5,7],  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project= output_video_path,  # save results to project/name
            name=video_id,  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=True,  # hide labels
            hide_conf=True,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    )
def sorter(file_name):
    number = int(file_name.split('_')[-1].replace('Frame','').replace('.png',''))
    return number
from pathlib import Path

def get_videos_from_split_files(test_norm,test_anom):
    videos = []
    with open(test_norm,'r') as f:
        videos = f.readlines()
        videos = [video.split('/')[-1].split(' ')[0] for video in videos if video != '\n']
    with open(test_anom,'r') as f:
        pass
        # videos+=f.readlines()
        # videos = [video.split('/')[-1].split('|')[0] for video in videos if video != '\n']
    return videos

def main():
    import wandb
    wandb.init(project='GenerateYoloUCF',notes='Part3Anom',mode='disabled')
    run_name = wandb.run.name
    split_root = '/DATA/ReworkSultani/pv-mini/UCF/'
    folder_name = '/DATA/Testing_Normal_Videos_Anomaly/'
    video_list = get_videos_from_split_files(split_root+'test_normalv2.txt',split_root+'test_anomalyv2.txt')
    dataset_folder_list = sorted(glob.glob(f'/DATA/{folder_name}/*'))

    for video in video_list:
        video = folder_name + video
        video_id = video.split('/')[-1].split('.')[0]
        output_video_path = Path('/DATA/runs_yolo/Normal_Videos/') 
        if os.path.exists(output_video_path / video_id / (video_id+'.mp4'))==False:
            print(output_video_path)
            pytorch_yolo(video,output_video_path)
if __name__ == "__main__":
    main()
        


