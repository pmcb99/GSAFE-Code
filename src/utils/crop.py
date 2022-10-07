from genericpath import getmtime
import os
import cv2
import glob
from cv2 import bitwise_and
import numpy as np
import re
from PIL import Image, ImageDraw
import time
def extract_frames(video_path):
    video_name = video_path.split('/')[-1].split('.')[0]
    if 'Norm' in video_name:
        class_name = 'Normal_Videos_event' 
        parent_directory = '/'.join(video_path.split('/')[:-1]).replace(class_name,video_name)+'/'
        masked_dir = '/'.join(video_path.split('/')[:-2])+'/Masked/'
    else:
        class_name = re.sub('\d+','',video_name.split('_')[0])
        parent_directory = '/'.join(video_path.split('/')[:-1]).replace(class_name,video_name)+'/'
        masked_dir = '/'.join(video_path.split('/')[:-2])+'/Masked/'
    # if not os.path.exists(parent_directory):
    #     os.mkdir(parent_directory)
    parent_directory = masked_dir

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        name = video_name+"_"+str(count+1)
        output_path = parent_directory+name+'.png'
        image = np.array(image)
        mask_image = mask_function(name, image, class_name,'fg')
        cv2.imwrite(output_path, mask_image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    processed_frames = sorted(glob.glob(parent_directory+'*.png'),key=os.path.getmtime)
    generate_video(processed_frames,parent_directory+video_name+'_fgmask.mp4')
    for f in processed_frames:
        os.remove(f)

    time.sleep(1)
    
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        name = video_name+"_"+str(count+1)
        output_path = parent_directory+name+'.png'
        image = np.array(image)
        mask_image = mask_function(name, image, class_name,'bg')
        cv2.imwrite(output_path, mask_image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    processed_frames = sorted(glob.glob(parent_directory+'*.png'),key=os.path.getmtime)
    generate_video(processed_frames,parent_directory+video_name+'_bgmask.mp4')
    for f in processed_frames:
        os.remove(f)
    os.rename(video_path,'/data/Normal/Done/'+video_name+'.mp4')
    

def generate_video(input_video_frame_list,output_video_file):
    img = cv2.imread(input_video_frame_list[0])
    height, width, layers = img.shape
    size = (width,height)
    out = cv2.VideoWriter(output_video_file,cv2.VideoWriter_fourcc(*'m','p','4','v'),30,size)
    for frame in input_video_frame_list:
        img = cv2.imread(frame)
        height, width, layers = img.shape
        size = (width,height)
        out.write(img)
    out.release()

def mask_function(name, image, class_name,fg_or_bg):
    f = f'/data/runs_yolo/label_folder/{class_name}/labels/{name}.txt'
    frame_glob = glob.glob(f)
    if frame_glob == []:
        '''No YOLO data for this frame'''
        if fg_or_bg == 'fg':
            image[:,:,:]=255 
        else:
            image[:,:,:]=0 
        return image
    frame = frame_glob[0]
    data = np.array(np.loadtxt(frame))
    h, w, layers = image.shape
    if data.ndim == 1:
        x_centre = int(data[1]*w)
        y_centre = int(data[2]*h)
        width = int(data[3]*w)
        height = int(data[4]*h)
        start_point = (x_centre-width//2,y_centre+height//2)
        end_point = (x_centre+width//2,y_centre-height//2)
        if fg_or_bg == 'fg':
            fg_image = cv2.rectangle(image,start_point,end_point,(0,0,0),-1)
            return fg_image
        else:
            # image = cv2.imread('/data/Distracted/Abuse029_x264_45.png')
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(mask, start_point, end_point, 255, -1)
            # cv2.imshow("Mask", mask)
            masked = cv2.bitwise_and(image, image, mask=mask)
            return masked
    for row in data:
        x_centre = int(row[1]*w)
        y_centre = int(row[2]*h)
        width = int(row[3]*w)
        height = int(row[4]*h)
        start_point = (x_centre-width//2,y_centre+height//2)
        end_point = (x_centre+width//2,y_centre-height//2)
        mask = np.zeros(image.shape[:2], dtype="uint8")
        if fg_or_bg == 'fg':
            fg_image = cv2.rectangle(image,start_point,end_point,(0,0,0),-1)
        else:
            cv2.rectangle(mask, start_point, end_point, 255, -1)
    if fg_or_bg == 'fg':
        return fg_image
    else:
        masked = cv2.bitwise_and(image, image, mask=mask)
        return masked


def full_run(video_path):
    output_dir = "/".join(video_path.split('/')[:-1])
    extract_frames(video_path)
    # print(glob.glob('/data/runs_yolo/label_folder/Abuse/labels/Abuse001*'))

from tqdm import tqdm
def main():
    video_list = glob.glob('/data/Normal/**/*.mp4',recursive=True)
    for video in tqdm(video_list):
        full_run(video)

    # og = cv2.imread('/data/Distracted/Abuse029_x264_45.png')
    # cv2.imshow('OG',og)
    # mask = np.zeros(og.shape[:2],dtype='uint8')
    # cv2.rectangle(mask,(0,90),(100,190),255,-1)
    # cv2.imshow('Mask',mask)
    # final = cv2.bitwise_and(og,og,mask)
    # cv2.imshow('Final',final)
    # cv2.waitKey(0)
    # image = cv2.imread('/data/Distracted/cool.png')

    # image = cv2.imread('/data/Distracted/Abuse029_x264_45.png')
    # mask = np.zeros(image.shape[:2], dtype="uint8")
    # cv2.rectangle(mask, (0, 90), (290, 300), 255, -1)
    # cv2.rectangle(mask, (20, 20), (290, 300), 255, -1)
    # cv2.imshow("Mask", mask)
    # masked = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow("Mask applied to Image", masked)
    # cv2.waitKey(0)


main()