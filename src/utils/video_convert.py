#! /usr/bin/env python
from lib2to3.pytree import convert
import os, os.path, sys, json, re
import glob

def convert_mp4v_to_h264(input_vid_list):
  for vid in input_vid_list:
    input_file = vid
    output_file = vid.replace('.mp4','a.mp4',)
    # print(input_file)
    # print(output_file)
    os.system(f'ffmpeg -i {input_file} -vcodec libx264 -acodec aac {output_file}')
    os.remove(input_file)
    os.rename(output_file,input_file)

def normal_vid_name():
  normal_dir = '/DATA/fss/all_rgbs/Normal_Videos_event/'
  # vid_list = glob.glob(normal_dir+'*.txt')
  # for vid in vid_list:
  #   os.rename(vid,vid.replace('/Normal_Videos_event/Normal_Videos','/Normal_Videos_event/Normal_Videos_'))
  vid_list = glob.glob(normal_dir+'*.txt')
  for vid in vid_list:
    os.rename(vid,vid.replace('__','_'))
import numpy as np
def convert_txt_to_numpy(f):
  data = np.loadtxt(f)
  np.save(f.replace('txt','npy'),data)
  os.remove(f)

def feature_rename():
  features = glob.glob('/DATA/')


def main():
  vid_list = glob.glob('/DATA/runs_yolo/*/*.mp4')
  convert_mp4v_to_h264(vid_list)
if __name__ == '__main__':
  files = '/DATA/r3d_feats/**/*.txt'
  for f in glob.glob(files,recursive=True):
    convert_txt_to_numpy(f)
