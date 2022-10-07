import mxnet as mx
import multiprocessing
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import glob
from tqdm import tqdm
from mxnet import image
import numpy as np
from mxnet.gluon.data.vision import transforms
import gluoncv
from matplotlib import pyplot as plt
# using cpu
ctx = mx.cpu(0)
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
import numpy as np
import cv2
from gluoncv.data.transforms.presets.segmentation import test_transform
model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import mxnet as mx
from mxnet import image
import numpy as np
from mxnet.gluon.data.vision import transforms
import gluoncv
from matplotlib import pyplot as plt
# using cpu
ctx = mx.cpu(0)
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
import numpy as np
import cv2
from gluoncv.data.transforms.presets.segmentation import test_transform
import gluoncv
# using cpu
ctx = mx.cpu(0)

def SegmentTheFrames(filename):

    img = image.imread(filename)

    from matplotlib import pyplot as plt

    ##############################################################################
    # normalize the image using dataset mean
    from gluoncv.data.transforms.presets.segmentation import test_transform
    img = test_transform(img, ctx)

    ##############################################################################
    # Load the pre-trained model and make prediction
    # ----------------------------------------------
    #
    # get pre-trained model
    model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)

    ##############################################################################
    # make prediction using single scale
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    ##############################################################################
    # Add color pallete for visualization
    from gluoncv.utils.viz import get_color_pallete
    import matplotlib.image as mpimg
    mask = get_color_pallete(predict, 'pascal_voc')
    mask.save(filename)

    ##############################################################################
    # show the predicted mask
    mmask = mpimg.imread(filename)
    
    #plt.imshow(mmask)
    #plt.show()
import shutil


def ExtractFrames(video_path,dest):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        name = dest+video_path.split('/')[-1].replace('.avi','').replace('.mp4','')+"_Frame"+str(count+1)+'.png'
        #Optional segment line below
        cv2.imwrite(name, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from matplotlib import pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform
import gluoncv
# using cpu
ctx = mx.cpu(0)
def SegmentTheFrames(src_file,dest_file,plot=False):
    if os.path.exists(dest_file):
        return
    img = image.imread(src_file)
    # normalize the image using dataset mean
    img = test_transform(img, ctx)
    # Load the pre-trained model and make prediction
    # ----------------------------------------------
    #
    # get pre-trained model
    model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
    ##############################################################################
    # make prediction using single scale
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    ##############################################################################
    # Add color pallete for visualization
    from gluoncv.utils.viz import get_color_pallete
    import matplotlib.image as mpimg
    mask = get_color_pallete(predict, 'pascal_voc')
    mask.save(dest_file)

    ##############################################################################
    # show the predicted mask
    
    img = cv2.imread(dest_file)
        # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    img[np.where((img==[0,0,0]).all(axis=2))] = [255,255,255]
    cv2.imwrite(dest_file,img)
    if plot == True:
        mmask = mpimg.imread(dest_file)
        plt.title('My Segmented Function')
        plt.imshow(mmask)
        plt.show()
import os
def GenerateSegVideo(og_video_frames,dest_file,framerate):
    '''Use already '''
    img = cv2.imread(og_video_frames[0])
    height, width, layers = img.shape
    size = (width,height)
    out = cv2.VideoWriter(dest_file,cv2.VideoWriter_fourcc(*'m','p','4','v'), framerate, size)
    for filename in og_video_frames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        out.write(img)
    out.release()
    
def GenerateLiteVideo(og_video_frames,dest_file,framerate):
    '''Less frames'''
    threeframes = og_video_frames[::10]
    #Intialise before loop for reference
    img = cv2.imread(threeframes[0])
    height, width, layers = img.shape
    size = (width,height)
    out = cv2.VideoWriter(dest_file,cv2.VideoWriter_fourcc(*'m','p','4','v'), framerate, size)
    for filename in threeframes:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        out.write(img)
    out.release()
    
def CreateFolders(video_path):
    #Make directory
    try:
        os.mkdir(video_path+'/OriginalFrames')
        os.mkdir(video_path+'/ProcessedVideos')
    except:
        pass

def RemoveImages(files):
    for f in files:
        os.remove(f)

from multiprocessing import Pool

subfolder = sorted(glob.glob('/home/paul/Documents/Mini/Abuse/*'))

def ProcessFunction(subfolders):
    for subfolder in subfolders:
        class_of_vids = sorted(glob.glob(subfolder+'/*.avi'),key=os.path.getmtime)
        CreateFolders(subfolder)
        for vid in class_of_vids:
            video_id = vid.replace('.avi','').split('/')[-1]
            if os.path.exists(subfolder+'/ProcessedVideos/'+video_id+'_Seg.mp4')==False:
                ExtractFrames(vid,subfolder+'/OriginalFrames/')
                print('Creating '+video_id+'_Seg.mp4')
                GenerateLiteVideo(sorted(glob.glob(subfolder+'/OriginalFrames/*.png'),key=os.path.getmtime),subfolder+'/ProcessedVideos/'+video_id+'_Lite.mp4',2)
                RemoveImages(glob.glob(subfolder+'/OriginalFrames/*.png'))
                ExtractFrames(subfolder+'/ProcessedVideos/'+video_id+'_Lite.mp4',subfolder+'/OriginalFrames/')
                for frame in sorted(glob.glob(subfolder+'/OriginalFrames/*.png'),key=os.path.getmtime):
                    dest = subfolder+'/ProcessedVideos/Segmented_'+frame.split('/')[-1]
                    SegmentTheFrames(frame,dest,False)


                GenerateSegVideo(sorted(glob.glob(subfolder+'/ProcessedVideos/*.png'),key=os.path.getmtime),subfolder+'/ProcessedVideos/'+video_id+'_Seg.mp4',2)
                RemoveImages(glob.glob(subfolder+'/ProcessedVideos/*.png'))
                RemoveImages(glob.glob(subfolder+'/OriginalFrames/*.png'))


def Process(input):
    '''with input as list of video in no further subfolders'''
    class_name = input.split('/')[6]
    dest_folder = '/home/paul/Documents/Kaggledata/Crime/'+class_name+'_Seg/'+input.split('/')[7].replace('.mp4','')
    try: 
        os.mkdir('/home/paul/Documents/Kaggledata/Crime/'+class_name+'_Seg/') #CLASS_seg
        os.mkdir(dest_folder) #VideoName
    except Exception as e:
        print('Bluder Pos1')
        print(e)

    CreateFolders(dest_folder)#create orig and processed folders

    


    #Preremove
    try:
        RemoveImages(glob.glob(dest_folder+'/OriginalFrames/*.png'))
    except Exception as e:
        print('Couldnt remove images as preremove step')

    video_id = input.replace('.mp4','').split('/')[-1]
    if os.path.exists(dest_folder+'/ProcessedVideos/'+video_id+'_Seg.mp4')==False:
        print('Creating '+video_id+'_Seg.mp4')
        #Lite vid generation
        ExtractFrames(input,dest_folder+'/OriginalFrames/')
        GenerateLiteVideo(sorted(glob.glob(dest_folder+'/OriginalFrames/*.png'),key=os.path.getmtime),dest_folder+'/OriginalFrames/'+video_id+'_Lite.mp4',2)
        RemoveImages(glob.glob(dest_folder+'/OriginalFrames/*.png'))
        #Segmented video generation
        ExtractFrames(dest_folder+'/OriginalFrames/'+video_id+'_Lite.mp4',dest_folder+'/OriginalFrames/')
        for frame in sorted(glob.glob(dest_folder+'/OriginalFrames/*.png'),key=os.path.getmtime):
            SegmentTheFrames(frame,dest_folder+'/ProcessedVideos/'+frame.split('/')[7]+'_Seg_'+frame.split('_')[-1],False)


        GenerateSegVideo(sorted(glob.glob(dest_folder+'/ProcessedVideos/*.png'),key=os.path.getmtime),dest_folder+'/ProcessedVideos/'+video_id+'_Seg.mp4',2)
        RemoveImages(glob.glob(dest_folder+'/ProcessedVideos/*.png'))
        RemoveImages(glob.glob(dest_folder+'/OriginalFrames/*.png'))

import concurrent.futures

if __name__ == "__main__":
    video_list = sorted(glob.glob('/home/paul/Documents/Kaggledata/Crime/Abuse/*'))
    for vid in video_list:
        Process(vid)
        break

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         f1 = executor.map(Process, video_list)

