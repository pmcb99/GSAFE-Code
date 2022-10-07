import contextlib
import numpy as np
import matplotlib.image as mpimg
import pickle as pickle
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.utils.viz import get_color_pallete
from gluoncv.data.transforms.presets.segmentation import test_transform
import multiprocessing
import cv2
import os
from os import listdir
from os.path import isfile, join
import glob
#from py import process
from sympy import root
from tqdm import tqdm
# using cpu
ctx = mx.gpu()

def SegmentTheFrames(filename):

    img = image.imread(filename)


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


# def ExtractFrames(video_path,dest):
#     vidcap = cv2.VideoCapture(video_path)
#     success,image = vidcap.read()
#     count = 0
#     if not success:
#         print('Couldnt make '+name)
#     while success:
#         name = dest+video_path.split('/')[-1].replace('.avi','').replace('.mp4','')+"_Frame"+str(count+1)+'.png'
#         #Optional segment line below
#         cv2.imwrite(name, image)     # save frame as JPEG file      
#         success,image = vidcap.read()
#         count += 1




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
    import matplotlib.pyplot as plt
    
    img = cv2.imread(dest_file)
        # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    img[np.where((img==[0,0,0]).all(axis=2))] = [255,255,255]
    cv2.imwrite(dest_file,img)
    if plot == True:
        mmask = mpimg.imread(dest_file)
        plt.title('My Segmented Function')
        plt.imshow(mmask)
        plt.show()


    
    
def CreateFolders(video_path):
    #Make directory
    try:
        os.mkdir(video_path+'/OriginalFrames')
        os.mkdir(video_path+'/ProcessedData')
    except:
        pass

def RemoveImages(files):
    for f in files:
        os.remove(f)





import concurrent.futures
from gluoncv import model_zoo, data, utils
import multiprocessing

class YoloNet:
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

    def yolo_function(self,image_file):
        save_file = image_file.replace('OriginalFrames','ProcessedData').replace('Lite','Yolo')
        if os.path.exists(save_file) or os.path.exists(save_file.replace('.png','.pkl')):
            return
        import matplotlib.pyplot as plt

        print('Working on '+save_file)
        x, img = data.transforms.presets.yolo.load_test(image_file, short=416)
        print('Shape of pre-processed image:', x.shape)
        class_IDs, scores, bounding_boxs = self.net(x)
        fig, ax = plt.subplots()

        ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=self.net.classes, ax=ax)
        ax.set_axis_off()
                            
        fig.savefig(save_file,bbox_inches='tight',pad_inches=0)
        plt.close(fig)
        yolo_vectors = [class_IDs[0],scores[0],bounding_boxs[0]]
        with open(save_file.replace('.png','.pkl'), 'wb') as outfile:
            pickle.dump(yolo_vectors, outfile, pickle.HIGHEST_PROTOCOL)
        mx.nd.waitall()


    def extract_frames(self,video_path,dest):
        '''Resizes to divisible by 32 dimensions as required by Gluon Yolo'''
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        while success:
            name = dest+video_path.split('/')[-1].replace('.avi','').replace('.mp4','')+"_Frame"+str(count+1)+'.png'
            #Optional segment line below
            img = cv2.resize(image,(256,320))
            cv2.imwrite(name, img)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1

    def create_lite_frames_and_video(self,file_dir):
        '''with input as file directory of video'''
        '''Use Seg or Yolo as processtype'''
        process_type='Yolo'
        dataset = 'UCF101'
        ###########Crime Dataset
        class_name = file_dir.split('/')[6]
        if 'x264' in file_dir:
            root_path = '/home/paul/Documents/Kaggledata/Crime/'
            dest_folder = root_path+class_name+'_'+process_type+'/'+file_dir.split('/')[7].replace('.mp4','').replace('.avi','')
        else:
        ###########UCF Dataset
            root_path = '/home/paul/Documents/Kaggledata/UCF101/'
            dest_folder = root_path+class_name+'_'+process_type+'/'+file_dir.split('/')[7].replace('.mp4','').replace('.avi','')


        if os.path.exists(root_path+class_name+'_'+process_type+'/') == False:
            try:
                os.mkdir(root_path+class_name+'_'+process_type+'/') #CLASS_seg
            except:
                print('Folder already exists check #class_seg comment')

        if os.path.exists(dest_folder)==False:
            os.mkdir(dest_folder) #VideoName
        CreateFolders(dest_folder)#create orig and processed folders
        #Preremove
        try:
            RemoveImages(glob.glob(dest_folder+'/OriginalFrames/*.png'))
        except Exception as e:
            print('Couldnt remove images as preremove step')

        video_id = file_dir.replace('.mp4','').replace('.avi','').split('/')[-1]

        if os.path.exists(dest_folder+'/OriginalFrames/'+video_id+'_Lite.mp4')==False:
            #Lite vid generation
            self.extract_frames(file_dir,dest_folder+'/OriginalFrames/')
            frame_reduction_factor_for_lite = 10
            self.generate_video_with_resize32(sorted(glob.glob(dest_folder+'/OriginalFrames/*.png')[::frame_reduction_factor_for_lite],
                                                key=sorter),dest_folder+'/OriginalFrames/'+video_id+'_Lite.mp4')
            print('Created '+video_id+'_Lite.mp4'+' in '+dest_folder+'/OriginalFrames/')
            RemoveImages(glob.glob(dest_folder+'/OriginalFrames/*.png'))
        self.extract_frames(dest_folder+'/OriginalFrames/'+video_id+'_Lite.mp4',dest_folder+'/OriginalFrames/')

    def do_processing(self,video_folder):
        processed_frame_list = sorted(glob.glob(video_folder+'/ProcessedData/*.png'),key=sorter)
        yolo_video_dest = video_folder+'/ProcessedData/'+video_folder.split('/')[-1]+'_Yolo.mp4'
        if not os.path.exists(yolo_video_dest):
            self.generate_video_with_resize32(processed_frame_list,yolo_video_dest)
        RemoveImages(glob.glob(video_folder+'/OriginalFrames/*.png'))
        RemoveImages(glob.glob(video_folder+'/ProcessedData/*.png'))

    def generate_video_with_resize32(self,frame_list, dest_file):
        '''Give list of frames, the destination video file, then framerate and itll resize to yolo requirements'''
        '''All frames will have different sizes after bboxs so need to resize to make video'''
        #print('Generating '+dest_file+' using the file '+og_video_frames[0]+' and similar')
        framerate = 2
        img = cv2.imread(frame_list[0])
        img = cv2.resize(img,(256,320),interpolation=cv2.INTER_AREA)
        height, width, layers = img.shape
        size = (width,height)
        out = cv2.VideoWriter(dest_file,cv2.VideoWriter_fourcc(*'m','p','4','v'), framerate, size)
        for filename in frame_list:
            img = cv2.imread(filename)
            img = cv2.resize(img,(256,320),interpolation=cv2.INTER_AREA)
            height, width, layers = img.shape
            size = (width,height)
            out.write(img)
        out.release()

def sorter(file_name):
    number = int(file_name.split('_')[-1].replace('Frame','').replace('.png',''))
    return number

def main():
    from multiprocessing import Pool
    from contextlib import closing
    folder_name = 'Crime' 
    dataset_folder_list = sorted(glob.glob(f'/home/paul/Documents/Kaggledata/{folder_name}/*'))
    dataset_folder_list = sorted(glob.glob(f'/DATA/{folder_name}/*'))
    non_yolo_class_folders = [video for video in dataset_folder_list if 'Yolo' not in video]
    yolo_obj = YoloNet()
    for class_folder in non_yolo_class_folders:
        print(f'class_folder is {class_folder}')
        if folder_name == 'UCF101':
            video_list = glob.glob(class_folder+'/*.avi')
        else:
            video_list = glob.glob(class_folder+'/*.mp4')
        with concurrent.futures.ProcessPoolExecutor(1) as ex:
            print(video_list)
            print('run yolo')
            results = ex.map(yolo_obj.create_lite_frames_and_video, video_list)
        class_video_subfolders = sorted(glob.glob(class_folder+'_Yolo/*'),key=os.path.getmtime)
        for class_video_subfolder in class_video_subfolders:
            print(class_video_subfolder)
            image_list = sorted(glob.glob(class_video_subfolder+'/OriginalFrames/*.png',recursive=True),key=os.path.getmtime)
            with concurrent.futures.ProcessPoolExecutor(1) as ex:
               results = ex.map(yolo_obj.yolo_function, image_list)
        video_folder_list = sorted(glob.glob(class_folder+'_Yolo/*',recursive=True),key=os.path.getmtime)
        with concurrent.futures.ProcessPoolExecutor(1) as ex:
            results = ex.map(yolo_obj.do_processing, video_folder_list)


if __name__ == "__main__":
    main()
        


