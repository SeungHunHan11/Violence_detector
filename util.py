import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import torch
from torchvision import datasets, models, transforms  
from torch.utils.data.sampler import SubsetRandomSampler  
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import sys
sys.path.append('./Violence_detector/')


def capture(filename,timesep,rgb,h,w):

    '''
    Transforming feeded image to array
    
    Insert frame with cv2 videocapture frame format
    
    Press q to exit
    '''
    tmp = []
    frames = np.zeros((timesep,rgb,h,w), dtype=np.float)
    i=0
    cap = cv2.VideoCapture(filename)

    if cap.isOpened():
        rval , frame = cap.read()
    else:
        rval = False
    frm = resize(frame,(h, w,rgb))
    frm = np.expand_dims(frm,axis=0)
    frm = np.moveaxis(frm, -1, 1)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1

    while i<timesep:
        tmp[:] = frm[:]
        rval , frame =cap.read()
        frm = resize(frame,(h, w,rgb))
        frm = np.expand_dims(frm,axis=0)
        frm = np.moveaxis(frm, -1, 1)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames[i][:] = frm

        i +=1
    del frm
    del rval
    return frames


import cv2
import os

def manual_count(filename):
    i=0
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval , frame = vc.read()
        if frame is None:
            print(filename)
            return False
    else:
        rval = False
    i +=1
    while i < 45:
        rval, frame = vc.read()
        if frame is None:
            print(filename)
            return False
        i +=1

    return True

def path_list(main_dir,filename):
    dk = []
    label = []
    i = 0
    for x in os.listdir(main_dir):
        print(x)
        if 1 == 1:
            td = main_dir+x+'/'
            for file in os.listdir(td):
                #print(td+file)
                if manual_count(td+file):
                    fl = os.path.join(td, file)
                    dk.append(fl)
                    if x == 'Violence':
                        label.append(1)
                    else:
                        label.append(0)
    df = pd.DataFrame(data={"file": dk, "label": label})
    df.to_csv(filename, sep=',',index=False)
    return filename



class TaskDataset(Dataset):
    def __init__(self, datas,timesep,rgb,h,w):
        """
        Args:
            datas: pandas dataframe contain path to videos files with label of them
            timesep: number of frames
            rgb: number of color chanles
            h: height
            w: width
            
            Courtesy of @mamonraab
                 
        """
        self.dataloctions = datas
        self.timesep,self.rgb,self.h,self.w=timesep,rgb,h,w

    def __len__(self):
        return len(self.dataloctions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video = capture(self.dataloctions.iloc[idx, 0],self.timesep,self.rgb,self.h,self.w)

        return torch.from_numpy(video), torch.from_numpy(np.asarray(self.dataloctions.iloc[idx, 1]))
    
    

def infer_capture(cap,timesep,rgb,h,w):

    '''
    Transforming feeded image to array
    
    Insert frame with cv2 videocapture frame format
    
    Press q to exit
    '''
    tmp = []
    frames = np.zeros((timesep,rgb,h,w), dtype=np.float)
    i=0

    if cap.isOpened():
        rval , frame = cap.read()
    else:
        rval = False
    frm = resize(frame,(h, w,rgb))
    frm = np.expand_dims(frm,axis=0)
    frm = np.moveaxis(frm, -1, 1)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1

    while i<timesep:
        tmp[:] = frm[:]
        rval , frame =cap.read()
        frm = resize(frame,(h, w,rgb))
        frm = np.expand_dims(frm,axis=0)
        frm = np.moveaxis(frm, -1, 1)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames[i][:] = frm

        i +=1
    del rval
    return frames
