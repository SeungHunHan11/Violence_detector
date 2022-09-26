import cv2
import numpy as np
from skimage import io
from skimage.io import imread
from skimage.transform import resize
import torch
from torchvision import datasets, models, transforms  
from torch.utils.data.sampler import SubsetRandomSampler  
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def capture(frame):
    
    '''
    Transforming feeded image to array
    
    Insert frame with cv2 videocapture frame format
    
    Press q to exit
    '''

    h,w,rgb=frame.shape
    frm = np.reshape(frame,(rgb,h,w))
    #frm=frame 
    frm = np.expand_dims(frm,axis=0)
    #frm = np.moveaxis(frm, -1, 1)
    if(np.max(frm)>1):
        frm = frm/255.0


    return frm


class TaskDataset(Dataset):
    def __init__(self, datas,h=1080,w=192,dotransform=True):
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
        self.rgb,self.h,self.w=3,300,350

        self.dotransform=dotransform

        self.Transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.h, self.w)),
        ])

    def __len__(self):
        return len(self.dataloctions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image=cv2.imread(self.dataloctions.iloc[idx, 0])

        image=np.where(np.max(image)>1,image/255.0,image)


        if self.dotransform:
            image=self.Transform(image)
        

        return image, torch.from_numpy(np.asarray(self.dataloctions.iloc[idx, 1]))