from pickle import FALSE
from tkinter.tix import Tree
import torch
from tqdm import tqdm 
from pathlib import Path
import os
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms  
from torch.utils.data.sampler import SubsetRandomSampler  
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection, metrics
import sys
from glob import glob
import pandas as pd 
import argparse
import warnings
import numpy as np
import random
sys.path.append('./Violence_detector')

from util import TaskDataset,path_list
from model import CNN_Vit


__file__='train.py'
ROOT = Path(os.path.dirname(os.path.realpath('__file__'))).absolute()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.abspath(os.path.join('/',ROOT))) 

def seed_everything(seed):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def loading(df,rgb=3,h=200,w=200,timestep=40,
numwork=4,batchsize=2,pin=True,droplast=True,isshuffle=False,splitsize=0.4):

    #vio=glob(os.path.join(ROOT/'dataset/train/10_FPS/violent/*jpg'))
    #novio=glob(os.path.join(ROOT/'dataset/train/10_FPS/non_violent/*jpg'))

    #vio=glob(os.path.join(ROOT/'real life violence situations/Real Life Violence Dataset/Violence/*.mp4'))
    #novio=glob(os.path.join(ROOT/'real life violence situations/Real Life Violence Dataset/NonViolence/*.mp4'))



    for idx in range(len(df)):
        df.iloc[idx,0]=os.path.join(ROOT/df.iloc[idx,0])

    #vio_df=pd.DataFrame(data={'filepath':vio,"label":1})
    #novio_df=pd.DataFrame(data={'filepath':novio,"label":0})
    #set=pd.concat([vio_df,novio_df],axis=0)

    train_df, valid_df = model_selection.train_test_split(
        df, test_size=splitsize, random_state=42,stratify=df.iloc[:,1]
    )

    t_loader = DataLoader(dataset=TaskDataset(train_df,timestep,rgb,h,w),
            batch_size=batchsize,pin_memory=pin,
            drop_last=droplast,
            num_workers=numwork,shuffle=isshuffle)
    v_loader = DataLoader(dataset=TaskDataset(valid_df,timestep,rgb,h,w),
            batch_size=batchsize,pin_memory=pin,
            drop_last=droplast,
            num_workers=numwork,shuffle=isshuffle)

    return t_loader,v_loader



def train(model,train_loader,val_loader,optimizer,
loss_fn, device,epochs,scheduler,threshold):

    threshold = torch.tensor([threshold]).to(dtype=torch.float32,device=device)


    print('Training with {}'.format(device))
    model.to(device)

    for epoch in tqdm(range(epochs)):
        
        model.train()
        acc=0
        val_loss=0
        val_acc=0
        for idx, (xx,labels) in enumerate(train_loader):

            xx=torch.as_tensor(xx,device=device,dtype=torch.float32)
            labels=torch.as_tensor(labels,device=device,dtype=torch.float).view(-1,1)
            pred=model(xx) 
            loss= loss_fn(pred,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_label=torch.sigmoid(pred)
            final_label=(pred_label>threshold).float()*1
            acc+=(torch.sum(final_label==labels).item()/len(labels))

            if idx%500==0:
                print('Train Accuracy at {} batch: {:.4f}'.format(idx,acc/((idx+1))))


        if 'ReduceLROnPlateau' not in str(sched):
            scheduler.step()


        with torch.no_grad():
            model.eval()
            for idx,(xx,labels) in enumerate(val_loader):

                xx=torch.as_tensor(xx,device=device,dtype=torch.float32)
                labels=torch.as_tensor(labels,device=device,dtype=torch.float).view(-1,1)
                pred=model(xx) 
                loss= loss_fn(pred,labels)
                val_loss+=loss

                pred_label=torch.sigmoid(pred)
                final_label=(pred_label>threshold).float()*1
                val_acc+=(torch.sum(final_label==labels).item()/len(labels))

                
                if idx%300==0:
                    print('Validation Accuracy at {} batch: {:.4f}'.format(idx,val_acc/((idx+1))))
        
        val_acc/=(idx+1)
        val_loss/=(idx+1)

        if 'ReduceLROnPlateau' in str(sched):
            scheduler.step(val_loss)

        if epoch==0: 
            best_val_loss=val_loss
            best_val_acc=val_acc

        if best_val_acc<val_acc:
            best_val_loss=val_loss
            best_val_acc=val_acc

            best_path=os.path.join(ROOT/'best_param/best.pt')

            print('Best Result renewed')


            torch.save(model.state_dict(),best_path)
        
        print('Current Loss is {:.4f} & Accuracy is {:.4f}'.format(val_loss,val_acc))
        print("Current Best Loss is {:.4f} & Accuracy is {:.4f}".format(best_val_loss,best_val_acc))

        last_path=os.path.join(ROOT/'best_param/last.pt')
        torch.save(model.state_dict(),last_path)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    help='Set number of epochs'
    )

parser.add_argument(
    '--device',
    type=str,
    default='cuda',
    help='cuda for gpu else cpu'
    )

parser.add_argument(
    '--batch',
    type=int,
    default=4,
    help='Batch Size'
    )

parser.add_argument(
    '--numwork',
    type=int,
    default=4,
    help='Loader Number of workers'
    )  

parser.add_argument(
    '--sourcename',
    type=str,
    default='mtrain.csv',
    help='Video Source Path'
    )  

parser.add_argument(
    '--threshold',
    type=float,
    default=0.6,
    help='Set threshold'
    )  

parser.add_argument(
    '--timestep',
    type=int,
    default=30,
    help='Set Timestep'
    )  

parser.add_argument(
    '--dropout',
    type=float,
    default=0.4,
    help='Set Dropout rate'
    )  

args=vars(parser.parse_args())


if __name__ =='__main__':

    warnings.filterwarnings("ignore")
    
    seed_everything(1001)

    try:
        df=pd.read_csv(os.path.join(ROOT/args['sourcename']))
    except:

        path_list('./real life violence situations/Real Life Violence Dataset/','./'+args['sourcename'])

        df=pd.read_csv(os.path.join(ROOT/args['sourcename']))

    t_loader,v_loader=loading(df=df,batchsize=args['batch'],numwork=args['numwork'],
    timestep=args['timestep'])

    if args['device']=='cuda' and torch.cuda.is_available():
        device='cuda'

    else:
        device='cpu'
        print('Using CPU')


    model=CNN_Vit(dev=device,timestep=args['timestep'],dropout=args['dropout'])
    criterion= nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,momentum=0.0001)
    sched =  lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.5, patience=2 , verbose=True)
    epochs=args['epochs']
    threshold=args['threshold']

    torch.cuda.empty_cache()

    train(model,t_loader,v_loader,optimizer,
    criterion,device,epochs,sched,threshold)

# python train.py --epochs 50 --batch 4 --numwork 8