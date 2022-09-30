import cv2
import sys
sys.path.append('./Violence_detector')

from utils import capture, infer_capture
from model import CNN_Vit
import torch
import argparse
import pafy
from pathlib import Path
import os
from torchvision import datasets, models, transforms  
import imutils
import time
import uuid


__file__='detect.py'
ROOT = Path(os.path.dirname(os.path.realpath('__file__'))).absolute()
ROOT = Path(os.path.abspath(os.path.join('/',ROOT))) 

def inference(model,device,vid_input,threshold=0.6,timesep=10,rgb=3,h=200,w=200):
    
    model.to(device)
    model.eval()

    with torch.no_grad():
    
        frm=infer_capture(vid_input,timesep,rgb,h,w)
        frm=torch.from_numpy(frm).to(dtype=torch.float32,device=device)
        frm=frm.view(1,timesep,rgb,h,w)
        label=model(frm)
    
        pred_label=torch.sigmoid(label)


    if pred_label>threshold:
        return 'Violence Detected'
    else:
        return 'No_violence'

def run(feed,model,device,threshold):

    cap=cv2.VideoCapture(feed)
    fp = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out=cv2.VideoWriter(os.path.join(ROOT/('vid_rec/'+str(uuid.uuid1())+time.ctime().replace(' ','').replace(':','')+'.mp4')),
                            cv2.VideoWriter_fourcc(*'mp4v'), fp, (w, h))
    while True:
            
        ret , frame = cap.read()

        if not (ret or cap.isOpened()):
            print('Video Feed not loaded')
            break
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        label=inference(model,device,cap,threshold=threshold)

        cv2.putText(frame,label,(0,h-10),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),1)
        out.write(frame)
        cv2.imshow('Video Feed',frame)
        
        key=cv2.waitKey(1)
        
        if key==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser()

parser.add_argument(
    '--weights',
    type=str,
    default=ROOT/'best_param/best.pt',
    help='Set directory for weight'
    )

parser.add_argument(
    '--device',
    type=str,
    default='cuda',
    help='cuda for gpu else cpu'
    
    )

parser.add_argument(
    '--source',
    type=str,
    default=0,
    help='set video feed. 0 for webcam 1.'
    )

parser.add_argument(
    '--threshold',
    type=float,
    default=0.6,
    help='Set threshold'
    )  

args=vars(parser.parse_args())


if __name__=='__main__':
    
    if str(args['source']).lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        url=args['source']
        video = pafy.new(url)
        best  = video.getbest(preftype="mp4")
        raw_input=best.url
    
    else:
        raw_input=args['source']
    
    if args['device']=='cuda' and torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    
    model=CNN_Vit(dev=device,timestep=10)
    model.load_state_dict(torch.load(args['weights']))
    
    run(raw_input,model,device,threshold=args['threshold'])