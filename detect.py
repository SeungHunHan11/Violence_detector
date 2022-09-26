import cv2
import sys
from utils import capture
from model import CNN_Vit
import torch
import argparse
import pafy
from pathlib import Path
import os

sys.path.append('/violence')
__file__='detect.py'
ROOT = Path(os.path.dirname(os.path.realpath('__file__'))).absolute()
ROOT = Path(os.path.abspath(os.path.join('/',ROOT))) 

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
    default=0,
    help='set video feed. 0 for webcam 1.'
    )

args=vars(parser.parse_args())


def infernece(model,device,vid_input):
    
    model.eval()
    
    frm=capture(vid_input)
    frm=torch.from_numpy(frm).to(dtype=torch.float32,device=device)
    label=model(frm)
    label=torch.argmax(label,axis=1).item()
    
    if label==0:
        return 'No_violence'
    else:
        return 'Violence'
        


def run(feed,model,device):

    cap=cv2.VideoCapture(feed)

    while True:
            
        ret , frame = cap.read()

        if not (ret or cap.isOpened()):
            print('Video Feed not loaded')
            break
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        

        label=infernece(model,device,frame)

        cv2.putText(frame,label,(0,h-10))
        cv2.imshow('Video Feed',frame)
        
        key=cv2.waitKey(1)
        
        if key==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    
    
    if args['source'].lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
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
    
    model=torch.load(args['weights'],device=device)
        
    run(raw_input,model,device)
    