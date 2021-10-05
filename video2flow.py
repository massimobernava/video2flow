import os
import sys
sys.path.append('RAFT/core')

import argparse 
from collections import OrderedDict

import cv2
#import numpy as np
import torch

from raft import RAFT
from utils import flow_viz

video_dir="./videos/fixed/"
model_path="./models/raft-sintel.pth"

def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame

def get_cpu_model(model):
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model


def inference(args):
    # get the RAFT model
    model = RAFT(args)
    # load pretrained weights
    pretrained_weights = torch.load(args.model,map_location=torch.device('cpu'))

    save = args.save

    if torch.cuda.is_available():
        device = "cuda"
        # parallel between available GPUs
        model = torch.nn.DataParallel(model)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)
        model.to(device)
    else:
        device = "cpu"
        # change key names for CPU runtime
        pretrained_weights = get_cpu_model(pretrained_weights)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)

    # change model's mode to evaluation
    model.eval()

    video_path = args.video
    # capture the video and get the first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame_1 = cap.read()

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    file_id = os.path.splitext(video_path)[0]
    out = cv2.VideoWriter(file_id+'_flow.mkv', fourcc, 20.0, (frame_1.shape[1],frame_1.shape[0]))
    
    # frame preprocessing
    frame_1 = frame_preprocess(frame_1, device)

    counter = 0
    with torch.no_grad():
        while True:
            # read the next frame
            ret, frame_2 = cap.read()
            if not ret:
                break

            # preprocessing
            frame_2 = frame_preprocess(frame_2, device)
            

            # predict the flow
            flow_low, flow_up = model(frame_1, frame_2, iters=args.iters, test_mode=True)
            
            #save the flow
            if save:
                flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
                flo = flow_viz.flow_to_image(flo,convert_to_bgr=True)
                out.write(flo)
            
            if not ret:
                break
            frame_1 = frame_2
            counter += 1
            print("   frame:",counter)
            
    cap.release()
    out.release()

for root, dirs, files in os.walk(video_dir):
     for file in files:
         if file.endswith(".mp4"):
             print("Extracting flow from: ",file)
             video_path=os.path.join(root, file)
             args = argparse.Namespace(model=model_path,small=False,save=True,video=video_path,iters=12,mixed_precision=False)
             inference(args)
