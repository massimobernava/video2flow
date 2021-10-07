import os
import sys
sys.path.append('RAFT/core')

import argparse 
from collections import OrderedDict

import cv2
import numpy as np
import torch

from raft import RAFT
from utils import flow_viz

#method_name="TVL1"
method_name="RAFT"
#method_name="RLOF"
#method_name="Farneback"
#method_name="LK"

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

def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # Read the video and first frame
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    file_id = os.path.splitext(video_path)[0]
    out = cv2.VideoWriter(file_id+'_'+method_name+'_flow.mkv', fourcc, 20.0, (old_frame.shape[1],old_frame.shape[0]))
    
    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    n_frame=1
    
    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        #frame_copy = new_frame
        if not ret:
            break
        print("   frame: ",n_frame)
        n_frame+=1
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #cv2.imshow("frame", frame_copy)
        #cv2.imshow("optical flow", bgr)
        out.write(bgr)
        #k = cv2.waitKey(25) & 0xFF
        #if k == 27:
        #    break

        # Update the previous frame
        old_frame = new_frame    
    out.release()
    cap.release()
    #cv2.destroyAllWindows()

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
             
             if method_name=="RAFT":
                 args = argparse.Namespace(model=model_path,small=False,save=True,video=video_path,iters=12,mixed_precision=False)
                 inference(args)
             elif method_name=="LK":
                 method = cv2.optflow.calcOpticalFlowSparseToDense
                 dense_optical_flow(method, video_path, to_gray=True)
             elif method_name=="Farneback":
                 method = cv2.calcOpticalFlowFarneback
                 params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback's algorithm parameters
                 dense_optical_flow(method, video_path, params, to_gray=True)
             elif method_name=="RLOF":
                 method = cv2.optflow.calcOpticalFlowDenseRLOF
                 dense_optical_flow(method, video_path)
             elif method_name=="TVL1":
                 method = cv2.optflow.DualTVL1OpticalFlow_create().calc
                 dense_optical_flow(method, video_path, to_gray=True)
                
