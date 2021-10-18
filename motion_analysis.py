#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:58:06 2021

@author: massimobernava
"""

import cv2
import scipy.io
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import KernelKMeans
import numpy as np
import math
import os
import pickle

split_size=32
data_path="./"
save_video_path="./save/test_{}_{}_{}_{}.avi"
cluster_count=128

def zero_rem(data):
    last=data[0]
    for i in range(1, len(data) - 1):
        
        if data[i,0]==0.0:
            data[i] = last#(data[i-1]+data[i+1])/2
        else:
            last=data[i]

def count_frames_manual(video):
	total = 0
	while True:
		(grabbed, frame) = video.read()
		if not grabbed:
			break
		total += 1
	return total

def get_splits(data_path,split_size):
    
    mat = scipy.io.loadmat(data_path)

    trj=mat['trajectory']
    zero_rem(trj)

    #plt.plot(trj[:,0],trj[:,1])
    
    data=np.reshape(trj[:math.floor(len(trj)/split_size)*split_size],(-1,split_size,2))
    #print("Data size: ",len(data)," for ",data_path)
    
    return data

def get_cluster(data_path,file_id,cluster_count,split_size):
    data_array=[]
    path_array=[]
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(file_id+".mat"):
                file_path=os.path.join(root, file)
                print(file_path)
                data=get_splits(file_path,split_size)
                data_array.append(data)
                path=[]
                start_frame=0
                for i in range(len(data)):
                    path.append([root,start_frame,start_frame+split_size-1])
                    start_frame+=split_size
                path_array.append(path)
    data=np.concatenate(data_array)
    path=np.concatenate(path_array)
        
    for i in range(len(data)):
        data[i]=(data[i]-data[i].mean())/data[i].std() #singola clip centrata nello zero
        #data[i]=(data[i]-data.mean())/data.std()
        
    #cluster_count=math.ceil(math.sqrt(len(data)))
    #model = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")
    model = KernelKMeans(n_clusters=cluster_count,kernel="gak",kernel_params={"sigma": "auto"},n_init=50)

    labels = model.fit_predict(data)
    return data,labels,path


def plot_cluster(data,labels,file_id, cluster_count):
    plot_count = math.ceil(math.sqrt(cluster_count))

    fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
    fig.suptitle('Clusters:'+file_id)
    row_i=0
    column_j=0
    # For each label there is,
    # plots every series with that label
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot(data[i,:,0],data[i,:,1],c="gray",alpha=0.4)
                cluster.append(data[i])
        if len(cluster) > 0:
            av=np.average(cluster,axis=0)
            axs[row_i, column_j].plot(av[:,0],av[:,1],c="red")
            #axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
        column_j+=1
        if column_j%plot_count == 0:
            row_i+=1
            column_j=0
        
    #plt.show()
    plt.savefig(file_id+'_cluster.pdf')  

def get_split_position(path_array,path,startframe,endframe):
    
    for i in range(len(path_array)):
        if path_array[i][0]==path and path_array[i][1]==startframe and path_array[i][2]==endframe:
            return i
    return -1

def save_clips(save_path,label,tot_labels,tot_path):
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(save_path.format(label[0],label[1],label[2],label[3]), fourcc, 20.0,(640,480))
    w_frame=0
    for i in range(len(tot_labels)):
        if label[0]==tot_labels[i][0] and label[1]==tot_labels[i][1] and label[2]==tot_labels[i][2] and label[3]==tot_labels[i][3]:
            print("{}->{}".format(i,tot_path[i]))
            video = cv2.VideoCapture(tot_path[i][0]+".avi")
            
            n_frame = 0
            while True:
                (grabbed, frame) = video.read()
                if not grabbed:
                    break
                if n_frame>=int(tot_path[i][1]) and n_frame<=int(tot_path[i][2]):
                    out.write(frame)
                    w_frame+=1
                n_frame += 1
            
            video.release()
    print("Writed {} frames".format(w_frame))
    out.release()
    
#=============MAIN
data_righthand,labels_righthand,path_righthand = get_cluster(data_path,"righthand",cluster_count,split_size)
plot_cluster(data_righthand,labels_righthand,"righthand", cluster_count)
data_lefthand,labels_lefthand,path_lefthand = get_cluster(data_path,"lefthand",cluster_count,split_size)
plot_cluster(data_lefthand,labels_lefthand,"lefthand", cluster_count)
data_rightfoot,labels_rightfoot,path_rightfoot = get_cluster(data_path,"rightfoot",cluster_count,split_size)
plot_cluster(data_rightfoot,labels_rightfoot,"rightfoot", cluster_count)
data_leftfoot,labels_leftfoot,path_leftfoot = get_cluster(data_path,"leftfoot",cluster_count,split_size)
plot_cluster(data_leftfoot,labels_leftfoot,"leftfoot", cluster_count)

size_labels=min(len(labels_righthand),len(labels_lefthand),len(labels_rightfoot),len(labels_leftfoot))
tot_labels=np.zeros((size_labels,4))
tot_data=[]
tot_path=[]

for i in range(size_labels):
    lefthand_index=get_split_position(path_lefthand,path_righthand[i][0],path_righthand[i][1],path_righthand[i][2])
    rightfoot_index=get_split_position(path_rightfoot,path_righthand[i][0],path_righthand[i][1],path_righthand[i][2])
    leftfoot_index=get_split_position(path_leftfoot,path_righthand[i][0],path_righthand[i][1],path_righthand[i][2])
    if lefthand_index!=-1 and rightfoot_index!=-1 and leftfoot_index!=-1:
        tot_labels[i]=[labels_righthand[i],labels_lefthand[lefthand_index],labels_rightfoot[rightfoot_index],labels_leftfoot[leftfoot_index]]
        tot_data.append([data_righthand[i],data_lefthand[lefthand_index],data_rightfoot[rightfoot_index],data_leftfoot[leftfoot_index]])
        tot_path.append(path_righthand[i])
        
with open('tot_labels.pkl', 'wb') as tot_labels_file:
        pickle.dump(tot_labels, tot_labels_file)

with open('tot_data.pkl', 'wb') as tot_data_file:
        pickle.dump(tot_data, tot_data_file)
     
with open('tot_path.pkl', 'wb') as tot_path_file:
        pickle.dump(tot_path, tot_path_file)
        
unique, counts = np.unique(tot_labels, return_counts=True,axis=0)
count_sort_ind = np.argsort(-counts)
unique[count_sort_ind]

#for i in range(32):
#    save_clips(save_video_path,unique[count_sort_ind][i],tot_labels,tot_path)
    
