#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:09:36 2019

@author: hesun
"""
from __future__ import print_function
import torch
import torch.utils.data as data
import os
import os.path
import nibabel as nib
import numpy as np
import copy
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from torchvision import transforms as T
import cv2

def my_segmentation_transform(input1, target1):	
        inout=[]	
        tarout=[]	
        for i in range(len(input1)):	
            r=copy.deepcopy(input1[i].squeeze())	
            target=F.to_pil_image(target1[i].astype("int32"),"I")	
            input2=F.to_pil_image(r.astype("int32"),"I")	


            if random.random() > 0.5:	
                input2 = F.hflip(input2)	
                target = F.hflip(target)	
            if np.random.rand() < 0:	
                affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), 
    
class fudandataset(data.Dataset):
    def __init__(self,root,train=True):
        self.root = root
        self.train = train
        if self.train:
            print('loading training data')
            self.train_data = []
            self.train_labels = []
            self.save1_data=[]
            self.save_labels=[]
            files = os.listdir(root)
            files.sort()
            for file_name in files:
                if 'manual' in file_name:
                    file_path = os.path.join(self.root,file_name)
                    file_data = nib.load(file_path)
                    file_data = file_data.get_data()
                    d = file_data.shape[2]
                    for i in range(d):
                        labels = copy.deepcopy(file_data[:,:,i])
                        labels[labels==200]=1
                        labels[labels==500]=2
                        labels[labels==600]=3
                        x= labels.shape[1]
                        x= int(0.3*x)
                        self.save_labels.append(labels[x:256+x, x:256+x])
                        self.train_labels.append(labels[x:256+x, x:256+x])
                        
                else:
                    file_path = os.path.join(self.root,file_name)
                    file_data = nib.load(file_path)
                    file_data = file_data.get_data()
                    d = file_data.shape[2]
                    for i in range(d):
                        data1 = copy.deepcopy(file_data[:,:,i])
                        x= data1.shape[1]
                        x= int(0.3*x)
                        
                        data = copy.deepcopy(data1[x:256+x, x:256+x])
                        self.save1_data.append(data)
                        self.train_data.append(data[:,:,np.newaxis].transpose(2,0,1))
                       
            for i in range(10):
                test1,label1=my_segmentation_transform(self.save1_data,self.save_labels)
                self.train_data.extend(test1)
                self.train_labels.extend(label1)                   
        else:
            print('loading test data ')
            self.test_data = [] 
            self.test_labels = []
                  
            files = os.listdir(root)
            files.sort()
            for file_name in files:
                if 'manual' in file_name:
                    file_path = os.path.join(self.root,file_name)
                    file_data = nib.load(file_path)
                    file_data = file_data.get_data()
                    d = file_data.shape[2]
                    for i in range(d):
                        labels = copy.deepcopy(file_data[:,:,i])
                        labels[labels==200]=1
                        labels[labels==500]=2
                        labels[labels==600]=3
                        x= labels.shape[1]
                        x= int(0.3*x)
                        self.test_labels.append(labels[x:256+x, x:256+x])
                        
                else:
                    file_path = os.path.join(self.root,file_name)
                    file_data = nib.load(file_path)
                    file_data = file_data.get_data()
                    d = file_data.shape[2]
                    for i in range(d):
                        data1 = copy.deepcopy(file_data[:,:,i])
                        x= data1.shape[1]
                        x= int(0.3*x)
                        data = copy.deepcopy(data1[x:256+x, x:256+x])
                        self.test_data.append(data[:,:,np.newaxis].transpose(2,0,1)) #.transpose(2,0,1)
                        
         
                        
    
    
    def __getitem__(self, index):
        if self.train:
    	    slices, label = self.train_data[index], self.train_labels[index] 
        else:
            slices, label = self.test_data[index], self.test_labels[index]
            
        return torch.from_numpy(slices).float(), torch.from_numpy(label).float(),
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
                            
                    
            
