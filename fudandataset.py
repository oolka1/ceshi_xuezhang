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
import random
import copy
from PIL import Image
class fudandataset(data.Dataset):
    def __init__(self,root,train=True):
        self.root = root
        self.train = train
        if self.train:
            print('loading training data')
            self.train_data = []
            self.train_labels = []
            self.train_data1 = []
            self.train_labels1 = []
            files = os.listdir(root)
            files.sort()
            for file_name in files:
                if 'manual' in file_name:
                    file_path = os.path.join(self.root,file_name)
                    file_data = nib.load(file_path)
                    file_data = file_data.get_data()
                    d = file_data.shape[2]
                    for i in range(2,d):
                        labels = copy.deepcopy(file_data[:,:,i])
                        labels[labels==200]=1
                        labels[labels==500]=2
                        labels[labels==600]=0
                        x=labels.shape[0]
                        img=Image.fromarray(np.uint8(labels))
                        img1=img.resize((256, 256))
                        x=256
                        labels = np.array(img1)
                        x1=int(0.25*x)
                        labels=labels[x1:x1+128,]
                        labels=labels[:,x1:x1+128]
                        self.train_labels.append(labels)
                else:
                    file_path = os.path.join(self.root,file_name)
                    file_data = nib.load(file_path)
                    file_data = file_data.get_data()
                    d = file_data.shape[2]
                    for i in range(2,d):
                        data = copy.deepcopy(file_data[:,:,i])
                        img=Image.fromarray(np.int32(data))
                        img1=img.resize((256, 256))
                        data = np.array(img1)
                        x=data.shape[0]
                        x=256
                        x1=int(0.25*x)
                        data=data[x1:x1+128,]
                        data=data[:,x1:x1+128]
                        data=data.astype(np.float32)
                        max1=data.max()
                        max1=max1.astype(np.float32)
                        data=data/max1  
                        self.train_data.append(data[:,:,np.newaxis].transpose(2,0,1))
            
            self.together=list(zip(self.train_data,self.train_labels))          
            random.shuffle(self.together)
            self.train_data,self.train_labels = zip(*self.together)
            print(len(self.train_data))            
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
                    for i in range(2,d):
                        labels = file_data[:,:,i]
                        labels[labels==200]=1
                        labels[labels==500]=2
                        labels[labels==600]=0
                        x=labels.shape[0]
                        
                        img=Image.fromarray(np.uint8(labels))
                        img1=img.resize((256, 256))
                        labels = np.array(img1)
                        x=256
                        x1=int(0.25*x)
                        labels=labels[x1:x1+128,]
                        labels=labels[:,x1:x1+128]
                        self.test_labels.append(labels)
                else:
                    file_path = os.path.join(self.root,file_name)
                    file_data = nib.load(file_path)
                    file_data = file_data.get_data()
                    d = file_data.shape[2]
                    for i in range(2,d):
                        data = file_data[:,:,i]
                        x=data.shape[0]
                        
                        img=Image.fromarray(np.int32(data))
                        img1=img.resize((256, 256))
                        data = np.array(img1)
                        x=256
                        x1=int(0.25*x)
                        data=data[x1:x1+128,]
                        data=data[:,x1:x1+128]
                        data=data.astype(np.float32)
                        max1=data.max()
                        max1=max1.astype(np.float32)
                        data=data/max1  
                        self.test_data.append(data[:,:,np.newaxis].transpose(2,0,1))
                        
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
                            
                    
            
