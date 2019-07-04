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
class fudandataset(data.Dataset):
    def __init__(self,root,train=True):
        self.root = root
        self.train = train
        if self.train:
            print('loading training data')
            self.train_data = []
            self.train_labels = []
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
                        labels[labels==200]=0
                        labels[labels==500]=1
                        labels[labels==600]=0
                        x=labels.shape[0]
                        y=int(0.39*x)
                        x=int(0.37*x)
                        labels=labels[y:y+128,]
                        labels=labels[:,x:x+128]  
                        self.train_labels.append(labels)
                else:
                    file_path = os.path.join(self.root,file_name)
                    file_data = nib.load(file_path)
                    file_data = file_data.get_data()
                    d = file_data.shape[2]
                    for i in range(2,d):
                        data = file_data[:,:,i]
                        x=data.shape[0]
                        y=int(0.39*x)
                        x=int(0.37*x)
                        data=data[y:y+128,]
                        data=data[:,x:x+128]
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
                        labels[labels==200]=0
                        labels[labels==500]=1
                        labels[labels==600]=0
                        x=labels.shape[0]
                        y=int(0.39*x)
                        x=int(0.37*x)
                        labels=labels[y:y+128,]
                        labels=labels[:,x:x+128] 
                        self.test_labels.append(labels)
                else:
                    file_path = os.path.join(self.root,file_name)
                    file_data = nib.load(file_path)
                    file_data = file_data.get_data()
                    d = file_data.shape[2]
                    for i in range(2,d):
                        data = file_data[:,:,i]
                        x=data.shape[0]
                        y=int(0.39*x)
                        x=int(0.37*x)
                        data=data[y:y+128,]
                        data=data[:,x:x+128]
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
                            
                    
            
