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
from torchvision import transforms as T
import torchvision.transforms.functional as F
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
                    for i in range(d):
                        labels = copy.deepcopy(file_data[:,:,i])
                        labels[labels==200]=0
                        labels[labels==500]=1
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
                    for i in range(d):
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
            '''for j in range(10):
                if j<9:
                   for i in range(len(self.train_data1)):
                        to_pil_image = T.ToPILImage()  
                        image=to_pil_image(self.train_data1[i])
                        segmentation=to_pil_image(self.train_labels1[i])
                        if random.random()>0.5:
                            angle = np.random.randint(-30, 30)
                            image = F.rotate(image, angle)
                            segmentation = F.rotate(segmentation, angle)
                        if random.random()>0.5:
                            positionx = np.random.random()
                            positiony = np.random.random()
                            image = F.affine(image, angle=0,translate=[128*positionx,128*positiony],scale=1,shear=0)
                            segmentation = F.affine(segmentation, angle=0,translate=[128*positionx,128*positiony],scale=1,shear=0)
                        if random.random()>0.5:
                            image = F.hflip(image)
                            segmentation = F.hflip(segmentation)
                        
                            
                        image=np.array(image, dtype=np.float32)
                        segmentation=np.array(segmentation, dtype=np.float32)
                        self.train_data.append(image[:,:,np.newaxis].transpose(2,0,1))
                        self.train_labels.append(segmentation)  
                else:
                     for i in range(len(self.train_data1)):
                         data2=self.train_data1[i]
                         label2=self.train_labels1[i]
                         label2[label2==200]=0
                         label2[label2==500]=1
                         label2[label2==600]=0
                         self.train_data.append(data2[:,:,np.newaxis].transpose(2,0,1))
                         self.train_labels.append(label2)'''
                                
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
                    for i in range(d):
                        labels = file_data[:,:,i]
                        labels[labels==200]=0
                        labels[labels==500]=1
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
                    for i in range(d):
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
                            
                    
            
