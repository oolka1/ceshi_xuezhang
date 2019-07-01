#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:55:36 2019

@author: hesun
"""
from datetime import datetime
import argparse
import os
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
#from torchnet import meter
import torchvision.transforms.functional as f
import numpy as np
from fudandataset import fudandataset
from Unet import UNet_Nested
import copy

traindata_root = "train"
testdata_root = "test"
log_root = "log"
if not os.path.exists(log_root): os.mkdir(log_root)
LOG_FOUT = open(os.path.join(log_root, 'train.log'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()

os.system('mkdir {0}'.format('model_checkpoint'))

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in optimizer')
parser.add_argument('-bs', '--batchsize', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=400, help='epochs to train')
parser.add_argument('-out', '--outf', type=str, default='./model_checkpoint', help='path to save model checkpoints')
config = parser.parse_args()
num_classes = 4

train_dataset = fudandataset(traindata_root,train=True)

val_dataset=fudandataset(testdata_root,train=False)
#seed = 123456
#random.seed(seed)
#torch.cuda.manual_seed(seed)

classifier = UNet_Nested(n_classes = num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)
lr=config.lr
optimizer = optim.Adam(classifier.parameters(), lr=lr,weight_decay = 3e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

traindataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True, num_workers=4)
valdataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batchsize, shuffle=True,  num_workers=4)
#loss = nn.CrossEntropyLoss()

#loss_meter = meter.AverageValueMeter()
#confusion_matrix = meter.ConfusionMeter(4)
previous_loss = 1e100	
loss_stroge=0
weight1 = torch.Tensor([1,30,30,30])
weight1 = weight1.to(device)	
#loss=nn.CrossEntropyLoss(weight=weight1)
loss=nn.cross_entropy()
print (config.epochs)
print ('Starting training...\n')
for epoch in range(config.epochs):
    log_string('**** EPOCH %03d ****' % (epoch+1))
    log_string(str(datetime.now()))
    print('**** EPOCH %03d ****' % (epoch+1))

    print(str(datetime.now()))
    train_acc_epoch, val_acc_epoch ,train_loss_epoch,val_loss_epoch= [], [],[],[]
    
    #loss_meter.reset()
    #confusion_matrix.reset()         
    for i, data in enumerate(traindataloader): 
        slices, label = data    
        slices, label = slices.to(device), label.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred = classifier(slices)
        pred = pred.view(-1, num_classes)
        label = label.view(-1).long()
        output =  loss(pred, label)#weight=weight1
       
        #print(pred.size(),label.size())
        output.backward()
        optimizer.step()
           
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(label.data).cpu().sum()
        train_acc = correct.item()/float(label.shape[0])
        train_acc_epoch.append(train_acc)
        train_loss_epoch.append(output.item())
        log_string(' -- %03d / %03d --' % (epoch+1, 1))
        log_string('train_loss: %f' % (output.item()))
        log_string('train_accuracy: %f' % (train_acc))
        
        if (i+1) % 10 == 0:
            log_string(str(datetime.now()))
            log_string('---- EPOCH %03d EVALUATION ----'%(epoch+1))
            for j, data in enumerate(valdataloader):
                slices,label = data
                slices, label = slices.to(device), label.to(device)
                #slices = slices.transpose(2, 0, 1)
                classifier = classifier.eval()
                pred = classifier(slices)
                pred = pred.view(-1, num_classes)
                label = label.view(-1).long()
                output = loss(pred, label)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(label.data).cpu().sum()
                val_acc = correct.item()/float(label.shape[0])
                val_acc_epoch.append(val_acc)
                val_loss_epoch.append(output.item())
                log_string(' -- %03d / %03d --' % (epoch+1, 1))
                log_string('val_loss: %f' % (output.item()))
                log_string('val_accuracy: %f' % (val_acc))
            
    #print("train loss:",loss_stroge[0])
    #print("train acc:", train_acc[0])
    print(('epoch %d | mean train acc: %f') % (epoch+1, np.mean(train_acc_epoch)))
    print(('epoch %d | mean test acc: %f') % (epoch+1, np.mean(val_acc_epoch)))
    print(('epoch %d | mean train loss: %f') % (epoch+1, np.mean(train_loss_epoch)))
    print(('epoch %d | mean test loss: %f') % (epoch+1, np.mean(val_loss_epoch)))
    print(' ')
    loss_stroge = np.mean(train_loss_epoch)
    torch.save(classifier.state_dict(), '%s/%s_model_%d.pth' % (config.outf, 'fudanc0', epoch))
    if loss_stroge > previous_loss:          	
         lr = lr * 0.9	
         for param_group in optimizer.param_groups:	
             param_group['lr'] = lr               	
    previous_loss = loss_stroge
    '''if loss_stroge[0] > previous_loss:          
        lr = lr * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr               
    previous_loss = loss_stroge[0] '''   
