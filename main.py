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
import numpy as np
from fudandataset import fudandataset
from Unet import UNet

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
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in optimizer')
parser.add_argument('-bs', '--batchsize', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=600, help='epochs to train')
parser.add_argument('-out', '--outf', type=str, default='./model_checkpoint', help='path to save model checkpoints')
config = parser.parse_args()
num_classes = 3

train_dataset = fudandataset(traindata_root,train=True)
test_dataset = fudandataset(testdata_root,train=False)

traindataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16*(config.batchsize), shuffle=True, 
                                              num_workers=4)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batchsize, shuffle=True, 
                                              num_workers=4)
#seed = 123456
#random.seed(seed)
#torch.cuda.manual_seed(seed)

classifier = UNet(n_classes = num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)
optimizer = optim.Adam(classifier.parameters(), lr=config.lr,weight_decay = 1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
train_acc_epoch, test_acc_epoch ,train_loss_epoch,test_loss_epoch= [], [],[],[]
weight1 = torch.Tensor([1,6,6])
weight1=weight1.to(device)
output = nn.CrossEntropyLoss(weight=weight1)

print ('Starting training...\n')
for epoch in range(config.epochs):
    log_string('**** EPOCH %03d ****' % (epoch+1))
    log_string(str(datetime.now()))
    train_acc_epoch, test_acc_epoch = [], []
    for i, data in enumerate(traindataloader):
        slices,label = data
        slices, label = slices.to(device), label.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred = classifier(slices)
        pred = pred.view(-1, num_classes)
        label = label.view(-1).long()
        #loss = F.cross_entropy(pred, label)
        loss = output(pred, label)
                
        #print(pred.size(),label.size())
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(label.data).cpu().sum()
        train_acc = correct.item()/float(label.shape[0])
        
        log_string(' -- %03d / %03d --' % (epoch+1, 1))
        log_string('loss: %f' % (loss.item()))
        log_string('accuracy: %f' % (train_acc))
        train_acc_epoch.append(train_acc)
        train_loss_epoch.append(loss.item())
        if (i+1) % 10 == 0:
            log_string(str(datetime.now()))
            log_string('---- EPOCH %03d EVALUATION ----'%(epoch+1))
            for j, data in enumerate(testdataloader):
                slices,label = data
                slices, label = slices.to(device), label.to(device)
                #slices = slices.transpose(2, 0, 1)
                classifier = classifier.eval()
                pred = classifier(slices)
                pred = pred.view(-1, num_classes)
                label = label.view(-1).long()
                #loss = F.cross_entropy(pred, label)
                loss = output(pred, label)
                
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(label.data).cpu().sum()
                test_acc = correct.item()/float(label.shape[0])
               
                test_acc_epoch.append(test_acc)
                test_loss_epoch.append(loss.item())
                log_string(' -- %03d / %03d --' % (epoch+1, 1))
                log_string('loss: %f' % (loss.item()))
                log_string('accuracy: %f' % (test_acc))
    print('**** EPOCH %03d ****' % (epoch+1))
    print(str(datetime.now()))        
    print(('epoch %d | mean train acc: %f') % (epoch+1, np.mean(train_acc_epoch)))
    print(('epoch %d | mean test acc: %f') % (epoch+1, np.mean(test_acc_epoch)))
    print(('epoch %d | mean train loss: %f') % (epoch+1, np.mean(train_loss_epoch)))
    print(('epoch %d | mean test loss: %f') % (epoch+1, np.mean(test_loss_epoch)))
    torch.save(classifier.state_dict(), '%s/%s_model_%d.pth' % (config.outf, 'fudanc0', epoch))
