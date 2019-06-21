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
import copy

traindata_root = "train"
testdata_root = "test"
log_root = "log"
if not os.path.exists(log_root): os.mkdir(log_root)
LOG_FOUT = open(os.path.join(log_root, 'train.log'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
os.system('mkdir {0}'.format('model_checkpoint'))

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in optimizer')
parser.add_argument('-bs', '--batchsize', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=500, help='epochs to train')
parser.add_argument('-out', '--outf', type=str, default='./model_checkpoint', help='path to save model checkpoints')
config = parser.parse_args()
num_classes = 4

train_dataset = fudandataset(traindata_root,train=True)
test_dataset = fudandataset(testdata_root,train=False)
train_datasetmask0=[]
train_datasetimg0=[]
train_datasetmask0001=[]
train_datasetimg0001=[]
train_datasetmask001=[]
train_datasetimg001=[]
train_datasetmask01=[]
train_datasetimg01=[]
label1=copy.deepcopy(train_dataset.train_labels)
img1=copy.deepcopy(train_dataset.train_data)
for i in range(len(label1)):
  r1=label1[i]
  r2=img1[i]
  maskzero=np.sum(r1==0)
  m=float(maskzero/r1.size)
  m=1-m
  if m==0:
    train_datasetmask0.append(r1)
    train_datasetimg0.append(r2)
  else:
    if m<0.001:
      train_datasetmask0001.append(r1)
      train_datasetimg0001.append(r2)
    else:
      if m<0.01:
        train_datasetmask001.append(r1)
        train_datasetimg001.append(r2)
      else:
        train_datasetmask01.append(r1)
        train_datasetimg01.append(r2)
train_dataset0=copy.deepcopy(train_dataset)
train_dataset0001=copy.deepcopy(train_dataset)
train_dataset001=copy.deepcopy(train_dataset)
train_dataset01=copy.deepcopy(train_dataset)
train_dataset0.train_labels=train_datasetmask0
train_dataset0.train_data=train_datasetimg0
train_dataset0001.train_labels=train_datasetmask0001
train_dataset0001.train_data=train_datasetimg0001
train_dataset001.train_labels=train_datasetmask001
train_dataset001.train_data=train_datasetimg001
train_dataset01.train_labels=train_datasetmask01
train_dataset01.train_data=train_datasetimg01

traindataloader0 = torch.utils.data.DataLoader(train_dataset0, batch_size=config.batchsize, shuffle=True, num_workers=4)
traindataloader0001 = torch.utils.data.DataLoader(train_dataset0001, batch_size=config.batchsize, shuffle=True, num_workers=4)
traindataloader001 = torch.utils.data.DataLoader(train_dataset001, batch_size=config.batchsize, shuffle=True,  num_workers=4)
traindataloader01 = torch.utils.data.DataLoader(train_dataset01, batch_size=config.batchsize, shuffle=True, num_workers=4)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batchsize, shuffle=True,  num_workers=4)
#seed = 123456
#random.seed(seed)
#torch.cuda.manual_seed(seed)

classifier = UNet(n_classes = num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)
optimizer = optim.Adam(classifier.parameters(), lr=config.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

#loss = nn.CrossEntropyLoss()
weight1 = torch.Tensor([1,1,1,1])
weight1 = weight1.to(device)
weight2 = torch.Tensor([1,1000,1000,1000])
weight2 = weight2.to(device)
weight3 = torch.Tensor([1,200,200,200])
weight3 = weight3.to(device)
weight4 = torch.Tensor([1,100,100,100])
weight4 = weight4.to(device)
print (config.epochs)
print ('Starting training...\n')
for epoch in range(config.epochs):
    log_string('**** EPOCH %03d ****' % (epoch+1))
    log_string(str(datetime.now()))
    train_acc_epoch, test_acc_epoch = [], []
    if epoch<250:
        for i, data in enumerate(traindataloader01):
            slices,label = data
            slices, label = slices.to(device), label.to(device)
            optimizer.zero_grad()
            classifier = classifier.train()
            pred = classifier(slices)
            pred = pred.view(-1, num_classes)
            label = label.view(-1).long()
            loss = nn.CrossEntropyLoss( weight=weight4)
            output = loss(pred, label)
            #print(pred.size(),label.size())
            output.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(label.data).cpu().sum()
            train_acc = correct.item()/float(label.shape[0])
            train_acc_epoch.append(train_acc)
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
                    output = loss(pred, label)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(label.data).cpu().sum()
                    test_acc = correct.item()/float(label.shape[0])
                    test_acc_epoch.append(test_acc)
                print('epoch %d: %d | test loss: %f | test acc: %f'
                % (epoch+1, i+1, output.item(), test_acc))
                log_string(' -- %03d / %03d --' % (epoch+1, 1))
                log_string('loss: %f' % (output.item()))
                log_string('accuracy: %f' % (test_acc))

        print('epoch %d: %d | train loss: %f | train acc: %f'
        % (epoch+1, i+1, output.item(), train_acc))
        log_string(' -- %03d / %03d --' % (epoch+1, 1))
        log_string('loss: %f' % (output.item()))
        log_string('accuracy: %f' % (train_acc))


        print(('epoch %d | mean train acc: %f') % (epoch+1, np.mean(train_acc_epoch)))
        print(('epoch %d | mean test acc: %f') % (epoch+1, np.mean(test_acc_epoch)))
        torch.save(classifier.state_dict(), '%s/%s_model_%d.pth' % (config.outf, 'fudanc0', epoch))
    else:
        if epoch<350:
            for i, data in enumerate(traindataloader001):
                slices,label = data
                slices, label = slices.to(device), label.to(device)
                optimizer.zero_grad()
                classifier = classifier.train()
                pred = classifier(slices)
                pred = pred.view(-1, num_classes)
                label = label.view(-1).long()
                loss = nn.CrossEntropyLoss( weight=weight3)
                output = loss(pred, label)
                #print(pred.size(),label.size())
                output.backward()
                optimizer.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(label.data).cpu().sum()
                train_acc = correct.item()/float(label.shape[0])
                train_acc_epoch.append(train_acc)
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
                        output = loss(pred, label)
                        pred_choice = pred.data.max(1)[1]
                        correct = pred_choice.eq(label.data).cpu().sum()
                        test_acc = correct.item()/float(label.shape[0])
                        test_acc_epoch.append(test_acc)
                    print('epoch %d: %d | test loss: %f | test acc: %f'
                    % (epoch+1, i+1, output.item(), test_acc))
                    log_string(' -- %03d / %03d --' % (epoch+1, 1))
                    log_string('loss: %f' % (output.item()))
                    log_string('accuracy: %f' % (test_acc))

            print('epoch %d: %d | train loss: %f | train acc: %f'
            % (epoch+1, i+1, output.item(), train_acc))
            log_string(' -- %03d / %03d --' % (epoch+1, 1))
            log_string('loss: %f' % (output.item()))
            log_string('accuracy: %f' % (train_acc))


            print(('epoch %d | mean train acc: %f') % (epoch+1, np.mean(train_acc_epoch)))
            print(('epoch %d | mean test acc: %f') % (epoch+1, np.mean(test_acc_epoch)))
            torch.save(classifier.state_dict(), '%s/%s_model_%d.pth' % (config.outf, 'fudanc0', epoch)) 
        else:
            if epoch<450:
                for i, data in enumerate(traindataloader0001):
                    slices,label = data
                    slices, label = slices.to(device), label.to(device)
                    optimizer.zero_grad()
                    classifier = classifier.train()
                    pred = classifier(slices)
                    pred = pred.view(-1, num_classes)
                    label = label.view(-1).long()
                    loss = nn.CrossEntropyLoss( weight=weight2)
                    output = loss(pred, label)
                    #print(pred.size(),label.size())
                    output.backward()
                    optimizer.step()
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(label.data).cpu().sum()
                    train_acc = correct.item()/float(label.shape[0])
                    train_acc_epoch.append(train_acc)
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
                            output = loss(pred, label)
                            pred_choice = pred.data.max(1)[1]
                            correct = pred_choice.eq(label.data).cpu().sum()
                            test_acc = correct.item()/float(label.shape[0])
                            test_acc_epoch.append(test_acc)
                        print('epoch %d: %d | test loss: %f | test acc: %f'
                        % (epoch+1, i+1, output.item(), test_acc))
                        log_string(' -- %03d / %03d --' % (epoch+1, 1))
                        log_string('loss: %f' % (output.item()))
                        log_string('accuracy: %f' % (test_acc))

                print('epoch %d: %d | train loss: %f | train acc: %f'
                % (epoch+1, i+1, output.item(), train_acc))
                log_string(' -- %03d / %03d --' % (epoch+1, 1))
                log_string('loss: %f' % (output.item()))
                log_string('accuracy: %f' % (train_acc))


                print(('epoch %d | mean train acc: %f') % (epoch+1, np.mean(train_acc_epoch)))
                print(('epoch %d | mean test acc: %f') % (epoch+1, np.mean(test_acc_epoch)))
                torch.save(classifier.state_dict(), '%s/%s_model_%d.pth' % (config.outf, 'fudanc0', epoch))
            else:
                for i, data in enumerate(traindataloader0):
                    slices,label = data
                    slices, label = slices.to(device), label.to(device)
                    optimizer.zero_grad()
                    classifier = classifier.train()
                    pred = classifier(slices)
                    pred = pred.view(-1, num_classes)
                    label = label.view(-1).long()
                    loss = nn.CrossEntropyLoss( weight=weight1)
                    output = loss(pred, label)
                    #print(pred.size(),label.size())
                    output.backward()
                    optimizer.step()
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(label.data).cpu().sum()
                    train_acc = correct.item()/float(label.shape[0])
                    train_acc_epoch.append(train_acc)
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
                            output = loss(pred, label)
                            pred_choice = pred.data.max(1)[1]
                            correct = pred_choice.eq(label.data).cpu().sum()
                            test_acc = correct.item()/float(label.shape[0])
                            test_acc_epoch.append(test_acc)
                        print('epoch %d: %d | test loss: %f | test acc: %f'
                        % (epoch+1, i+1, output.item(), test_acc))
                        log_string(' -- %03d / %03d --' % (epoch+1, 1))
                        log_string('loss: %f' % (output.item()))
                        log_string('accuracy: %f' % (test_acc))

                print('epoch %d: %d | train loss: %f | train acc: %f'
                % (epoch+1, i+1, output.item(), train_acc))
                log_string(' -- %03d / %03d --' % (epoch+1, 1))
                log_string('loss: %f' % (output.item()))
                log_string('accuracy: %f' % (train_acc))


                print(('epoch %d | mean train acc: %f') % (epoch+1, np.mean(train_acc_epoch)))
                print(('epoch %d | mean test acc: %f') % (epoch+1, np.mean(test_acc_epoch)))
                torch.save(classifier.state_dict(), '%s/%s_model_%d.pth' % (config.outf, 'fudanc0', epoch))                          
