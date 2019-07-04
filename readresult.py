#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:14:02 2019

@author: hesun
"""

from PIL import Image
import numpy as np
from torchvision import transforms
import os

data_root = "/home/hesun/phd/medical image/unet_test/result"
save_root = "/home/hesun/phd/medical image/unet_test/pre0528"
if not os.path.exists(save_root): os.mkdir(save_root)

label = np.load("/home/hesun/phd/medical image/unet_test/result/label_0.npy")
mask = np.load("/home/hesun/phd/medical image/unet_test/result/mask_0.npy")
raw = np.load("/home/hesun/phd/medical image/unet_test/result/raw_0.npy")

label[label==1]=200
label[label==2]=500
label[label==3]=600
mask1 = np.reshape(mask,(label.shape[1],label.shape[2]))
mask1[mask1==1]=200
mask1[mask1==2]=500
mask1[mask1==3]=600
#unloader = transforms.ToPILImage()
#def tensor_to_PIL(tensor):
#    image = tensor.cpu().clone()
#    image = image.squeeze(0)
#    image = unloader(image)
#    return image

#mm = mask.squeeze(0)
#mm = unloader(mm)
i = 0
result = Image.fromarray(np.uint8(mask1))
result.show()
#result.save('%s/result_%d.png' % (save_root,i))

gt = Image.fromarray(np.uint8(label[0]))
gt.show()
#gt.save('%s/gt_%d.png' % (save_root,i))

rawdata = Image.fromarray(np.uint8(raw[0,0,:,:]))
rawdata.show()
#rawdata.save('%s/rawdata_%d.png' % (save_root,i))
