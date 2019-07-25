import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class DiceLoss(nn.Module):
    def __init__(self, class_num=4,smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self,input, target):
        input = torch.exp(input)
        self.smooth = 1
        Dice = (torch.Tensor([0]).float()).cuda()
        for i in range(1,self.class_num):
            input_i = input[:,i,:,:]
            target_i = (target == i).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = (torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice/(self.class_num - 1)
        return dice_loss

class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=4,smooth=1,gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self,input, target):
        input = torch.exp(input)
        self.smooth = 1
        Dice = (torch.Tensor([0]).float()).cuda()
        for i in range(1,self.class_num):
            input_i = input[:,i,:,:]
            target_i = (target == i).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = (torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice))**self.gamma
        dice_loss = Dice/(self.class_num - 1)
        return dice_loss
