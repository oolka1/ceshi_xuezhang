    
import _init_paths
import torch
import torch.nn as nn
from layers import unetConv2, unetUp
from utils import init_weights, count_param

class UNet_Nested(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        filters = [36, 72, 144, 288, 576,1152,2304]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.conv50 = unetConv2(filters[4], filters[5], self.is_batchnorm)
        self.conv60 = unetConv2(filters[5], filters[6], self.is_batchnorm)
        
        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat41 = unetUp(filters[5], filters[4], self.is_deconv)
        self.up_concat51 = unetUp(filters[6], filters[5], self.is_deconv)
        
        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)
        self.up_concat32 = unetUp(filters[4], filters[3], self.is_deconv, 3)
        self.up_concat42 = unetUp(filters[5], filters[4], self.is_deconv, 3)


        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)
        self.up_concat23 = unetUp(filters[3], filters[2], self.is_deconv, 4)
        self.up_concat33 = unetUp(filters[4], filters[3], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)
        self.up_concat14 = unetUp(filters[2], filters[1], self.is_deconv, 5)
        self.up_concat24 = unetUp(filters[3], filters[2], self.is_deconv, 5) 
        
        
        self.up_concat05 = unetUp(filters[1], filters[0], self.is_deconv, 6)
        self.up_concat15 = unetUp(filters[2], filters[1], self.is_deconv, 6)
        
        self.up_concat06 = unetUp(filters[1], filters[0], self.is_deconv, 7)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_5 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_6 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)       # 16*512*512
        maxpool0 = self.maxpool(X_00)    # 16*256*256
        X_10= self.conv10(maxpool0)      # 32*256*256
        maxpool1 = self.maxpool(X_10)    # 32*128*128
        X_20 = self.conv20(maxpool1)     # 64*128*128
        maxpool2 = self.maxpool(X_20)    # 64*64*64
        X_30 = self.conv30(maxpool2)     # 128*64*64
        maxpool3 = self.maxpool(X_30)    # 128*32*32
        X_40 = self.conv40(maxpool3)     # 256*32*32
        maxpool4 = self.maxpool(X_40)    # 128*32*32
        X_50 = self.conv50(maxpool4)     # 256*32*32
        maxpool5 = self.maxpool(X_50)    # 128*32*32
        X_60 = self.conv60(maxpool5)     # 256*32*32
        
        nn.Dropout(0.5)
        # column : 1
        X_01 = self.up_concat01(X_10,X_00)
        X_11 = self.up_concat11(X_20,X_10)
        X_21 = self.up_concat21(X_30,X_20)
        X_31 = self.up_concat31(X_40,X_30)
        X_41 = self.up_concat41(X_50,X_40)
        X_51 = self.up_concat51(X_60,X_50)
        
       
        # column : 2
        X_02 = self.up_concat02(X_11,X_00,X_01)
        X_12 = self.up_concat12(X_21,X_10,X_11)
        X_22 = self.up_concat22(X_31,X_20,X_21)
        X_32 = self.up_concat32(X_41,X_30,X_31)
        X_42 = self.up_concat42(X_51,X_40,X_41)
        
        
        # column : 3
        X_03 = self.up_concat03(X_12,X_00,X_01,X_02)
        X_13 = self.up_concat13(X_22,X_10,X_11,X_12)
        X_23 = self.up_concat23(X_32,X_20,X_21,X_22)
        X_33 = self.up_concat33(X_42,X_30,X_31,X_32)
               
        # column : 4
        X_04 = self.up_concat04(X_13,X_00,X_01,X_02,X_03)
        X_14 = self.up_concat14(X_23,X_10,X_11,X_12,X_13)
        X_24 = self.up_concat24(X_33,X_20,X_21,X_22,X_23)
        
        X_05 = self.up_concat05(X_14,X_00,X_01,X_02,X_03,X_04)
        X_15 = self.up_concat15(X_24,X_10,X_11,X_12,X_13,X_14)
               
        X_06 = self.up_concat06(X_15,X_00,X_01,X_02,X_03,X_04,X_05)
                
        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)
        final_5 = self.final_4(X_05)
        final_6 = self.final_4(X_06)

        final = (final_1+final_2+final_3+final_4+final_5+final_6)/6

        if self.is_ds:
            return final
        else:
            return final_4

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(2,1,64,64)).cuda()
    model = UNet_Nested().cuda()
    param = count_param(model)
    y = model(x)
    print('Output shape:',y.shape)
    print('UNet++ totoal parameters: %.2fM (%d)'%(param/1e6,param))
