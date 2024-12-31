import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, PWConvBNAct, ConvBNAct, PyramidPoolingModule
from backbone import ResNet, Mobilenetv2


class SwiftNet(nn.Module):
    def __init__(self, num_class=4, n_channel=3, backbone_type='resnet18', up_channels=128, 
                    act_type='relu'):
        super(SwiftNet, self).__init__()
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [64, 128, 256, 512] if backbone_type in ['resnet18', 'resnet34'] else [256, 512, 1024, 2048]
        elif backbone_type == 'mobilenet_v2':
            self.backbone = Mobilenetv2()
            channels = [24, 32, 96, 320]
        else:
            raise NotImplementedError()

        self.connection1 = ConvBNAct(channels[0], up_channels, 1, act_type=act_type)
        self.connection2 = ConvBNAct(channels[1], up_channels, 1, act_type=act_type)
        self.connection3 = ConvBNAct(channels[2], up_channels, 1, act_type=act_type)
        self.spp = PyramidPoolingModule(channels[3], up_channels, act_type, bias=True)
        self.decoder = Decoder(up_channels, num_class, act_type)

    def forward(self, x):
        size = x.size()[2:]

        x1, x2, x3, x4 = self.backbone(x)

        x1 = self.connection1(x1)
        x2 = self.connection2(x2)
        x3 = self.connection3(x3)
        x4 = self.spp(x4)

        x = self.decoder(x4, x1, x2, x3)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        output = F.softmax(x, dim=1)

        return output


class Decoder(nn.Module):
    def __init__(self, channels, num_class, act_type):
        super(Decoder, self).__init__()
        self.up_stage3 = ConvBNAct(channels, channels, 3, act_type=act_type)
        self.up_stage2 = ConvBNAct(channels, channels, 3, act_type=act_type)
        self.up_stage1 = ConvBNAct(channels, num_class, 3, act_type=act_type)
        
    def forward(self, x, x1, x2, x3):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x += x3
        x = self.up_stage3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x += x2
        x = self.up_stage2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x += x1
        x = self.up_stage1(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import DWConvBNAct, PWConvBNAct, ConvBNAct, DeConvBNAct, Activation
from enet import InitialBlock as DownsamplingUnit


class MiniNetv2(nn.Module):
    def __init__(self, num_class=4, n_channel=3, feat_dt=[1,2,1,4,1,8,1,16,1,1,1,2,1,4,1,8],
                    act_type='relu'):
        super(MiniNetv2, self).__init__()
        self.d1_2 = nn.Sequential(
                        DownsamplingUnit(n_channel, 16, act_type),
                        DownsamplingUnit(16, 64, act_type),
                    )
        self.ref = nn.Sequential(
                        DownsamplingUnit(n_channel, 16, act_type),
                        DownsamplingUnit(16, 64, act_type)
                    )
        self.m1_10 = build_blocks(MultiDilationDSConv, 64, 10, act_type=act_type)
        self.d3 = DownsamplingUnit(64, 128, act_type)
        self.feature_extractor = build_blocks(MultiDilationDSConv, 128, len(feat_dt), feat_dt, act_type)
        self.up1 = DeConvBNAct(128, 64, act_type=act_type)
        self.m26_29 = build_blocks(MultiDilationDSConv, 64, 4, act_type=act_type)
        self.output = DeConvBNAct(64, num_class, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]

        x_ref = self.ref(x)

        x = self.d1_2(x)
        x = self.m1_10(x)
        x = self.d3(x)
        x = self.feature_extractor(x)
        x = self.up1(x)
        # x += x_ref
        x = x + x_ref

        x = self.m26_29(x)
        x = self.output(x)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        output = F.softmax(x, dim=1)

        return output


def build_blocks(block, channels, num_block, dilations=[], act_type='relu'):
    if len(dilations) == 0:
        dilations = [1 for _ in range(num_block)]
    else:
        if len(dilations) != num_block:
            raise ValueError(f'Number of dilation should be equal to number of blocks')

    layers = []
    for i in range(num_block):
        layers.append(block(channels, channels, 3, 1, dilations[i], act_type))
    return  nn.Sequential(*layers)


class MultiDilationDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, act_type='relu'):
        super(MultiDilationDSConv, self).__init__()
        self.dilated = dilation > 1
        self.dw_conv = DWConvBNAct(in_channels, in_channels, kernel_size, stride, 1, act_type)
        self.pw_conv = PWConvBNAct(in_channels, out_channels, act_type, inplace=True)
        if self.dilated:
            self.ddw_conv = DWConvBNAct(in_channels, in_channels, kernel_size, stride, dilation, act_type, inplace=True)

    def forward(self, x):
        x_dw = self.dw_conv(x)
        if self.dilated:
            x_ddw = self.ddw_conv(x)
            # x_dw += x_ddw
            x_dw = x_dw + x_ddw
        x = self.pw_conv(x_dw)

        return x

"FBSNet: A Fast Bilateral Symmetrical Network for Real-Time Semantic Segmentation"
""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from HWD import Down_wt
# from DepthWiseConv2d import DepthWiseConv2d
# from torchsummary import summary


__all__ = ["FBSNet"]


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        # self.conv = DepthWiseConv2d(nIn, nOut, kernel_size=kSize,
        #                                     stride=stride, padding=padding,
        #                                     dilation=dilation, bias=bias)


        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)




class BRUModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):  #
        super().__init__()
        #
        self.bn_relu_1 = BNPReLU(nIn)  #

        self.conv1x1_init = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=True)  #
        self.ca0 = eca_layer(nIn // 2)
        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)

        self.dconv1x3_l = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.dconv3x1_l = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)

        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3_r = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1_r = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.ca11 = eca_layer(nIn // 2)
        self.ca22 = eca_layer(nIn // 2)
        self.ca = eca_layer(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle_end = ShuffleBlock(groups=nIn // 2)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_init(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        b1 = self.ca11(br1)
        br1 = self.dconv1x3_l(b1)
        br1 = self.dconv3x1_l(br1)

        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        b2 = self.ca22(br2)
        br2 = self.ddconv1x3_r(b2)
        br2 = self.ddconv3x1_r(br2)


        output = br1 + br2 + self.ca0(output )+ b1 + b2

        output = self.bn_relu_2(output)

        output = self.conv1x1(output)
        output = self.ca(output)
        out = self.shuffle_end(output + input)
        return out



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)




class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool],
                               1)

        output = self.bn_prelu(output)

        return output



class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace= True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class FBSNet(nn.Module):
    def __init__(self, classes=4, block_1=5, block_2=5, block_3 = 16, block_4 = 3, block_5 = 3):
        super().__init__()

        # ---------- Encoder -------------#
        self.init_conv = nn.Sequential(
            Conv(3, 16, 3, 2, padding=1, bn_acti=True),
            Conv(16, 16, 3, 1, padding=1, bn_acti=True),
            Conv(16, 16, 3, 1, padding=1, bn_acti=True),
        )
        # 1/2
        self.bn_prelu_1 = BNPReLU(16)

        # Branch 1
        # Attention 1
        self.attention1_1 = eca_layer(16)

        # BRU Block 1
        dilation_block_1 = [1, 1, 1, 1, 1]
        self.BRU_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.BRU_Block_1.add_module("BRU_Module_1_" + str(i) ,BRUModule(16, d=dilation_block_1[i]))
        self.bn_prelu_2 = BNPReLU(16)
        # Attention 2
        self.attention2_1 = eca_layer(16)



        # Down 1  1/4
        self.downsample_1 = DownSamplingBlock(16, 64)
        # self.downsample_1 = Down_wt(16, 64)
        # BRU Block 2
        dilation_block_2 = [1, 2, 5, 9, 17]
        self.BRU_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.BRU_Block_2.add_module("BRU_Module_2_" + str(i) ,BRUModule(64, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(64)
        # Attention 3
        self.attention3_1 = eca_layer(64)


        # Down 2  1/8
        self.downsample_2 = DownSamplingBlock(64, 128)
        # self.downsample_2 = Down_wt(64, 128)
        # BRU Block 3
        dilation_block_3 = [1, 2, 5, 9, 1, 2, 5, 9,       2, 5, 9, 17, 2, 5, 9, 17]
        self.BRU_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.BRU_Block_3.add_module("BRU_Module_3_" + str(i), BRUModule(128, d=dilation_block_3[i]))
        self.bn_prelu_4 = BNPReLU(128)
        # Attention 4
        self.attention4_1 = eca_layer(128)





        # --------------Decoder   ----------------- #
        # Up 1 1/4
        self.upsample_1 = UpsamplerBlock(128, 64)

        # BRU Block 4
        dilation_block_4 = [1, 1, 1]
        self.BRU_Block_4 = nn.Sequential()
        for i in range(0, block_4):
            self.BRU_Block_4.add_module("BRU_Module_4_" + str(i), BRUModule(64, d=dilation_block_4[i]))
        self.bn_prelu_5 = BNPReLU(64)
        self.attention5_1 = eca_layer(64)
        # self.attention5_1 = CoordAtt(64,64)



        # Up 2 1/2
        self.upsample_2 = UpsamplerBlock(64, 32)
        # BRU Block 5
        dilation_block_5 = [1, 1, 1]
        self.BRU_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.BRU_Block_5.add_module("BRU_Module_5_" + str(i), BRUModule(32, d=dilation_block_5[i]))
        self.bn_prelu_6 = BNPReLU(32)
        self.attention6_1 = eca_layer(32)




        # Branch 2
        self.conv_sipath1 = Conv(16, 32, 3, 1, 1, bn_acti=True)
        self.conv_sipath2 = Conv(32, 128, 3, 1, 1, bn_acti=True)
        self.conv_sipath3 = Conv(128, 32, 3, 1, 1, bn_acti=True)

        self.atten_sipath = SpatialAttention()
        self.bn_prelu_8 = BNPReLU(32)
        self.bn_prelu_9 = BNPReLU(32)

        self.endatten = CoordAtt(32, 32)

        self.output_conv = nn.ConvTranspose2d(32, classes, 2, stride=2, padding=0, output_padding=0, bias=True)




    def forward(self, input):

        output0 = self.init_conv(input)
        output0 = self.bn_prelu_1(output0)

        # Branch1
        output1 = self.attention1_1(output0)

        # block1
        output1 = self.BRU_Block_1(output1)
        output1 = self.bn_prelu_2(output1)
        output1 = self.attention2_1(output1)

        # down1
        output1 = self.downsample_1(output1)

        # block2
        output1 = self.BRU_Block_2(output1)
        output1 = self.bn_prelu_3(output1)
        output1 = self.attention3_1(output1)

        # down2
        output1 = self.downsample_2(output1)

        # block3
        output2 = self.BRU_Block_3(output1)
        output2 = self.bn_prelu_4(output2)
        output2 = self.attention4_1(output2)


        # ---------- Decoder ----------------
        # up1
        output = self.upsample_1(output2)

        # block4
        output = self.BRU_Block_4(output)
        output = self.bn_prelu_5(output)
        output = self.attention5_1(output)

        # up2
        output = self.upsample_2(output)

        # block5
        output = self.BRU_Block_5(output)
        output = self.bn_prelu_6(output)
        output = self.attention6_1(output)


        # Detail Branch
        output_sipath = self.conv_sipath1(output0)
        output_sipath = self.conv_sipath2(output_sipath)
        output_sipath = self.conv_sipath3(output_sipath)
        output_sipath = self.atten_sipath(output_sipath)

        # Feature Fusion Module
        output = self.bn_prelu_8(output + output_sipath)

        # Feature Augment Module
        output = self.endatten(output)

        # output projection
        out = self.output_conv(output)
        out = F.softmax(out, dim=1)

        return out


################################# base line #####################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable
from cc import CC_module as CrissCrossAttention

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.inc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=False)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=False)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d((2, 2))
        )
        # self.down1 = Down_wt(in_ch=64, out_ch=128)

        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=False)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d((2, 2))
        )
        # self.down2 = Down_wt(in_ch=128, out_ch=256)

        self.res3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(inplace=False)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.GroupNorm(4, 512),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d((2, 2))
        )
        # self.down3 = Down_wt(256, 512)


    def forward(self, x):
        inc_feature = self.inc(x)

        res_1 = self.res1(inc_feature)
        res_1 = res_1 + inc_feature

        down1 = self.down1(res_1)

        res_2 = self.res2(down1)
        res_2 = res_2 + down1
       
        down2 = self.down2(res_2)
      
        res_3 = self.res3(down2)
        res_3 = res_3 + down2
     
        down3 = self.down3(res_3)


        return down3
    
    
class RCCmodule(nn.Module):
    def __init__(self):
        super(RCCmodule, self).__init__()

        self.conva = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.ReLU(inplace=False)
                                   )
        self.cca = CrissCrossAttention(128)

        self.convb = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.ReLU(inplace=False)
                                   )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(640, 512, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(4, 512),
            nn.ReLU(inplace=True),

        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512,256, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(inplace=True),
        )

        self.S3 = nn.ConvTranspose2d(512, 4, kernel_size=(8, 8), stride=(8, 8))
        self.S2 = nn.ConvTranspose2d(128, 4, kernel_size=(4, 4), stride=(4, 4))
        self.S1 = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0, bias=True)

        self.out = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0, bias=True)
        )


    def forward(self,x,recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        up1 = self.up1(output)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        S3 = self.S3(output)
        S2 = self.S2(up1)
        S1 = self.S1(up3)
        S = torch.cat([S3,S2,S1],dim=1)
        output = self.out(S)
        output = F.softmax(output, dim=1)
        return output
    
class SegNetwork(nn.Module):
    def __init__(self):
        super(SegNetwork, self).__init__()

        self.cnn = ResNet()
        self.seg = RCCmodule()

    def forward(self,x):
        # print(x.shape)
        out = self.cnn(x)
        # print(out.shape)
        out = self.seg(out)

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from cc import CC_module as CrissCrossAttention
# from IPython import embed
# from thop import profile
import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到 sys.path 中
if current_directory not in sys.path:
    sys.path.append(current_directory)
from transformer import TransBlock
from patch import reverse_patches
from HWD import Down_wt
# from scSE import scSE
# from MDTA import Attention
# from MDCR import MDCR
# from ema import EMA
# from torchviz import make_dot



__all__ = ["LETNet"]


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.conv3x1 = Conv(nIn // 2, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.conv1x3 = Conv(nIn // 2, nIn // 2, (1, kSize), 1, padding=(0, 1), bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ca11 = eca_layer(nIn // 2)
        
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ca22 = eca_layer(nIn // 2)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle = ShuffleBlock(nIn // 2)
        
    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        output = self.conv3x1(output)
        output = self.conv1x3(output)
        
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.ca11(br1)
        
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br1 + br2 + output
        output = self.bn_relu_2(output)

        output = self.conv1x1(output)
        output = self.shuffle(output + input)

        return output

        #return output + input



class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    
class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output

class UpsampleingBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ContextBlock(nn.Module):
    def __init__(self,inplanes,ratio,pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out    
        
class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize,  bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti
        self.dconv3x1 = nn.Conv2d(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0))
        self.dconv1x3 = nn.Conv2d(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1))
        
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output
                 

class LETNet(nn.Module):
    def __init__(self, classes=4, block_1=3, block_2=12, block_3=12, block_4=3, block_5 = 3, block_6 = 3):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 2, padding=1, bn_acti=True),
        )

        self.bn_prelu_1 = BNPReLU(32)

        # self.downsample_1 = DownSamplingBlock(32, 64)
        self.downsample_1 = Down_wt(32,64)

        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(64)

        # DAB Block 2
        dilation_block_2 = [1,1, 2, 2, 4, 4, 8, 8, 16, 16,32,32]
        # self.downsample_2 = DownSamplingBlock(64, 128)
        self.downsample_2 = Down_wt(64, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(128)

        # DAB Block 3
        #dilation_block_3 = [2, 5, 7, 9, 13, 17]
        dilation_block_3 = [1,1, 2, 2, 4, 4, 8, 8, 16, 16,32,32]
        self.downsample_3 = DownSamplingBlock(128, 32)
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        DABModule(32, d=dilation_block_3[i]))
        self.bn_prelu_4 = BNPReLU(32)
        
        

        self.transformer1 = TransBlock(dim=288)
        
        
#DECODER
        dilation_block_4 = [2, 2, 2]
        self.DAB_Block_4 = nn.Sequential()
        for i in range(0, block_4):
           self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
                                       DABModule(32, d=dilation_block_4[i]))
        self.upsample_1 = UpsampleingBlock(32, 16)
        self.bn_prelu_5 = BNPReLU(16)
        
        dilation_block_5 = [2, 2, 2]
        self.DAB_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
                                        DABModule(16, d=dilation_block_5[i]))
        self.upsample_2 = UpsampleingBlock(16, 16)
        self.bn_prelu_6 = BNPReLU(16)
        
        
        dilation_block_6 = [2, 2, 2]
        self.DAB_Block_6 = nn.Sequential()
        for i in range(0, block_6):
            self.DAB_Block_6.add_module("DAB_Module_6_" + str(i),
                                        DABModule(16, d=dilation_block_6[i]))
        self.upsample_3 = UpsampleingBlock(16, 16)
        self.bn_prelu_7 = BNPReLU(16)
        
        
        self.PA1 = PA(16)
        self.PA2 = PA(16)
        self.PA3 = PA(16)
        
        self.LC1 = LongConnection(64, 16, 3)
        self.LC2 = LongConnection(128, 16, 3)
        self.LC3 = LongConnection(32, 32, 3)
        
        self.classifier = nn.Sequential(Conv(16, classes, 1, 1, padding=0))

    def forward(self, input):
        output0 = self.init_conv(input)
        output0 = self.bn_prelu_1(output0)

        # # DAB Block 1
        output1_0 = self.downsample_1(output0)
        output1 = self.DAB_Block_1(output1_0)
        output1 = self.bn_prelu_2(output1)

        # DAB Block 2
        output2_0 = self.downsample_2(output1)
        output2 = self.DAB_Block_2(output2_0)
        output2 = self.bn_prelu_3(output2)

        # DAB Block 3
        output3_0 = self.downsample_3(output2)
        output3 = self.DAB_Block_3(output3_0)
        output3 = self.bn_prelu_4(output3)   
        b,c,h,w = output3.shape
# #Transformer

        b, c, h, w = output3.shape
        output4 = self.transformer1(output3)
        
        output4 = output4.permute(0, 2, 1)
        output4 = reverse_patches(output4, (h, w), (3, 3), 1, 1)

        
# #DECODER           
        output4 = self.DAB_Block_4(output4)
    
        output4 = self.upsample_1(output4 + self.LC3(output3))
        output4 = self.bn_prelu_5(output4)
        
        
        output5 = self.DAB_Block_5(output4)
        # 获取两个张量在第三维（高度）的尺寸
        height5 = output5.shape[2]
        height2 = self.LC2(output2).shape[2]

        # 确定较小的尺寸
        min_height = min(height5, height2)

        # 裁剪较大的张量
        if height5 > min_height:
            output5 = output5[:, :, :min_height, :]
        output5 = self.upsample_2(output5 + self.LC2(output2))
        output5 = self.bn_prelu_6(output5)
        
        output6 = self.DAB_Block_6(output5)

        output6 = self.upsample_3(output6 + self.LC1(output1))
        output6 = self.PA3(output6)
        output6 = self.bn_prelu_7(output6)
        
        out = F.interpolate(output6, input.size()[2:], mode='bilinear', align_corners=False)
        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        return out


################################## R-SegLETNet #################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from cc import CC_module as CrissCrossAttention
# from IPython import embed
# from thop import profile
import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到 sys.path 中
if current_directory not in sys.path:
    sys.path.append(current_directory)
from transformer import TransBlock
from patch import reverse_patches
# from HWD import Down_wt
# from scSE import scSE
# from MDTA import Attention
# from MDCR import MDCR
# from ema import EMA
# from torchviz import make_dot



__all__ = ["R-SegLETNet"]


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.conv3x1 = Conv(nIn // 2, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.conv1x3 = Conv(nIn // 2, nIn // 2, (1, kSize), 1, padding=(0, 1), bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ca11 = eca_layer(nIn // 2)
        
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ca22 = eca_layer(nIn // 2)

        # self.conva = nn.Sequential(
        #     nn.Conv2d(nIn, nIn // 2, (1, 3), padding=(0, 1), bias=False),  # Horizontal conv
        #     nn.Conv2d(nIn // 2, nIn // 2, (3, 1), padding=(1, 0), bias=False),  # Vertical conv
        #     # nn.GroupNorm(4, 16),
        #     # nn.ReLU(inplace=False)
        #     BNPReLU(nIn // 2)
        # )
        # self.cca = CrissCrossAttention(nIn // 2)

        # self.convb = nn.Sequential(
        #     nn.Conv2d(nIn // 2, nIn // 2, (1, 3), padding=(0, 1), bias=False),  # Horizontal conv
        #     nn.Conv2d(nIn // 2, nIn // 2, (3, 1), padding=(1, 0), bias=False),  # Vertical conv
        #     # nn.GroupNorm(4, 16),
        #     # nn.ReLU(inplace=False)
        #     BNPReLU(nIn // 2)
        # )

        # self.bottleneck = nn.Sequential(
        #     # 第一个卷积层：3x1 卷积
        #     nn.Conv2d(nIn + (nIn // 2), nIn, kernel_size=(3, 1), padding=(1, 0), bias=False),
        #     # 第二个卷积层：1x3 卷积
        #     nn.Conv2d(nIn, nIn, kernel_size=(1, 3), padding=(0, 1), bias=False),
        #     # 组归一化
        #     # nn.GroupNorm(4, 32),
        #     # # 激活函数
        #     # nn.ReLU(inplace=True),
        #     BNPReLU(nIn)
        # )

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle = ShuffleBlock(nIn // 2)
        
    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        output = self.conv3x1(output)
        output = self.conv1x3(output)
        
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.ca11(br1)
        
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br1 + br2 + output
        output = self.bn_relu_2(output)

        output = self.conv1x1(output)
        output = self.shuffle(output + input)

        # output_cc = self.conva(output)
        # for i in range(2):
        #     output_cc = self.cca(output_cc)
            
        # output_cc = self.convb(output_cc)
        # output = self.bottleneck(torch.cat([output, output_cc], 1))

        return output

        #return output + input



class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    
class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output

class UpsampleingBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class MSPA(nn.Module):
    '''Multi-Scale Pixel Attention Module'''
    def __init__(self, nf):
        super(MSPA, self).__init__()

        # 多尺度卷积层
        # self.conv1x1 = nn.Conv2d(nf, nf, kernel_size=1, padding=0)  # 1x1 卷积
        self.conv3x3 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)  # 1x1 卷积
        self.conv5x5 = nn.Conv2d(nf, nf, kernel_size=5, padding=2)  # 3x3 卷积
        self.conv7x7 = nn.Conv2d(nf, nf, kernel_size=7, padding=3)  # 5x5 卷积

        # 每个尺度对应的像素注意力模块
        #self.pa1x1 = PA(nf)
        self.pa3x3 = PA(nf)
        self.eca_layer = eca_layer(16)
        self.pa5x5 = PA(nf)
        self.pa7x7 = PA(nf)

        # 特征融合层
        self.fusion = nn.Conv2d(nf * 4, nf, kernel_size=1, padding=0)
        

    def forward(self, x):
        # 多尺度特征提取
        # scale1x1 = self.conv1x1(x)
        # scale1x1 = self.eca_layer(scale1x1)

        scale3x3 = self.conv3x3(x)
        scale3x3 = self.eca_layer(scale3x3)

        scale5x5 = self.conv5x5(x)
        scale5x5 = self.eca_layer(scale5x5)
        
        scale7x7 = self.conv7x7(x)
        scale7x7 = self.eca_layer(scale7x7)

        # 分别对不同尺度的特征应用像素注意力
        # pa1x1 = self.pa1x1(scale1x1)
        pa3x3 = self.pa3x3(scale3x3)
        pa5x5 = self.pa5x5(scale5x5)
        pa7x7 = self.pa7x7(scale7x7)

        # 将多尺度注意力特征拼接起来
        multi_scale_features = torch.cat([x, pa3x3, pa5x5, pa7x7], dim=1) # 尝试加入x

        # 融合多尺度特征
        out = self.fusion(multi_scale_features)

        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ContextBlock(nn.Module):
    def __init__(self,inplanes,ratio,pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out    
        
class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize,  bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti
        self.dconv3x1 = nn.Conv2d(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0))
        self.dconv1x3 = nn.Conv2d(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1))
        
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output

# class DepthwiseSeparableConv(nn.Module):
#     """深度可分离卷积"""
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
#                                    padding=padding, dilation=dilation, groups=in_channels, bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x

# class LongConnectionMultiScaleDilatedDS(nn.Module):
#     def __init__(self, nIn, nOut, scales=[3, 5, 7], dilation_rates=[4, 5, 6], bn_acti=False, bias=False):
#         super(LongConnectionMultiScaleDilatedDS, self).__init__()
        
#         self.bn_acti = bn_acti
        
#         # 分支1：三种不同尺度的深度可分离卷积
#         self.conv_branch1_1 = DepthwiseSeparableConv(nIn, nIn // 4, kernel_size=scales[0], padding=scales[0] // 2)
#         self.conv_branch1_2 = DepthwiseSeparableConv(nIn, nIn // 4, kernel_size=scales[1], padding=scales[1] // 2)
#         self.conv_branch1_3 = DepthwiseSeparableConv(nIn, nIn // 4, kernel_size=scales[2], padding=scales[2] // 2)

#         # 分支2：三种不同膨胀率的空洞卷积
#         self.dconv_branch2_1 = DepthwiseSeparableConv(nIn, nIn // 4, kernel_size=3, padding=dilation_rates[0], dilation=dilation_rates[0], bias=bias)
#         self.dconv_branch2_2 = DepthwiseSeparableConv(nIn, nIn // 4, kernel_size=3, padding=dilation_rates[1], dilation=dilation_rates[1], bias=bias)
#         self.dconv_branch2_3 = DepthwiseSeparableConv(nIn, nIn // 4, kernel_size=3, padding=dilation_rates[2], dilation=dilation_rates[2], bias=bias)
        
#         # BN + PReLU 激活函数 (可选)
#         if self.bn_acti:
#             self.bn_prelu = BNPReLU(nOut)
        
#         # 最终融合层，修改为匹配通道数 384
#         self.conv_fuse = nn.Conv2d(nIn * 6 // 4, nOut, kernel_size=1, stride=1, padding=0, bias=bias)

#     def forward(self, input):
#         # 分支1：多尺度卷积
#         branch1_output_1 = self.conv_branch1_1(input)
#         branch1_output_2 = self.conv_branch1_2(input)
#         branch1_output_3 = self.conv_branch1_3(input)
        
#         # 分支2：空洞卷积
#         branch2_output_1 = self.dconv_branch2_1(input)
#         branch2_output_2 = self.dconv_branch2_2(input)
#         branch2_output_3 = self.dconv_branch2_3(input)
        
#         # 将两条分支的输出进行融合
#         fused_output = torch.cat([branch1_output_1, branch1_output_2, branch1_output_3, 
#                                   branch2_output_1, branch2_output_2, branch2_output_3], dim=1)  # 在通道维度上拼接
#         fused_output = self.conv_fuse(fused_output)  # 融合通道
        
#         # 如果需要 BN 和 PReLU
#         if self.bn_acti:
#             fused_output = self.bn_prelu(fused_output)

#         return fused_output

                 

class RSegLETNet(nn.Module):
    def __init__(self, classes=4, block_1=3, block_2=12, block_3=12, block_4=3, block_5 = 3, block_6 = 3):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 2, padding=1, bn_acti=True),
        )

        self.bn_prelu_1 = BNPReLU(32)

        # self.conva = nn.Sequential(
        #     nn.Conv2d(32, 16, (1, 3), padding=(0, 1), bias=False),  # Horizontal conv
        #     nn.Conv2d(16, 16, (3, 1), padding=(1, 0), bias=False),  # Vertical conv
        #     # nn.GroupNorm(4, 16),
        #     # nn.ReLU(inplace=False)
        #     BNPReLU(16)
        # )
        # self.cca = CrissCrossAttention(16)

        # self.convb = nn.Sequential(
        #     nn.Conv2d(16, 16, (1, 3), padding=(0, 1), bias=False),  # Horizontal conv
        #     nn.Conv2d(16, 16, (3, 1), padding=(1, 0), bias=False),  # Vertical conv
        #     # nn.GroupNorm(4, 16),
        #     # nn.ReLU(inplace=False)
        #     BNPReLU(16)
        # )

        # self.bottleneck = nn.Sequential(
        #     # 第一个卷积层：3x1 卷积
        #     nn.Conv2d(48, 32, kernel_size=(3, 1), padding=(1, 0), bias=False),
        #     # 第二个卷积层：1x3 卷积
        #     nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
        #     # 组归一化
        #     # nn.GroupNorm(4, 32),
        #     # # 激活函数
        #     # nn.ReLU(inplace=True),
        #     BNPReLU(32)
        # )
        # self.attentation = MDCR(in_features=32, out_features=32)

        # self.conva_5 = nn.Sequential(
        #     nn.Conv2d(16, 8, (1, 3), padding=(0, 1), bias=False),  # Horizontal conv
        #     nn.Conv2d(8, 8, (3, 1), padding=(1, 0), bias=False),  # Vertical conv
        #     # nn.GroupNorm(4, 16),
        #     # nn.ReLU(inplace=False)
        #     BNPReLU(8)
        # )
        # self.cca_5 = CrissCrossAttention(8)

        # self.convb_5 = nn.Sequential(
        #     nn.Conv2d(8, 8, (1, 3), padding=(0, 1), bias=False),  # Horizontal conv
        #     nn.Conv2d(8, 8, (3, 1), padding=(1, 0), bias=False),  # Vertical conv
        #     # nn.GroupNorm(4, 16),
        #     # nn.ReLU(inplace=False)
        #     BNPReLU(8)
        # )

        # self.bottleneck_5 = nn.Sequential(
        #     # 第一个卷积层：3x1 卷积
        #     nn.Conv2d(24, 16, kernel_size=(3, 1), padding=(1, 0), bias=False),
        #     # 第二个卷积层：1x3 卷积
        #     nn.Conv2d(16, 16, kernel_size=(1, 3), padding=(0, 1), bias=False),
        #     # 组归一化
        #     # nn.GroupNorm(4, 32),
        #     # # 激活函数
        #     # nn.ReLU(inplace=True),
        #     BNPReLU(16)
        # )

        # self.conva_6 = nn.Sequential(
        #     nn.Conv2d(16, 8, (1, 3), padding=(0, 1), bias=False),  # Horizontal conv
        #     nn.Conv2d(8, 8, (3, 1), padding=(1, 0), bias=False),  # Vertical conv
        #     # nn.GroupNorm(4, 16),
        #     # nn.ReLU(inplace=False)
        #     BNPReLU(8)
        # )
        # self.cca_6 = CrissCrossAttention(8)

        # self.convb_6 = nn.Sequential(
        #     nn.Conv2d(8, 8, (1, 3), padding=(0, 1), bias=False),  # Horizontal conv
        #     nn.Conv2d(8, 8, (3, 1), padding=(1, 0), bias=False),  # Vertical conv
        #     # nn.GroupNorm(4, 16),
        #     # nn.ReLU(inplace=False)
        #     BNPReLU(8)
        # )

        # self.bottleneck_6 = nn.Sequential(
        #     # 第一个卷积层：3x1 卷积
        #     nn.Conv2d(24, 16, kernel_size=(3, 1), padding=(1, 0), bias=False),
        #     # 第二个卷积层：1x3 卷积
        #     nn.Conv2d(16, 16, kernel_size=(1, 3), padding=(0, 1), bias=False),
        #     # 组归一化
        #     # nn.GroupNorm(4, 32),
        #     # # 激活函数
        #     # nn.ReLU(inplace=True),
        #     BNPReLU(16)
        # )
        self.downsample_1 = DownSamplingBlock(32, 64)
        # self.downsample_1 = Down_wt(32,64)

        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(64)

        # DAB Block 2
        dilation_block_2 = [1,1, 2, 2, 4, 4, 8, 8, 16, 16,32,32]
        self.downsample_2 = DownSamplingBlock(64, 128)
        # self.downsample_2 = Down_wt(64, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(128)

        # DAB Block 3
        #dilation_block_3 = [2, 5, 7, 9, 13, 17]
        dilation_block_3 = [1,1, 2, 2, 4, 4, 8, 8, 16, 16,32,32]
        self.downsample_3 = DownSamplingBlock(128, 32)
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        DABModule(32, d=dilation_block_3[i]))
        self.bn_prelu_4 = BNPReLU(32)
        
        

        self.transformer1 = TransBlock(dim=288)
        # self.EMA_32 = EMA(32)
        # self.EMA_16 = EMA(16)
        
        
#DECODER
        dilation_block_4 = [2, 2, 2]
        self.DAB_Block_4 = nn.Sequential()
        for i in range(0, block_4):
           self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
                                       DABModule(32, d=dilation_block_4[i]))
        self.upsample_1 = UpsampleingBlock(32, 16)
        self.bn_prelu_5 = BNPReLU(16)
        

        # dilation_block_5 = [2, 2, 2]
        # self.DAB_Block_5 = nn.Sequential()
        # for i in range(0, block_5):
        #     self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
        #                                 DABModule(16, d=dilation_block_5[i]))
        # self.upsample_2 = UpsampleingBlock(16, 16)
        # self.bn_prelu_6 = BNPReLU(16)
        
        dilation_block_5 = [2, 2, 2]
        self.DAB_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
                                        DABModule(16, d=dilation_block_5[i]))
        self.upsample_2 = UpsampleingBlock(16, 16)
        self.bn_prelu_6 = BNPReLU(16)
        
        
        dilation_block_6 = [2, 2, 2]
        self.DAB_Block_6 = nn.Sequential()
        for i in range(0, block_6):
            self.DAB_Block_6.add_module("DAB_Module_6_" + str(i),
                                        DABModule(16, d=dilation_block_6[i]))
        self.upsample_3 = UpsampleingBlock(16, 16)
        self.bn_prelu_7 = BNPReLU(16)
        
        
        self.PA1 = PA(16)
        self.PA2 = PA(16)
        self.PA3 = MSPA(16)
        # self.PA3 = PA(16)
        
        self.LC1 = LongConnection(64, 16, 3)
        self.LC2 = LongConnection(128, 16, 3)
        self.LC3 = LongConnection(32, 32, 3)

        # self.LC1 = LongConnectionMultiScaleDilatedDS(64, 16)
        # self.LC2 = LongConnectionMultiScaleDilatedDS(128, 16)
        # self.LC3 = LongConnectionMultiScaleDilatedDS(32, 32)
        
        self.classifier = nn.Sequential(Conv(16, classes, 1, 1, padding=0))

    def forward(self, input):
        output0 = self.init_conv(input)
        output0 = self.bn_prelu_1(output0)

        # # DAB Block 1
        output1_0 = self.downsample_1(output0)
        output1 = self.DAB_Block_1(output1_0)
        output1 = self.bn_prelu_2(output1)

        # DAB Block 2
        output2_0 = self.downsample_2(output1)
        output2 = self.DAB_Block_2(output2_0)
        output2 = self.bn_prelu_3(output2)

        # DAB Block 3
        output3_0 = self.downsample_3(output2)
        output3 = self.DAB_Block_3(output3_0)
        output3 = self.bn_prelu_4(output3)   
        b,c,h,w = output3.shape
# #Transformer

        b, c, h, w = output3.shape
        
        # output_cc = self.conva(output3)
        # for i in range(2):
        #     output_cc = self.cca(output_cc)
            
        # output_cc = self.convb(output_cc)
        # output3 = self.bottleneck(torch.cat([output3, output_cc], 1))
        # output3 = self.attentation(output3)
        # output4 = self.EMA_32(output3)
        output4 = self.transformer1(output3)
        
        output4 = output4.permute(0, 2, 1)
        output4 = reverse_patches(output4, (h, w), (3, 3), 1, 1)
        # output_cc = self.conva(output4)
        # for i in range(2):
        #     output_cc = self.cca(output_cc)
            
        # output_cc = self.convb(output_cc)
        # output4 = self.bottleneck(torch.cat([output4, output_cc], 1))

        
# #DECODER           
        output4 = self.DAB_Block_4(output4)
        # output_cc = self.conva_4(output4)
        # for i in range(2):
        #     output_cc = self.cca_4(output_cc)
            
        # output_cc = self.convb_4(output_cc)
        # output4 = self.bottleneck_4(torch.cat([output4, output_cc], 1))
    
        output4 = self.upsample_1(output4 + self.LC3(output3))
        output4 = self.bn_prelu_5(output4)
        
        
        output5 = self.DAB_Block_5(output4)
        # output_cc = self.conva_5(output5)
        # for i in range(2):
        #     output_cc = self.cca_5(output_cc)
            
        # output_cc = self.convb_5(output_cc)
        # output5 = self.bottleneck_5(torch.cat([output5, output_cc], 1))
        # 获取两个张量在第三维（高度）的尺寸
        height5 = output5.shape[2]
        height2 = self.LC2(output2).shape[2]

        # 确定较小的尺寸
        min_height = min(height5, height2)

        # 裁剪较大的张量
        if height5 > min_height:
            output5 = output5[:, :, :min_height, :]
        output5 = self.upsample_2(output5 + self.LC2(output2))
        output5 = self.bn_prelu_6(output5)
        
        output6 = self.DAB_Block_6(output5)
        
        # output_cc = self.conva_6(output6)
        # for i in range(2):
        #     output_cc = self.cca_6(output_cc)
            
        # output_cc = self.convb_6(output_cc)
        # output6 = self.bottleneck_6(torch.cat([output6, output_cc], 1))

        output6 = self.upsample_3(output6 + self.LC1(output1))
        output6 = self.PA3(output6)
        output6 = self.bn_prelu_7(output6)
        
        # output_cc = self.conva_5(output6)
        # for i in range(2):
        #     output_cc = self.cca_5(output_cc)
            
        # output_cc = self.convb_5(output_cc)
        # output6 = self.bottleneck_5(torch.cat([output6, output_cc], 1))
        
        out = F.interpolate(output6, input.size()[2:], mode='bilinear', align_corners=False)
        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        return out
