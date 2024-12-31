import torch
from torch import nn
import torch.nn.functional as F
import timm

from hybridnets.modelv2 import BiFPN, Regressor, Classifier, BiFPNDecoder
from utils.utils import Anchors
from hybridnets.modelv2 import SegmentationHead

from encoders import get_encoder
from utils.constants import *


class AttentionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class HybridNetsBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, seg_classes=1, backbone_name=None, seg_mode=MULTICLASS_MODE,
                 onnx_export=False, seg_p2_in_channels=32, pretrain_weight='', **kwargs):
        super(HybridNetsBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.seg_classes = seg_classes
        self.seg_mode = seg_mode

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, ]
        # self.anchor_scale = [2.,2.,2.,2.,2.,2.,2.,2.,2.,]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [24, 48, 576],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        total_bifpn_channels = self.fpn_num_filters[self.compound_coef] * 6
        if backbone_name == 'efficientnet_b3':
            conv_channel_coef[3] = [48, 136, 384]
            total_bifpn_channels = 832
        elif backbone_name == 'mobilenetv3_large_100_miil_in21k':
            conv_channel_coef[3] = [40, 112, 960]
            total_bifpn_channels = 824

        self.onnx_export = onnx_export
        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7,
                    onnx_export=onnx_export)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef],
                                   onnx_export=onnx_export)

        '''Modified by Dat Vu'''
        # self.decoder = DecoderModule()
        self.bifpndecoder = BiFPNDecoder(pyramid_channels=self.fpn_num_filters[self.compound_coef],
                                         seg_p2_in_channels=seg_p2_in_channels)

        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=1 if self.seg_mode == BINARY_MODE else self.seg_classes + 1,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

        self.ll_attention_head = AttentionHead(in_channels=total_bifpn_channels, out_channels=1)

        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef],
                                     onnx_export=onnx_export)

        self.encoder = timm.create_model(backbone_name, pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
        if pretrain_weight != '':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.encoder.load_state_dict(torch.load(pretrain_weight, map_location=self.device), strict=False)

        if not onnx_export:
            self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                                   pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                                   onnx_export=onnx_export,
                                   **kwargs)
        else:
            ## TODO: timm
            self.encoder.set_swish(memory_efficient=False)

        self.initialize_decoder(self.bifpndecoder)
        self.initialize_head(self.segmentation_head)
        self.initialize_decoder(self.bifpn)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        # p1, p2, p3, p4, p5 = self.backbone_net(inputs)
        encoder_features = self.encoder(inputs)[-4:]
        p2, p3, p4, p5 = self.encoder(inputs)[-4:]  # self.backbone_net(inputs)
        # print(p2.shape, p3.shape, p4.shape, p5.shape)
        features = (p3, p4, p5)

        features = self.bifpn(features)

        p3, p4, p5, p6, p7 = features

        outputs = self.bifpndecoder((p2, p3, p4, p5, p6, p7))

        segmentation = self.segmentation_head(outputs)

        # Concatenate all resized feature maps
        target_size = p2.shape[2:]  # Use the size of p3 as the target size
        resized_p3 = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        resized_p4 = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        resized_p5 = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)
        resized_p6 = F.interpolate(p6, size=target_size, mode='bilinear', align_corners=False)
        resized_p7 = F.interpolate(p7, size=target_size, mode='bilinear', align_corners=False)
        concat_features = torch.cat([p2, resized_p3, resized_p4, resized_p5, resized_p6, resized_p7], dim=1)
        # print(concat_features.shape)
        attention_map = self.ll_attention_head(concat_features)
        attention_map = F.interpolate(attention_map, size=inputs.shape[2:], mode='bilinear', align_corners=False)
        ll_mask = (attention_map > 0.3).float()
        segmentation = segmentation * ll_mask

        regression = self.regressor(features)
        classification = self.classifier(features)

        if not self.onnx_export:
            anchors = self.anchors(inputs, inputs.dtype)
            return features, regression, classification, anchors, segmentation
        else:
            return regression, classification, segmentation

    def initialize_decoder(self, module):
        for m in module.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def initialize_head(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
