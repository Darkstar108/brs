import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet121',
           'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': './lip_densenet_121.pth',
}

BOTTLENECK_WIDTH = 128
COEFF = 12.0
weights_dict = dict()
skip_connections = dict() # To store references to the concat layers from the encoder

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding)/F.avg_pool2d(weight, kernel, stride, padding)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SoftGate(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).mul(COEFF)


class BottleneckLIP(nn.Module):
    def __init__(self, channels):
        super(BottleneckLIP, self).__init__()

        rp = BOTTLENECK_WIDTH

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv1', conv1x1(channels, rp)),
                ('bn1', nn.InstanceNorm2d(rp, affine=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', conv3x3(rp, rp)),
                ('bn2', nn.InstanceNorm2d(rp, affine=True)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', conv1x1(rp, channels)),
                ('bn3', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.logit[6].weight.data.fill_(0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x), kernel=2, stride=2, padding=0)
        return frac


class SimplifiedLIP(nn.Module):
    def __init__(self, channels):
        super(SimplifiedLIP, self).__init__()

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv1', conv3x3(channels, channels)),
                ('bn1', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.logit[0].weight.data.fill_(0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x), kernel=3, stride=2, padding=1)
        return frac


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features, momentum=0.0)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)

            '''
                Addding references to the required concat layers to a dictionary
            '''
            if i == num_layers - 1:
                if num_layers == 6:
                    skip_connections['concat_2_6'] = layer
                elif num_layers == 12:
                    skip_connections['concat_3_12'] = layer
                elif num_layers == 24:
                    skip_connections['concat_4_24'] = layer
                else
                    skip_connections['concat_5_16'] = layer

            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        #self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        self.add_module('pool', BottleneckLIP(num_output_features))


class Encoder(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), coarse_block_config=(32, 16, 8, 4)
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(Encoder, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features, momentum=0.0)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0_pad', nn.ConstantPad2d(3, 0)),
            ('pool0', SimplifiedLIP(num_init_features)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class Decoder(nn.Module):
    def __init__(self, weight_file):
        super(Decoder, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.conv_SE_1_32_1 = self.__conv(2, name='conv_SE_1/32_1', in_channels=1024, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_SE_1_32_2 = self.__conv(2, name='conv_SE_1/32_2', in_channels=64, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_1_32_1d = self.__conv(2, name='conv_1/32_1d', in_channels=1024, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_1_32_1d = self.__batch_normalization(2, 'bn_1/32_1d', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_32_2d = self.__conv(2, name='conv_1/32_2d', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_1_32_2d = self.__batch_normalization(2, 'bn_1/32_2d', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_32_3d = self.__conv(2, name='conv_1/32_3d', in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn_1_32_3d = self.__batch_normalization(2, 'bn_1/32_3d', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_1_16d = self.__batch_normalization(2, 'bn_1/16d', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_16_1d = self.__conv(2, name='conv_1/16_1d', in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_1_16_1d = self.__batch_normalization(2, 'bn_1/16_1d', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_16_2d = self.__conv(2, name='conv_1/16_2d', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_1_16_2d = self.__batch_normalization(2, 'bn_1/16_2d', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_16_3d = self.__conv(2, name='conv_1/16_3d', in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn_1_16_3d = self.__batch_normalization(2, 'bn_1/16_3d', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_1_8d = self.__batch_normalization(2, 'bn_1/8d', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_8_1d = self.__conv(2, name='conv_1/8_1d', in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_1_8_1d = self.__batch_normalization(2, 'bn_1/8_1d', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_8_2d = self.__conv(2, name='conv_1/8_2d', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_1_8_2d = self.__batch_normalization(2, 'bn_1/8_2d', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_8_3d = self.__conv(2, name='conv_1/8_3d', in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn_1_8_3d = self.__batch_normalization(2, 'bn_1/8_3d', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_1_4d = self.__batch_normalization(2, 'bn_1/4d', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_4_1d = self.__conv(2, name='conv_1/4_1d', in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_1_4_1d = self.__batch_normalization(2, 'bn_1/4_1d', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_4_2d = self.__conv(2, name='conv_1/4_2d', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_1_4_2d = self.__batch_normalization(2, 'bn_1/4_2d', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_1_4_3d = self.__conv(2, name='conv_1/4_3d', in_channels=64, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn_1_4_3d = self.__batch_normalization(2, 'bn_1/4_3d', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.pred_1_4 = self.__conv(2, name='pred_1/4', in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_atrous1_1 = self.__conv(2, name='conv_atrous1_1', in_channels=6, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv_atrous2_1 = self.__conv(2, name='conv_atrous2_1', in_channels=6, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv_atrous3_1 = self.__conv(2, name='conv_atrous3_1', in_channels=6, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_atrous1_1 = self.__batch_normalization(2, 'bn_atrous1_1', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_atrous2_1 = self.__batch_normalization(2, 'bn_atrous2_1', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_atrous3_1 = self.__batch_normalization(2, 'bn_atrous3_1', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_atrous1_2 = self.__conv(2, name='conv_atrous1_2', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv_atrous2_2 = self.__conv(2, name='conv_atrous2_2', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.conv_atrous3_2 = self.__conv(2, name='conv_atrous3_2', in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_atrous1_2 = self.__batch_normalization(2, 'bn_atrous1_2', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_atrous2_2 = self.__batch_normalization(2, 'bn_atrous2_2', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_atrous3_2 = self.__batch_normalization(2, 'bn_atrous3_2', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_atrous1_3 = self.__conv(2, name='conv_atrous1_3', in_channels=32, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_atrous2_3 = self.__conv(2, name='conv_atrous2_3', in_channels=32, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.conv_atrous3_3 = self.__conv(2, name='conv_atrous3_3', in_channels=32, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn_atrous1_3 = self.__batch_normalization(2, 'bn_atrous1_3', num_features=16, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_atrous2_3 = self.__batch_normalization(2, 'bn_atrous2_3', num_features=16, eps=9.999999747378752e-06, momentum=0.0)
        self.bn_atrous3_3 = self.__batch_normalization(2, 'bn_atrous3_3', num_features=16, eps=9.999999747378752e-06, momentum=0.0)
        self.conv_s2_down = self.__conv(2, name='conv_s2_down', in_channels=48, out_channels=3, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_s2_up = self.__conv(2, name='conv_s2_up', in_channels=3, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_p1_1 = self.__conv(2, name='conv_p1_1', in_channels=48, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn_p1_1 = self.__batch_normalization(2, 'bn_p1_1', num_features=16, eps=9.999999747378752e-06, momentum=0.0)
        self.pred_step_2 = self.__conv(2, name='pred_step_2', in_channels=16, out_channels=1, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.deconv_1_16d = nn.ConvTranspose2D(in_channels=256, out_channels=256, kernel_size=2, stride=2, bias=False)
        self.deconv_1_8d = nn.ConvTranspose2D(in_channels=128, out_channels=256, kernel_size=2, stride=2, bias=False)
        self.deconv_1_4d = nn.ConvTranspose2D(in_channels=64, out_channels=256, kernel_size=2, stride=2, bias=False)
        self.pred_step_1 = nn.ConvTranspose2D(in_channels=1, out_channels=1, kernel_size=4, stride=4, bias=False)

    def forward(self, x):
        '''
            concat_5_16     :       Last concat of the 5th denseblock
            concat_4_24     :       Last concat of the 4th denseblock
            concat_3_12     :       Last concat of the 3rd denseblock
            concat_2_6      :       Last concat of the 2nd denseblock

            TO DO:
                - Store the last concat from the _DenseLayer into a list/reference variable
                - Change the avg_pool2d to LIP in the below code (Need to decide)
        '''
        ################ COARSE DECODER ###############
        '''
            NOTE: For now, I have implemented the skip connections as so. If you have any other way, please suggest :)
        '''
        concat_5_16 = skip_connections['concat_5_16'].detach().clone()
        concat_4_24 = skip_connections['concat_4_24'].detach().clone()
        concat_3_12 = skip_connections['concat_3_12'].detach().clone()
        concat_2_6  = skip_connections['concat_2_6'].detach().clone()

        ################ BLOCK 1 ################      
        pool_SE_1_32    = F.avg_pool2d(concat_5_16, kernel_size=(15, 15), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        conv_SE_1_32_1  = self.conv_SE_1_32_1(pool_SE_1_32)
        relu_SE_1_32_1  = F.relu(conv_SE_1_32_1)
        conv_SE_1_32_2  = self.conv_SE_1_32_2(relu_SE_1_32_1)
        sigm_SE_1_32    = F.sigmoid(conv_SE_1_32_2)
        reshape_SE_1_32 = torch.reshape(input = sigm_SE_1_32, shape = (1,1024,1,1))
        scale_SE_1_32   = concat_5_16 * reshape_SE_1_32

        conv_1_32_1d_pad = F.pad(scale_SE_1_32, (1, 1, 1, 1))
        conv_1_32_1d    = self.conv_1_32_1d(conv_1_32_1d_pad)
        prelu_1_32_1d   = F.prelu(conv_1_32_1d, torch.from_numpy(_weights_dict['prelu_1/32_1d']['weights']))
        bn_1_32_1d      = self.bn_1_32_1d(prelu_1_32_1d)

        conv_1_32_2d_pad = F.pad(bn_1_32_1d, (1, 1, 1, 1))
        conv_1_32_2d    = self.conv_1_32_2d(conv_1_32_2d_pad)
        prelu_1_32_2d   = F.prelu(conv_1_32_2d, torch.from_numpy(_weights_dict['prelu_1/32_2d']['weights']))
        bn_1_32_2d      = self.bn_1_32_2d(prelu_1_32_2d)

        conv_1_32_3d    = self.conv_1_32_3d(bn_1_32_2d)
        prelu_1_32_3d   = F.prelu(conv_1_32_3d, torch.from_numpy(_weights_dict['prelu_1/32_3d']['weights']))
        bn_1_32_3d      = self.bn_1_32_3d(prelu_1_32_3d)

        deconv_1_16d    = self.deconv_1_16d(conv_1_32_3d)
        prelu_1_16d     = F.prelu(deconv_1_16d, torch.from_numpy(_weights_dict['prelu_1/16d']['weights']))
        bn_1_16d        = self.bn_1_16d(prelu_1_16d)

        pool_SE_1_16    = F.avg_pool2d(concat_4_24, kernel_size=(30, 30), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        conv_SE_1_16_1  = self.conv_SE_1_16_1(pool_SE_1_16)
        relu_SE_1_16_1  = F.relu(conv_SE_1_16_1)
        conv_SE_1_16_2  = self.conv_SE_1_16_2(relu_SE_1_16_1)
        sigm_SE_1_16    = F.sigmoid(conv_SE_1_16_2)
        reshape_SE_1_16 = torch.reshape(input = sigm_SE_1_16, shape = (1,1024,1,1))
        scale_SE_1_16   = concat_4_24 * reshape_SE_1_16
        conv_SE_1_16    = self.conv_SE_1_16(scale_SE_1_16)
        bn_SE_1_16      = self.bn_SE_1_16(prelu_SE_1_16)
        #########################################

        ################ BLOCK 2 ################
        concat_1_16d    = torch.cat((bn_1_16d, bn_SE_1_16,), 1)

        conv_1_16_1d_pad = F.pad(concat_1_16d, (1, 1, 1, 1))
        conv_1_16_1d    = self.conv_1_16_1d(conv_1_16_1d_pad)
        prelu_1_16_1d   = F.prelu(conv_1_16_1d, torch.from_numpy(_weights_dict['prelu_1/16_1d']['weights']))
        bn_1_16_1d      = self.bn_1_16_1d(prelu_1_16_1d)

        conv_1_16_2d_pad = F.pad(bn_1_16_1d, (1, 1, 1, 1))
        conv_1_16_2d    = self.conv_1_16_2d(conv_1_16_2d_pad)
        prelu_1_16_2d   = F.prelu(conv_1_16_2d, torch.from_numpy(_weights_dict['prelu_1/16_2d']['weights']))
        bn_1_16_2d      = self.bn_1_16_2d(prelu_1_16_2d)

        conv_1_16_3d    = self.conv_1_16_3d(bn_1_16_2d)
        prelu_1_16_3d   = F.prelu(conv_1_16_3d, torch.from_numpy(_weights_dict['prelu_1/16_3d']['weights']))
        bn_1_16_3d      = self.bn_1_16_3d(prelu_1_16_3d)

        deconv_1_8d     = self.deconv_1_8d(conv_1_16_3d)
        prelu_1_8d      = F.prelu(deconv_1_8d, torch.from_numpy(_weights_dict['prelu_1/8d']['weights']))
        bn_1_8d         = self.bn_1_8d(prelu_1_8d)

        pool_SE_1_8     = F.avg_pool2d(concat_3_12, kernel_size=(60, 60), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        conv_SE_1_8_1   = self.conv_SE_1_8_1(pool_SE_1_8)        
        relu_SE_1_8_1   = F.relu(conv_SE_1_8_1)
        conv_SE_1_8_2   = self.conv_SE_1_8_2(relu_SE_1_8_1)
        sigm_SE_1_8     = F.sigmoid(conv_SE_1_8_2)
        reshape_SE_1_8  = torch.reshape(input = sigm_SE_1_8, shape = (1,512,1,1))
        scale_SE_1_8    = concat_3_12 * reshape_SE_1_8
        conv_SE_1_8     = self.conv_SE_1_8(scale_SE_1_8)
        prelu_SE_1_8    = F.prelu(conv_SE_1_8, torch.from_numpy(_weights_dict['prelu_SE_1/8']['weights']))
        bn_SE_1_8       = self.bn_SE_1_8(prelu_SE_1_8)
        #########################################

        ################ BLOCK 3 ################                                                               
        concat_1_8d     = torch.cat((bn_1_8d, bn_SE_1_8,), 1)

        conv_1_8_1d_pad = F.pad(concat_1_8d, (1, 1, 1, 1))
        conv_1_8_1d     = self.conv_1_8_1d(conv_1_8_1d_pad)
        prelu_1_8_1d    = F.prelu(conv_1_8_1d, torch.from_numpy(_weights_dict['prelu_1/8_1d']['weights']))
        bn_1_8_1d       = self.bn_1_8_1d(prelu_1_8_1d)

        conv_1_8_2d_pad = F.pad(bn_1_8_1d, (1, 1, 1, 1))
        conv_1_8_2d     = self.conv_1_8_2d(conv_1_8_2d_pad)
        prelu_1_8_2d    = F.prelu(conv_1_8_2d, torch.from_numpy(_weights_dict['prelu_1/8_2d']['weights']))
        bn_1_8_2d       = self.bn_1_8_2d(prelu_1_8_2d)

        conv_1_8_3d     = self.conv_1_8_3d(bn_1_8_2d)
        prelu_1_8_3d    = F.prelu(conv_1_8_3d, torch.from_numpy(_weights_dict['prelu_1/8_3d']['weights']))
        bn_1_8_3d       = self.bn_1_8_3d(prelu_1_8_3d)

        deconv_1_4d     = self.deconv_1_4d(conv_1_8_3d)
        prelu_1_4d      = F.prelu(deconv_1_4d, torch.from_numpy(_weights_dict['prelu_1/4d']['weights']))
        bn_1_4d         = self.bn_1_4d(prelu_1_4d)

        pool_SE_1_4     = F.avg_pool2d(concat_2_6, kernel_size=(120, 120), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        conv_SE_1_4_1   = self.conv_SE_1_4_1(pool_SE_1_4)
        relu_SE_1_4_1   = F.relu(conv_SE_1_4_1)
        conv_SE_1_4_2   = self.conv_SE_1_4_2(relu_SE_1_4_1)
        sigm_SE_1_4     = F.sigmoid(conv_SE_1_4_2)
        reshape_SE_1_4  = torch.reshape(input = sigm_SE_1_4, shape = (1,256,1,1))
        scale_SE_1_4    = concat_2_6 * reshape_SE_1_4
        conv_SE_1_4     = self.conv_SE_1_4(scale_SE_1_4)
        prelu_SE_1_4    = F.prelu(conv_SE_1_4, torch.from_numpy(_weights_dict['prelu_SE_1/4']['weights']))
        bn_SE_1_4       = self.bn_SE_1_4(prelu_SE_1_4)
        #########################################

        ################ BLOCK 4 ################                                                               
        concat_1_4d     = torch.cat((bn_1_4d, bn_SE_1_4,), 1)

        conv_1_4_1d_pad = F.pad(concat_1_4d, (1, 1, 1, 1))
        conv_1_4_1d     = self.conv_1_4_1d(conv_1_4_1d_pad)
        prelu_1_4_1d    = F.prelu(conv_1_4_1d, torch.from_numpy(_weights_dict['prelu_1/4_1d']['weights']))
        bn_1_4_1d       = self.bn_1_4_1d(prelu_1_4_1d)

        conv_1_4_2d_pad = F.pad(bn_1_4_1d, (1, 1, 1, 1))
        conv_1_4_2d     = self.conv_1_4_2d(conv_1_4_2d_pad)
        prelu_1_4_2d    = F.prelu(conv_1_4_2d, torch.from_numpy(_weights_dict['prelu_1/4_2d']['weights']))
        bn_1_4_2d       = self.bn_1_4_2d(prelu_1_4_2d)

        conv_1_4_3d     = self.conv_1_4_3d(bn_1_4_2d)
        prelu_1_4_3d    = F.prelu(conv_1_4_3d, torch.from_numpy(_weights_dict['prelu_1/4_3d']['weights']))
        bn_1_4_3d       = self.bn_1_4_3d(prelu_1_4_3d)
        #########################################

        ################ PREDICTION AT 1/4 ################
        pred_1_4        = self.pred_1_4(bn_1_4_3d)

        ################ UNSAMPLE THE PREDICTION FROM 1/4 TO 1/1 ################
        pred_step_1     = self.pred_step_1(pred_1_4)
        sigp_step_1     = F.sigmoid(pred_step_1)

        ##############################################

        ##################### FINE DECODER #####################
        concat_step_1   = torch.cat((concat_input, sigp_step_1,), 1)

        ################ ATROUS POOLING BLOCK 1 ################
        conv_atrous1_1_pad = F.pad(concat_step_1, (1, 1, 1, 1))
        conv_atrous1_1  = self.conv_atrous1_1(conv_atrous1_1_pad)
        prelu_atrous1_1 = F.prelu(conv_atrous1_1, torch.from_numpy(_weights_dict['prelu_atrous1_1']['weights']))
        bn_atrous1_1    = self.bn_atrous1_1(prelu_atrous1_1)

        conv_atrous1_2_pad = F.pad(bn_atrous1_1, (1, 1, 1, 1))
        conv_atrous1_2  = self.conv_atrous1_2(conv_atrous1_2_pad)
        prelu_atrous1_2 = F.prelu(conv_atrous1_2, torch.from_numpy(_weights_dict['prelu_atrous1_2']['weights']))
        bn_atrous1_2    = self.bn_atrous1_2(prelu_atrous1_2)

        conv_atrous1_3  = self.conv_atrous1_3(bn_atrous1_2)
        prelu_atrous1_3 = F.prelu(conv_atrous1_3, torch.from_numpy(_weights_dict['prelu_atrous1_3']['weights']))
        bn_atrous1_3    = self.bn_atrous1_3(prelu_atrous1_3)
        ########################################################

        ################ ATROUS POOLING BLOCK 2 ################
        conv_atrous2_1_pad = F.pad(concat_step_1, (2, 2, 2, 2))
        conv_atrous2_1  = self.conv_atrous2_1(conv_atrous2_1_pad)
        prelu_atrous2_1 = F.prelu(conv_atrous2_1, torch.from_numpy(_weights_dict['prelu_atrous2_1']['weights']))
        bn_atrous2_1    = self.bn_atrous2_1(prelu_atrous2_1)

        conv_atrous2_2_pad = F.pad(bn_atrous2_1, (1, 1, 1, 1))
        conv_atrous2_2  = self.conv_atrous2_2(conv_atrous2_2_pad)
        prelu_atrous2_2 = F.prelu(conv_atrous2_2, torch.from_numpy(_weights_dict['prelu_atrous2_2']['weights']))
        bn_atrous2_2    = self.bn_atrous2_2(prelu_atrous2_2)

        conv_atrous2_3  = self.conv_atrous2_3(bn_atrous2_2)
        prelu_atrous2_3 = F.prelu(conv_atrous2_3, torch.from_numpy(_weights_dict['prelu_atrous2_3']['weights']))
        bn_atrous2_3    = self.bn_atrous2_3(prelu_atrous2_3)
        ########################################################

        ################ ATROUS POOLING BLOCK 3 ################
        conv_atrous3_1_pad = F.pad(concat_step_1, (3, 3, 3, 3))
        conv_atrous3_1  = self.conv_atrous3_1(conv_atrous3_1_pad)
        prelu_atrous3_1 = F.prelu(conv_atrous3_1, torch.from_numpy(_weights_dict['prelu_atrous3_1']['weights']))
        bn_atrous3_1    = self.bn_atrous3_1(prelu_atrous3_1)

        
        conv_atrous3_2_pad = F.pad(bn_atrous3_1, (1, 1, 1, 1))
        conv_atrous3_2  = self.conv_atrous3_2(conv_atrous3_2_pad)
        prelu_atrous3_2 = F.prelu(conv_atrous3_2, torch.from_numpy(_weights_dict['prelu_atrous3_2']['weights']))
        bn_atrous3_2    = self.bn_atrous3_2(prelu_atrous3_2)
        
        
        conv_atrous3_3  = self.conv_atrous3_3(bn_atrous3_2)
        prelu_atrous3_3 = F.prelu(conv_atrous3_3, torch.from_numpy(_weights_dict['prelu_atrous3_3']['weights']))
        bn_atrous3_3    = self.bn_atrous3_3(prelu_atrous3_3)
        ########################################################

        ################ CONCAT + SQUEEZ & EXCITATION ################
        concat_step_2   = torch.cat((bn_atrous1_3, bn_atrous2_3, bn_atrous3_3,), 1)
        gpool_s2        = F.avg_pool2d(concat_step_2, kernel_size=(480, 480), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        conv_s2_down    = self.conv_s2_down(gpool_s2)
        relu_s2_down    = F.relu(conv_s2_down)
        conv_s2_up      = self.conv_s2_up(relu_s2_down)
        sig_s2_up       = F.sigmoid(conv_s2_up)
        resh_s2         = torch.reshape(input = sig_s2_up, shape = (1,48,1,1))
        scale_s2        = concat_step_2 * resh_s2
        ##############################################################

        ################ PREDICTION ################
        conv_p1_1_pad   = F.pad(scale_s2, (1, 1, 1, 1))
        conv_p1_1       = self.conv_p1_1(conv_p1_1_pad)
        prelu_p1_1      = F.prelu(conv_p1_1, torch.from_numpy(_weights_dict['prelu_p1_1']['weights']))
        bn_p1_1         = self.bn_p1_1(prelu_p1_1)
        pred_step_2     = self.pred_step_2(bn_p1_1)
        ############################################

        ################ PREDICTION ################
        sig_pred        = F.sigmoid(pred_step_2)
        ############################################
        return sig_pred

        ################################

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

def densenet121(pretrained=False, **kwargs):
    model = Encoder(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            model_urls['densenet121'], map_location='cpu'))
    return model
