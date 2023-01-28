import torch.nn as nn
import torch.optim.lr_scheduler
import torch.nn.init

from torch.nn import Module, Sequential, Conv3d
#from torch.nn import Module, Sequential, Conv3d
import torch.nn.functional as F



def count_parameters(module: Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_conv3d(module: Module):
    return len([m for m in module.modules() if isinstance(m, Conv3d)])


'''
MixedScaleDenseLayer is the basic layer, functions like Conv2d, 
however, its input feature map number is linearly increasing as
net depth increases;

Net class is used to build a MS-D net
num_layers: depth of MS-D net
growth_rate: width in MS-D net paper, that is w
kernel_size: in MS-D, there is only one kind of kernel, such as 3*3
dilation_mod: in MS-D, every ten layers combine a weak block, we will talk it
              in detail in readme.txt

'''


class MixedScaleDenseLayer(Module):
    def __init__(self, in_channels, dilations, kernel_size=3, dilation_mod=10):
        super(MixedScaleDenseLayer, self).__init__()

        if type(dilations) == int:
            dilations = [j % dilation_mod + 1 for j in range(dilations)]

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = in_channels + len(dilations)

        for j, dilation in enumerate(dilations):
            # Equal to: kernel_size + (kernel_size - 1) * (dilation - 1)
            dilated_kernel_size = (kernel_size - 1) * dilation + 1
            padding = dilated_kernel_size // 2
            self.add_module(f'conv_{j}', Conv3d(
                in_channels, 1,
                kernel_size=kernel_size, dilation=dilation, padding=padding
            ))

    def forward(self, x):
        return torch.cat((x,) + tuple(c(x) for c in self.children()), dim=1)


class MSDNet(Sequential):
    def __init__(self, in_channels=1, input_size=(224,224,224), out_channels=1, output_dim=40, growth_rate=1, kernel_size=3,
                 dilation_mod=[4, 3, 2], layers_number = [8,6,4]):
        super(MSDNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        current_channels = in_channels
        self.output_dim = output_dim
        self.dilation_mod = dilation_mod

        k = 0
        for s, layer_num_mod in enumerate(layers_number):
            dilation_mod_s = dilation_mod[s]
            for i in range(layer_num_mod):
                dilations = [((i * growth_rate + j) % dilation_mod_s) +
                             1 for j in range(growth_rate)]
                l = MixedScaleDenseLayer(current_channels, dilations, kernel_size, dilation_mod_s)
                current_channels = l.out_channels
                self.add_module(f'layer_{k, i}', l)
                self.add_module(f'relu_{k, i}', nn.LeakyReLU())
            # print(current_channels)
            # nn.BatchNorm3d(out_channel)
            # self.add_module(f'last_{k}', Conv2d(current_channels, 3,  kernel_size=1, padding=0))
            # self.admodule(f'last_activation_{k}', nn.LeakyReLU())
            self.add_module(f'last_conv_{k}', nn.Conv3d(current_channels, current_channels, kernel_size=3, padding=1))
            self.add_module(f'last_bn_{k}', nn.BatchNorm3d(current_channels))
            self.add_module(f'last_pooling_{k}', nn.MaxPool3d(2, stride=2))
            self.add_module(f'relu_{k}', nn.LeakyReLU())
            # current_channels = 3
            k += 1

        self.add_module('last_conv', Conv3d(current_channels, 1, kernel_size=3, padding=1))
        self.add_module('last_conv_activation', nn.LeakyReLU())
        self.add_module('last_flatten', nn.Flatten())
        linear_length = (input_size[0] // (2 ** k)) * (input_size[1] // (2 ** k)) * (input_size[2] // (2 ** k))
        self.add_module('last_fc', nn.Linear(linear_length, self.output_dim))
        # self.add_module('last', Conv2d(current_channels, out_channels, kernel_size=1))
        # self.add_module('last_bn', nn.BatchNorm1d(self.output_dim))
        # self.add_module('last_activation', nn.LeakyReLU())
        self.add_module('last_softmax', torch.nn.LogSoftmax(dim=1)) # dim=1

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        '''




def MSD3D(cfg):
    in_channels = cfg.NET_INPUT.IN_CHANNELS
    input_size = cfg.NET_INPUT.CROP_SIZE
    out_channels = cfg.NET_MODEL.OUT_CHANNELS
    output_dim = cfg.NET_MODEL.CLASS_NUM

    growth_rate = cfg.MSD_MODEL.GROWTH_RATE
    kernel_size = cfg.MSD_MODEL.KERNEL_SIZE
    dilation_mod = cfg.MSD_MODEL.DILATION_MOD
    layers_number = cfg.MSD_MODEL.LAYERS_NUMBER

    return MSDNet(in_channels=in_channels, input_size=input_size, out_channels=out_channels,
                  output_dim=output_dim, growth_rate=growth_rate, kernel_size=kernel_size,
                  dilation_mod=dilation_mod, layers_number=layers_number)

def MSD9(input_size = (224,224,224), output_dim=512):
    return MSDNet(in_channels=1, input_size=input_size, out_channels=1, output_dim=output_dim, growth_rate=1, kernel_size=3,
                  dilation_mod=[4, 3, 2])

def MSD12(input_size = (224,224,224), output_dim=512):
    return MSDNet(in_channels=1, input_size=input_size, out_channels=1, output_dim=output_dim, growth_rate=1, kernel_size=3,
                  dilation_mod=[5, 4, 3])




