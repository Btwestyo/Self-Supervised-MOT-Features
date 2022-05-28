import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from math import exp
from torch.autograd import Variable

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class UNet(nn.Module):
    def __init__(self, device=DEVICE):
        super(UNet, self).__init__()
        self.unet_full = torch.hub.load('milesial/Pytorch-UNet', 
                                    'unet_carvana', 
                                    pretrained=True, 
                                    scale=0.5
                                )

        # self.unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #             in_channels=3, out_channels=1, init_features=32, pretrained=True)

        # remove the last classifier layer
        self.unet = (list(self.unet_full.children())[:-1])

        # add a new layer to return the image to 3 channels
        self.custom_layer = nn.Sequential(nn.Conv2d(64, 3, 1))
        init_weights(self.custom_layer[0])

        # freeze all layers before the last block
        # for layer in self.unet:
        #     for param in layer.parameters():
        #         param.requires_grad = False

        # turn on gradients for last block
        for layer in self.unet[4:9]:
            for param in layer.parameters():
                param.requires_grad = True

        # re-init the weights 
        for i in range(4,9):
            init_weights(self.unet[i])

        # turn on gradients for custom layers
        for param in self.custom_layer.parameters():
          param.requires_grad = True

        self.inc = torch.nn.Sequential(nn.Conv2d(1,32,3,padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(32,64,3,padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.down1 = self.unet[1]
        self.down2 = self.unet[2]
        self.down3 = self.unet[3]
        self.down4 = self.unet[4]
        self.up1 = self.unet[5]
        self.up2 = self.unet[6]
        self.up3 = self.unet[7]
        self.up4 = self.unet[8]

        # print("Parameters that have gradients turned on")

        # print("Unet layers:")
        # for layer in self.unet:
        #     for name, param in layer.named_parameters():
        #         if param.requires_grad:
        #             print(layer)
        #             break
        #             # print(name)

        # print("Custom layers:")
        # for name, param in self.custom_layer.named_parameters():
        #     if param.requires_grad:
        #         print(name)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.custom_layer(x)

    def compute_accuracy(self, preds, target):
        """The training and testing scripts assume that there is a
        method in the model class that computes the accuracy. This is
        because we'll have different metrics of accuracy for each task,
        but we'd like to reuse training code."""
        return ssim(preds, target)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


"""The following code was taken from 
https://github.com/Po-Hsun-Su/pytorch-ssim/tree/master/pytorch_ssim as 
a fully differentiable pytorch implementation of the SSIM accuracy/loss
metric."""
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = f.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = f.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = f.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = f.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = f.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

if __name__ == '__main__':

    model = UNet()
    print(model)
