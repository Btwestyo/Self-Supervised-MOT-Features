import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision.models as models

import cv2


class Fully_Conv_AE(nn.Module):
    
    def __init__(self, latent_dim_size=512, dataset='mot17', use_cuda=False):
        super(Fully_Conv_AE, self).__init__()

        self.latent_dim_size = latent_dim_size

        self.num_of_params = 1
        kernel_size = 1

        self.encoder = nn.Sequential(
            # conv 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv 5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # conv 6
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # conv 7
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # conv 8
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # conv 9
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # conv 10 out
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()   # multi-class classification
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def compute_accuracy(self,x,y):
        return torch.tensor(0)

if __name__ == '__main__':

    input = torch.randn(5, 3, 112, 112)
    output = Fully_Conv_AE()(input)
    print(input.shape)
    print(output.shape)
    model = Fully_Conv_AE()
    encoded_output = model.encoder(input)
    print(encoded_output.shape)

#    
#    print(model)