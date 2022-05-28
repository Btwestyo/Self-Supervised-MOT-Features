import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision.models as models

import cv2


class Conv_AE(nn.Module):
    
    def __init__(self, latent_dim_size=1000, dataset='mot17', use_cuda=False):
        super(Conv_AE, self).__init__()

        self.latent_dim_size = latent_dim_size

        self.num_of_params = 1
        kernel_size = 1

        if dataset == 'mot17':
            self.num_of_params = 7*7 #14*14 #14 # 14 for MOT17 and 4 for Cifar10
            kernel_size = 14
        else:
            self.num_of_params = 2*2
            kernel_size = 2

        self.encoder = nn.Sequential(
            # Input size for Cifar 10 is 32 x 32 x 3. Formula = (N - f + 2p) / s + 1
            # Input for MOT17 is 224 x 224 x 3. 
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            # Bottleneck size for Cifar 10 is 2 x 2 x 128 and MOT17 is 14 x 14 x 128
        )
        #self.fc1 = nn.Sequential(                    
        #    nn.Linear(self.num_of_params*128, self.latent_dim_size),
        #    nn.ReLU(),
        #)

        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(self.latent_dim_size, 128, kernel_size=kernel_size, stride=1), # kernel_size = 2, stride 1 for Cifar10 and kernel_size = 14, stride = 1 for MOT17
            #nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4,  stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x_recon = self.decoder(x)       
        #print(x.shape)
        #z = self.fc1(x.view(-1, self.num_of_params*128))
        #print(z.shape)
        #z = self.fc1(x)
        #x_recon = self.decoder(z.view(-1, self.latent_dim_size, 1, 1))              
        return x_recon
    
    
    # TODO: hopefully this does not mess up inference pipelines when loading models
    def get_latent_vec(self, x):
        x = self.encoder(x)
        z = self.fc1(x.view(-1, self.num_of_params*128))
        return z
        
    def get_recon_from_latent_vec(self, latent_vec):
        x_recon = self.decoder(latent_vec.view(-1, self.latent_dim_size, 1, 1))
        return x_recon

    def compute_accuracy(self,x,y):
        return torch.tensor(0)

if __name__ == '__main__':

    model = Conv_AE(1000,'mot17')