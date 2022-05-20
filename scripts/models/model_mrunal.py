import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision.models as models

import cv2

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.resnet =  models.resnet18(pretrained=True)
        # self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        # self.pretrained = nn.Sequential(*list(self.resnet.modules())[:-1])
        self.custom_layers = nn.Sequential(nn.Linear(1000, 512),
                                           nn.ReLU(),
                                           nn.Linear(512, 128),
                                           nn.Softmax(dim=0))
        init_weights(self.custom_layers)

        for param in self.resnet.parameters():
          param.requires_grad = False
        self.resnet.requires_grad = False

        for param in self.custom_layers.parameters():
          param.requires_grad = True

        print("Parameters that have gradients turned on")
        print("Custom layers:")
        for name, param in self.custom_layers.named_parameters():
            if param.requires_grad:
                print(name)

        print("Resnet:")
        for name, param in self.resnet.named_parameters():
            if param.requires_grad:
                print(name)

    def forward(self, x):
        x = self.resnet(x)
        x = self.custom_layers(x)
        return x

    def compute_accuracy(self, scores, target):
        """The training and testing scripts assume that there is a
        method in the model class that computes the accuracy. This is
        because we'll have different metrics of accuracy for each task,
        but we'd like to reuse training code."""
        preds = torch.argmax(scores, dim=1)
        num_correct = torch.sum(preds==target)
        return num_correct / len(target.reshape(-1,))

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)