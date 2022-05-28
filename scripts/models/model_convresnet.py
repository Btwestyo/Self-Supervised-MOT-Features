import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision.models as models

import cv2

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CustomResNetConv(nn.Module):
    def __init__(self, hidden_dim=512, classes=4, device=DEVICE):
        super(CustomResNetConv, self).__init__()
        self.resnet =  models.resnet18(pretrained=True)

        # remove the resnet linear layer
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.custom_layers = nn.Sequential(nn.Flatten(),
                                           nn.Linear(512, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, classes),
                                           nn.Softmax(dim=1))
        init_weights(self.custom_layers)

        n_blocks = len(self.resnet)

        # freeze all layers before the last block
        for param in self.resnet[:n_blocks-2].parameters():
            param.requires_grad = False

        # reinit the last block of weights
        init_weights(self.resnet[n_blocks-2])

        # turn on gradients for last plock
        for name, param in self.resnet[n_blocks-2].named_parameters():
            param.requires_grad = True

        # turn on gradients for custom layers
        for param in self.custom_layers.parameters():
          param.requires_grad = True

        # print("Parameters that have gradients turned on")

        # print("Custom layers:")
        # for name, param in self.resnet.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        # print("Custom layers:")
        # for name, param in self.custom_layers.named_parameters():
        #     if param.requires_grad:
        #         print(name)

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

if __name__ == '__main__':

    model = CustomResNetConv()
    print(model)
