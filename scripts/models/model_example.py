import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


"""Here's an example model. Feel free to use this structure.
    - Bradley"""

# check if GPU is avaliable
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvClassifier(nn.Module):
    """Define a fully convolutional network with batchnorm and ReLU
    activations that accepts CxHxW image and uses 
    (base_filt*dim_reduce)x(H/4)x(W/4) feature maps to predict class
    scores for a classification task."""

    def __init__(self, in_ch=3, in_H=28, in_W=28, base_filt=8, 
                    dim_reduce=4, classes=10, device=DEVICE):
        super(ConvClassifier, self).__init__()

        out_1 = int(base_filt)
        out_2 = int(base_filt * dim_reduce)
        out_3 = int(out_2 * (in_H/dim_reduce) * (in_W/dim_reduce))

        self.model = nn.Sequential(

            ConvLayer2D(in_ch, out_1, kernel=3, stride=1, padding=1, device=device),
            BNLayer(out_1), ReLULayer(),

            ConvLayer2D(out_1, out_1, kernel=3, stride=1, padding=1, device=device),
            BNLayer(out_1), ReLULayer(),

            ConvLayer2D(out_1, out_2, kernel=5, stride=2, padding=2, device=device),
            BNLayer(out_2), ReLULayer(),

            ConvLayer2D(out_2, out_2, kernel=3, stride=1, padding=1, device=device),
            BNLayer(out_2), ReLULayer(),

            ConvLayer2D(out_2, out_2, kernel=3, stride=1, padding=1, device=device),
            BNLayer(out_2), ReLULayer(),

            torch.nn.Flatten(),

            Affine(out_3, 10),
        )

    def forward(self, input):
        """Input is a batch set of C-channel images."""
        return self.model(input)

    def compute_accuracy(self, scores, target):
        preds = torch.argmax(scores, dim=1)
        num_correct = torch.sum(preds==target)
        return num_correct / len(target.reshape(-1,))

def Affine(in_features, out_features, device=None):

    return nn.Linear(in_features=in_features, out_features=out_features,
                bias=True, device=device)

def ConvLayer2D(in_ch, out_ch, stride=1, kernel=3, device=None, 
            padding=0, padmode='zeros', bias=True):

    return nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
        padding=padding, padding_mode=padmode, bias=bias, device=device)

def BNLayer(in_ch, eps=1e-05, momentum=0.1, device=None):

    return nn.BatchNorm2d(in_ch, eps=eps, momentum=momentum, 
                            track_running_stats=True, device=device)

def ReLULayer():
    return nn.ReLU()