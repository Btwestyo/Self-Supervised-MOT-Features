import numpy as np
import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from tqdm import tqdm
from time import sleep

# unfortunately this is required to use relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))

from scripts.models.model_conv_ae import Conv_AE
from scripts.dataloaders.dataloader_vanilla import VanillaImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_dir = '/Users/brianweston/Documents/Stanford_CS/CS231N_CNN/Project/cs231n_final_project/data/detection_bounding_boxes'
train_dir = data_dir + '/train'
test_dir  = data_dir + '/test'

training_data = VanillaImageDataset(train_dir)
testing_data = VanillaImageDataset(test_dir)
#m=len(training_data)
#print(m)
batch_size = 64
#print(training_data[0].shape)


train_loader = DataLoader(training_data, batch_size = batch_size, shuffle=False)
test_loader = DataLoader(testing_data, batch_size = batch_size, shuffle=False)

fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    # random.choice allows to randomly sample from a list-like object (basically anything that can be accessed with an index, like our dataset)
    #print(axs.flatten().shape)
    img = random.choice(testing_data).numpy()
    ax.imshow(np.transpose(img, (1, 2, 0)))
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
fig.savefig('full_figure.png')


# helper function to un-normalize and display an image
def imshow(img):
    img = img  # unnormalize
    #plt.imshow(img[0,:,:,:])
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    

# import convolutional autoencoder
model = Conv_AE(1000,'mot17')

dataiter = iter(train_loader)
images = dataiter.next()
print(images.shape)
img = images[0,:,:,:]
encoded_output = model.encoder(img)
print(model)

# prints number of parameters
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {pytorch_total_params} ")

# specify loss function
criterion = nn.BCELoss() #BCE vs MSE

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 1

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data in tqdm(train_loader):
        # _ stands in for labels, here
        # no need to flatten images
        images = data
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))

# obtain one batch of test images
dataiter = iter(test_loader)
images = dataiter.next()

# get sample outputs
output = model(images)
# prep images for display
images = images.numpy()

# output is resized into a batch of images
output = output.view(-1, 3, 224, 224) #batch_size, 3, 32, 32 #224 #224
print(output.shape)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    fig.savefig('reconstructed_mot17.png')
    #ax.set_title(classes[labels[idx]])
    
# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    fig.savefig('original_mot17.png')
    #ax.set_title(classes[labels[idx]])