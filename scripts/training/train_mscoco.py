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
from scripts.solver import Solver
from scripts.plotting.plot_results import plot_loss_and_acc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dir = '/Users/brianweston/Documents/Stanford_CS/CS231N_CNN/Project/cs231n_final_project/data/coco_small' #coco_unlabeled2017_mid'
test_dir = '/Users/brianweston/Documents/Stanford_CS/CS231N_CNN/Project/cs231n_final_project/data/coco_small' #coco_unlabeled2017_mid'
#train_dir = data_dir + '/train'
#test_dir  = data_dir + '/test'

training_data = VanillaImageDataset(train_dir)
#print(type(training_data))
testing_data = VanillaImageDataset(test_dir)
#m=len(training_data)
#print(m)
batch_size = 64 #64
#print(training_data[0].shape)


train_loader = DataLoader(training_data, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(testing_data, batch_size = batch_size, shuffle=True)

fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    # random.choice allows to randomly sample from a list-like object (basically anything that can be accessed with an index, like our dataset)
    #print(axs.flatten().shape)
    img, _ = random.choice(training_data) #.numpy()
    img = img.permute(1, 2, 0)
    image = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
fig.savefig('full_figure.png')


# helper function to un-normalize and display an image
def imshow(img):
    img = img  # unnormalize
    #plt.imshow(img[0,:,:,:])
    img = np.transpose(img, (1, 2, 0))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    #plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


# import convolutional autoencoder
model = Conv_AE(1000,'mot17')

# specify loss function
criterion = nn.MSELoss() #BCE vs MSE
loss_fcn = criterion

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #weight_decay=1e-5

# number of epochs to train the model
n_epochs = 1

solver = Solver(model, optimizer, loss_fcn)

# train the model
solver.train(epochs=n_epochs, 
            train_data_loader=train_loader,
            val_data_loader=test_loader,
            save_path=None,#os.path.join(SCRIPT_DIR,'checkpoints'),
            save_every=1,
            print_every=64,
            verbose=True)

# plot training/validation accuracy and loss
plot_loss_and_acc(solver.train_loss_history, solver.train_acc_history)
plot_loss_and_acc(solver.val_loss_history, solver.val_acc_history)

# save final solver object
solver.save_solver(os.path.join(SCRIPT_DIR,'checkpoints','example_trained_model.pt')) 


dataiter = iter(train_loader)
images = dataiter.next()
print(len(images))
img = images[0][0,:,:,:]
print(img.shape)
encoded_output = model.encoder(img)
print(model)
print(f"max size: {torch.max(img)}, min size: {torch.min(img)}")


# prints number of parameters
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {pytorch_total_params} ")


# obtain one batch of test images
dataiter = iter(train_loader)
images = dataiter.next()

# get sample outputs
output = model(images[0])
# prep images for display
images = images[0].numpy()

# output is resized into a batch of images
output = output.view(-1, 3, 112, 112) #batch_size, 3, 32, 32 #224 #224
print(output.shape)
# use detach when it's an output that requires_grad
output = output.detach().numpy()


# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(1):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    fig.savefig('reconstructed_mscoco.png')
    #ax.set_title(classes[labels[idx]])
    
# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(1):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    fig.savefig('original_mscoco.png')
    #ax.set_title(classes[labels[idx]])

