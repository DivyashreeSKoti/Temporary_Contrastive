#!/usr/bin/env python3
# coding: utf-8

import time

# Keep track of run time
start_time = time.time()

# ********* Start ************
# This section is to ensure to get the right directory when submitting a job
import sys
import os
current_dir = os.path.abspath('./')
parent_dir = os.path.abspath('./../')
sys.path.append(current_dir)
sys.path.append(parent_dir)

# Set the working directory to the directory containing the script
custom_path = current_dir

# Get the absolute path of the current script
script_dir = os.path.abspath(custom_path)

# ********* END ************

# ********* Start of main code *************
import DataLoader as DL
import SetTransformer_Extrapolating as ST
import ContrastiveModel as CM

#to plot data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader
import random
from torch.optim.lr_scheduler import ExponentialLR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from itertools import islice
import numpy as np

# To track the total number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

folder_path = "../../../Data/3dbsf_txt/"

min_lines, max_lines = DL.get_min_max_lines(folder_path)
print(f"The minimum number of lines among all files: {min_lines}")
print(f"The maximum number of lines among all files: {max_lines}")

# Load data and corresponding targets
dataset, targets, _ = DL.load_dataset(folder_path, num_lines = min_lines)

# Get device to run on available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom data split 80:20
splitter = DL.DatasetSplitter(validation_ratio=0.2, shuffle = True)
train_data,train_targets, val_data, val_targets = splitter.split_dataset(dataset, targets)

# Custom collate function to subssample the point cloud data
def custom_collate(subsample_size):
    def collate_fn(batch):
        subsamples = []
        for data, target in batch:
            num_samples = data.shape[0]
            current_subsample_size = min(subsample_size, num_samples)
            indices = random.sample(range(num_samples), subsample_size)
            subsample = data[indices]
            subsamples.append((subsample, target))

        data, targets = zip(*subsamples)
        data = torch.stack(data, dim=0)
        targets = torch.stack(targets, dim=0)

        return data, targets
    
    return collate_fn


# Define the batch size, training subsample size and validation subsample size
batch_size = 8
subsample_size = 8000

# Load the batches using CustomDataLoader as to create more number of batches with replacement
train_subsampled_dataloader = DL.CustomDataLoader(train_data, train_targets, batch_size=batch_size, num_batches=80, subsample_size=subsample_size, shuffle=True, augmentation_by_cube=True)
val_subsampled_dataloader = DL.CustomDataLoader(val_data, val_targets, batch_size=batch_size, num_batches=50, subsample_size=subsample_size, shuffle=True, augmentation_by_cube=True)

########### Strat to remove visalize ############

index = 12 # the index of the batch to show
t = train_subsampled_dataloader[index]
print("The digit is a", train_targets[index])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
t0 = t[0][0]
ax.scatter(*t0.T,s=0.1)
ax.view_init(elev=0, azim=0)
plt.show()
fp = os.path.join(script_dir)
# Resolve the absolute path
fp = os.path.abspath(fp)
temp_folder_path = fp + '/Example_Cube_visualize'
visualize = temp_folder_path+'/Train_'+str(batch_size)+str(t0.shape[0])+'.png'
print('-----',visualize)
plt.savefig(visualize)

print("The digit is a", val_targets[index])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
t1 = t[1][0]
ax.scatter(*t1.T,s=0.1)
ax.view_init(elev=0, azim=0)
plt.show()
visualize = temp_folder_path+'/Train2_'+str(batch_size)+str(t1.shape[0])+'.png'
print('-----',visualize)
plt.savefig(visualize)

t = val_subsampled_dataloader[index]
print("The digit is a", train_targets[index])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
t0 = t[0][0]
ax.scatter(*t0.T,s=0.1)
ax.view_init(elev=0, azim=0)
plt.show()
visualize = temp_folder_path+'/Val_'+str(batch_size)+str(t0.shape[0])+'.png'
print('-----',visualize)
plt.savefig(visualize)

print("The digit is a", val_targets[index])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
t1 = t[1][0]
ax.scatter(*t1.T,s=0.1)
ax.view_init(elev=0, azim=0)
plt.show()
visualize = temp_folder_path+'/Val2_'+str(batch_size)+str(t1.shape[0])+'.png'
print('-----',visualize)
plt.savefig(visualize)

########### END to remove ############

# Track the number of batches
subsampled_total_DLbatches = len(train_subsampled_dataloader)
subsampled_val_DLbatches = len(val_subsampled_dataloader)

# Define architecture for the model
embed_dim = 64
num_heads = 16
num_induce = 128
stack=3
ff_activation="gelu"
dropout=0.1
use_layernorm=False
pre_layernorm=False
is_final_block = False
num_classes = dataset.shape[0]

# Create an instances of the PyTorch model for contrastive learning base model
masked_encoder = ST.PyTorchModel(
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_induce=num_induce,
    stack=stack,
    ff_activation=ff_activation,
    dropout=dropout,
    use_layernorm=use_layernorm,
    is_final_block = is_final_block,
    num_classes = num_classes
)

unmasked_encoder = ST.PyTorchModel(
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_induce=num_induce,
    stack=stack,
    ff_activation=ff_activation,
    dropout=dropout,
    use_layernorm=use_layernorm,
    is_final_block = is_final_block,
    num_classes = num_classes
)

# Placing model onto GPU
masked_encoder.to(device)
unmasked_encoder.to(device)

# Define Contrastive model
cm = CM.ContrastiveModel(device, masked_encoder=masked_encoder, unmasked_encoder=unmasked_encoder, embed_dim = embed_dim)
cm.to(device)


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Define the number of training epochs
num_epochs = 250

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    total_acc = 0
    val_total_loss = 0
    val_total_acc = 0
    # Convert the training data to PyTorch tensors
    for i, (batch_data_1,batch_data_2) in enumerate(train_subsampled_dataloader):
        batch_data_1 = torch.tensor(batch_data_1)
        batch_data_2 = torch.tensor(batch_data_2)
        cm.train()
        batch_data_1 = batch_data_1.to(device)
        batch_data_2 = batch_data_2.to(device)
        train_loss, contrastive_acc = cm.train_step((batch_data_1, batch_data_2))
        train_loss = train_loss.item()
        total_loss += train_loss
        total_acc += contrastive_acc

        # Print progress
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Progress: {i+1}/{subsampled_total_DLbatches}, Train Loss: {train_loss:.4f}, ContrastiveAccuracy: {contrastive_acc: .4f}", end="")
        
        # Unload tensor from device
        batch_data_2 = batch_data_2.cpu()
        batch_data_1 = batch_data_1.cpu()
        
    for i, (batch_data_1,batch_data_2) in enumerate(val_subsampled_dataloader):
        batch_data_1 = torch.tensor(batch_data_1)
        batch_data_2 = torch.tensor(batch_data_2)
        cm.eval()
        batch_data_1 = batch_data_1.to(device)
        batch_data_2 = batch_data_2.to(device)
        val_loss, contrastive_acc = cm.val_step((batch_data_1, batch_data_2))
        val_loss = val_loss.item()
        val_total_loss += val_loss
        val_total_acc += contrastive_acc
        
        # Unload tensor from device
        batch_data_2 = batch_data_2.cpu()
        batch_data_1 = batch_data_1.cpu()

    # Compute the average train and validation loss and accuracy
    avg_train_loss = total_loss/subsampled_total_DLbatches
    avg_train_acc = total_acc/subsampled_total_DLbatches
    avg_val_loss = val_total_loss/subsampled_val_DLbatches
    avg_val_acc = val_total_acc/subsampled_val_DLbatches
    
    # keep track of loss and accuracy for Monte carlo simulation
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_acc)
    # Print final results of epoch
    print(f"\rEpoch [{epoch+1}/{num_epochs}], Progress: {i+1}/{subsampled_total_DLbatches}, Train Loss: {avg_train_loss:.4f}, Train ContrastiveAccuracy: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val ContrastiveAccuracy: {avg_val_acc:.4f}", end="")
    print('')
    
    # Empty GPU cache
    torch.cuda.empty_cache()
    
print("Validation accuracy:",*["%.8f"%(x) for x in val_accuracies])
print("Train accuracy:",*["%.8f"%(x) for x in train_accuracies])
print("Validation Loss:",*["%.8f"%(x) for x in val_losses])
print("Train Loss:",*["%.8f"%(x) for x in train_losses])


print('Details:', 'embed_dim =', embed_dim,
'num_heads =', num_heads,
'num_induce =', num_induce,
'stack =', stack,
'dropout =', dropout, 
'batch_size =', batch_size,
'subsample_size =', subsample_size, 'projection_dim = 1024')
fp = os.path.join(script_dir)
fp = os.path.abspath(fp)
masked_encoder_model = fp+'/saved_models/files99/cubecutout'+'/masked_encoder_'+str(embed_dim)+'_'+str(num_heads)+'_'+str(num_induce)+'_'+str(stack)+'_'+str(dropout)+str(subsample_size)+'2.pth'
unmasked_encoder_model = fp+'/saved_models/files99/cubecutout'+'/unmasked_encoder_'+str(embed_dim)+'_'+str(num_heads)+'_'+str(num_induce)+'_'+str(stack)+'_'+str(dropout)+str(subsample_size)+'2.pth'
# Save the ContrastiveModel object
torch.save(masked_encoder, masked_encoder_model)
torch.save(unmasked_encoder, unmasked_encoder_model)




# Stop the timer
end_time = time.time()

# Calculate the total time
total_time = end_time - start_time

# Print the total time
print("Total time:", total_time, "seconds")
