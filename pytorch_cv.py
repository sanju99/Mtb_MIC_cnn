import subprocess, sys, glob, os, yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import torch.optim as optim
import timeit
from sklearn.model_selection import KFold
from tqdm import tqdm

# import dataloader and model
from pytorch_utils import MtbGeneDataset, CNN

_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file, "r"))

alignment_names = kwargs["aln_lst"]
data_file = kwargs["data_file"]
drug = kwargs["drug"]
longest_locus = kwargs["longest_locus"]
filter_size = kwargs["filter_size"]
num_loci = kwargs["num_loci"]
epochs = kwargs["epochs"]
BATCH_SIZE = kwargs["batch_size"]
output_dir = kwargs["output_dir"]

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# load all of the data (split train/test later?)
dataset = MtbGeneDataset(alignment_names, data_file, drug, longest_locus)

# MAE loss function
loss_func = nn.L1Loss()

# check that GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


def train_loop(dataloader, model, loss_func):
    '''
    Performs one epoch of training (one epoch = one iteration through all the batches in the dataset)
    '''
    
    num_batches = len(dataloader)
    epoch_loss = 0
    
    # first argument defines which parameters need to be optimized
    optimizer = optim.Adam(model.parameters(), lr=np.exp(-1.0 * 9))
    
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        
        # put the data on the device too
        X = X.to(device)
        y = y.to(device)
                
        # Compute the loss function based on the model prediction
        loss = loss_func(model(X), y)
        
        # move optimizer to reset the gradients of model parameters
        optimizer.zero_grad()
        
        # backpropagate the prediction loss
        loss.backward()
        
        # adjust the params by the gradients computed in the previous step
        optimizer.step()
        
        epoch_loss += loss.item()
                        
    # return the average loss across the batches, which is considered the overall epoch loss
    return epoch_loss / num_batches
        
            
        
def test_loop(dataloader, model, loss_func):
    
    num_batches = len(dataloader)
    test_loss = 0

    # add the loss for each point
    with torch.no_grad():
        
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            
            # put the data on the device too
            X = X.to(device)
            y = y.to(device)
            
            test_loss += loss_func(model(X), y).item()
            
    # return the average loss across the batches
    return test_loss / num_batches
    
    
cv_splits = 5
cv = KFold(n_splits=cv_splits, shuffle=True, random_state=1)

for fold, (train_idx, test_idx) in enumerate(cv.split(dataset)):
    
    print(f'Working on fold {fold+1} of {cv_splits} CV folds.....\n')
    
    # store losses for a single crossvalidation split
    train_losses = []
    test_losses = []
    
    # do the subsampling
    train_subsampler = SubsetRandomSampler(train_idx)
    test_subsampler = SubsetRandomSampler(test_idx)

    trainloader = DataLoader(
                      dataset, 
                      batch_size=BATCH_SIZE, sampler=train_subsampler)
    testloader = DataLoader(
                      dataset,
                      batch_size=BATCH_SIZE, sampler=test_subsampler)

    # initialize a new model for every split and transfer to GPU
    model = CNN(num_loci, filter_size)
    model = model.to(device)

    for epoch in range(1, epochs + 1):
        
        print(f"Epoch {epoch}/{epochs} \n")

        print('fitting..')
        train_losses.append(train_loop(trainloader, model, loss_func))
        
        print('\n predicting..')
        test_losses.append(test_loop(trainloader, model, loss_func))
            
    history = pd.DataFrame({"loss": train_losses, "val_loss": test_losses})
    history.to_csv(os.path.join(output_dir, f"history_split_{fold+1}.csv"))
    
    # save just the model weights
    torch.save(model.state_dict(), os.path.join(output_dir, f'model_weights_{fold+1}.pth'))
    
    # save the model
    torch.save(model, os.path.join(output_dir, f'model_weights_{fold+1}.pth'))