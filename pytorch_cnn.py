import subprocess, sys, glob, os, yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm

# import dataloader and model
from pytorch_utils import MtbGeneDataset, train_test_Dataset, CNN

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

# check that GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(f"Running on {device}\n")

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
                        
    # return the average loss across the batches
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
    

# initialize training and testing datasets
train_dataset = train_test_Dataset(alignment_names, data_file, drug, longest_locus, "original_train_set")
test_dataset = train_test_Dataset(alignment_names, data_file, drug, longest_locus, "original_test_set")

print(f"{len(train_dataset)} training points")
print(f"{len(test_dataset)} testing points")

trainloader = DataLoader(
                  train_dataset, 
                  batch_size=BATCH_SIZE)

testloader = DataLoader(
                  test_dataset,
                  batch_size=BATCH_SIZE)

# initialize the model and transfer to device
model = CNN(num_loci, filter_size)
model = model.to(device)

# MAE loss function
loss_func = nn.L1Loss()

# store losses
train_losses = []
test_losses = []

for epoch in range(epochs):
    
    print(f"\n Epoch {epoch+1}/{epochs} \n")

    print('fitting..')
    train_losses.append(train_loop(trainloader, model, loss_func))
    
    print('\n predicting..')
    test_losses.append(test_loop(trainloader, model, loss_func))

    print(f"Train loss: {train_losses[-1]}, Test loss: {test_losses[-1]}")

history = pd.DataFrame({"loss": train_losses, "val_loss": test_losses})
history.to_csv(os.path.join(output_dir, f"history.csv"))
    
# save the model weights only
torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))

# save the model
torch.save(model, os.path.join(output_dir, 'model.pth'))