import subprocess, sys, glob, os, yaml
from torch.types import Device
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

# check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_one_hot(sequence):
    """
	Creates a one-hot encoding of a sequence
	Parameters
	----------
	sequence: iterable of str
		Sequence containing only ACTG- characters

	Returns
	-------
	np.ndarray of int
		L (seq len) x 4 one-hot encoded sequence
	"""

    # Mapping to use for one-hot encoding
    BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
    
    seq_len = len(sequence)
    seq_in_index = [BASE_TO_COLUMN.get(b, b) for b in sequence]
    one_hot = np.zeros((seq_len, 5))

    # Assign the found positions to 1
    one_hot[np.arange(seq_len), np.array(seq_in_index)] = 1

    assert np.unique(one_hot.sum(axis=1))==1
    
    return one_hot


#### The Dataset is responsible for accessing and processing single instances of data.

#### The DataLoader pulls instances of data from the Dataset, collects them in batches, and returns them to be used by the training loop.

class MtbGeneDataset(Dataset):
    
    # read in sequence from AGC collection, perform one-hot encoding, and pad the ends
    
    def __init__(self, alignment_names, data_file, drug, longest_locus):
        '''
        Initialize the alignment files for each locus and the sample names
        '''
        
        # use grep to get a list of the headers from one of the alignment files, doesn't matter which. They all need to be in the same order
        proc = subprocess.Popen(["awk", ""'sub(/^>/, "")' "", alignment_names[0]], stdout=subprocess.PIPE)

        output = proc.stdout.read().decode(encoding='utf-8').split("\n").remove("")

        # if there's a newline character at the end of the fasta file (which there most likely will be), it will be added as an empty line. So remove it
        if "" in output:
            output.remove("")
          
        # this includes H37Rv
        self.sample_names = np.sort(output)
        
        # log-MIC values for the specified drug. Does not include H37Rv, so add it in the next row
        data = np.log(pd.read_csv(data_file).sort_values("Path")[drug+"_midpoint"].values)
        
        # add the minimum value as the MIC for H37Rv
        self.data = np.concatenate([data, np.array([np.min(data)])])
        
        # save all the other stuff to the class too
        self.longest_locus = longest_locus
        self.alignment_names = alignment_names

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Use this to get each sequence in the training dataset (based on indices, idx).
        '''
                
        sample_name = self.sample_names[idx]    
    
        # initialize empty tensor
        res = torch.empty((len(self.alignment_names), 5, self.longest_locus))
    
        # iterate through the alignment files, one for each locus    
        for i in range(len(self.alignment_names)):
            
            # use grep to get a single sequence from the alignment file
            # use Popen for more advanced cases than run can handle. Extract the sequence name
            proc = subprocess.Popen("grep -A 1 '" + sample_name + "' " + self.alignment_names[i], stdout=subprocess.PIPE, shell=True)
                
            # read from stdout and decode from the bytes object to a string
            output = proc.stdout.read().decode(encoding='utf-8')
            
            # remove newline characters, the sample name, and the fasta character
            seq = output.replace("\n", "").replace(sample_name, "").replace(">", "")
        
            # convert to one-hot encoding
            one_hot_torch = torch.from_numpy(get_one_hot(seq))
            
            # pad the ends so that everything is the same length (longest_locus)
            res[i] = torch.cat((one_hot_torch, torch.zeros((self.longest_locus-one_hot_torch.shape[0], 5)))).T
            
        # return the one-hot encoded matrix and the log-MIC value for the specified index
        return res.to(device), torch.from_numpy(np.array([self.data[idx]])).to(device)

    
    
class train_test_Dataset(Dataset):
    
    # read in sequence from AGC collection, perform one-hot encoding, and pad the ends
    
    def __init__(self, alignment_names, data_file, drug, longest_locus, train_or_test):
        '''
        Initialize the alignment files for each locus and the sample names
        '''
        
        # use grep to get a list of the headers from one of the alignment files, doesn't matter which. They all need to be in the same order
        proc = subprocess.Popen(["awk", ""'sub(/^>/, "")' "", alignment_names[0]], stdout=subprocess.PIPE)

        output = proc.stdout.read().decode(encoding='utf-8').split("\n")

        if "" in output:
            output.remove("")
          
        # this includes H37Rv
        output = np.sort(output)

        # sort to match order of outputs. Need to reset index so that the indices
        # are correct when getting only the training or testing set. 
        df = pd.read_csv(data_file).sort_values("Path").reset_index(drop=True)
        
        # get only training or testing data
        single_group_data = df.loc[df.category==train_or_test]
        self.data = np.log(single_group_data[drug+"_midpoint"].values)
        
        # get the sample names using the same index
        self.sample_names = output[single_group_data.index]        
        
        # save the other stuff to the class too
        self.longest_locus = longest_locus
        self.alignment_names = alignment_names

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Use this to get each sequence in the dataset (based on indices, idx).
        '''
                
        sample_name = self.sample_names[idx]    
    
        # initialize empty tensor
        res = torch.empty((len(self.alignment_names), 5, self.longest_locus))
    
        # iterate through the alignment files, one for each locus    
        for i in range(len(self.alignment_names)):
            
            # use grep to get a single sequence from the alignment file
            # use Popen for more advanced cases than run can handle. Extract the sequence name
            proc = subprocess.Popen("grep -A 1 '" + sample_name + "' " + self.alignment_names[i], stdout=subprocess.PIPE, shell=True)
                
            # read from stdout and decode from the bytes object to a string
            output = proc.stdout.read().decode(encoding='utf-8')
            
            # remove newline characters, the sample name, and the fasta character
            seq = output.replace("\n", "").replace(sample_name, "").replace(">", "")
        
            # convert to one-hot encoding
            one_hot_torch = torch.from_numpy(get_one_hot(seq))
            
            # pad the ends so that everything is the same length (longest_locus)
            res[i] = torch.cat((one_hot_torch, torch.zeros((self.longest_locus-one_hot_torch.shape[0], 5)))).T
            
        # return the one-hot encoded matrix and the log-MIC value for the specified index
        return res.to(device), torch.from_numpy(np.array([self.data[idx]])).to(device)


    
class CNN(nn.Module):
    
    def __init__(self, num_loci, filter_size):
        '''
        Notes:
        
        out_channels is the number of filters in a convolution
        '''
        
        super(CNN, self).__init__()
        
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=num_loci, out_channels=64, kernel_size=(4,filter_size)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,12)),
            nn.ReLU(),
            nn.MaxPool2d(1, 3),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,3)),
            nn.ReLU(),
            nn.MaxPool2d(1, 3),
            nn.Flatten(1, -1),
        )
        
        self.dense_relu_stack = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        '''
        Use torch.nn.functional in the forward method. Use nn.Module when initalizing a sequential model in init.
        '''
        conv_output = self.conv_relu_stack(x)
        
        # make the first dense layer. Need to do it here to get the correct number of input channels        
        # the second dimension of conv_output is the flattened array dimension. The first dimension is the batch size
        # this layer doesn't get put on the device like the rest of the model, so it has to be done manually here
        dense_1 = nn.Linear(conv_output.shape[-1], 256).to(device)
        
        # call the first dense layer on the flattened array
        output = dense_1(conv_output)
        
        return self.dense_relu_stack(output)