import csv
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import numpy as np

class mlpdataset(Dataset):
    def __init__(self, csv_file, modelname,args, istrained = True):

        self.data = pd.read_csv(csv_file)

        labels = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
        
        self.labels = labels

        features = torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32)
        
        
        self.features = features
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        features = self.features[idx]
        return features, label

def binarize_and_expand(input_tensor,args):

    if args.dataset == "UNSWNB15":
        if args.onlypacket:
            bit_lengths = [4, 16, 8, 8]
        else:
            bit_lengths = [4, 16, 8, 8, 16, 16, 16, 16, 16]
    else:
        if args.onlypacket:
            bit_lengths = [4, 16, 8, 8, 8, 4]
        else:
            bit_lengths = [4, 16, 8, 8, 8, 4, 16, 16, 16, 16, 16]

    expanded_features = []

    for i, bit_len in enumerate(bit_lengths):
        # Extract the feature column and convert to integer type
        feature = input_tensor[:, i].unsqueeze(1).int()  # Convert to int to apply bitwise operations
        
        # Create a mask to extract each bit using bitwise operations
        binary_feature = ((feature.unsqueeze(-1) & (1 << torch.arange(bit_len)))) > 0

        # Convert binary feature to float and map 0 to -1.0, and 1 to 1.0
        bits_tensor = binary_feature.float() * 2 - 1.0
        
        # Reshape bits_tensor from [N, 1, X] to [N, X]
        bits_tensor = bits_tensor.squeeze(1)
        
        # Add to the list of expanded features
        expanded_features.append(bits_tensor)
    
    # Concatenate all expanded binary features along the last dimension
    output_tensor = torch.cat(expanded_features, dim=1)

    return output_tensor


def new_mlp_dataloader(csv_file, model_name, args, batch_size=32, istrained = True, shuffle=True):
    dataset = mlpdataset(csv_file, model_name,args,istrained = istrained)
    val_ratio = 0.1
    known_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    if istrained:
        labels = [dataset[i][1] for i in range(len(dataset))]  
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=val_ratio,
            stratify=np.array(labels).squeeze()  
        )
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader, val_loader, known_loader
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader





