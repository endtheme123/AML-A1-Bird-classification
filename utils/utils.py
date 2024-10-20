from data.CUBDataset import CUBTrainDataset,CUBTestDataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_path, train_transform, test_transform, batch_size, shuffle, compact):
    
    train_dataset = CUBTrainDataset(root=data_path, transform=train_transform, compact = compact)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataset = CUBTestDataset(root=data_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    return train_loader, test_loader



