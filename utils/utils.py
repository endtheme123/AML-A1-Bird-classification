from data.CUBDataset import CUBTrainDataset,CUBTestDataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_path, transform, batch_size, shuffle):
    
    train_dataset = CUBTrainDataset(root=data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataset = CUBTestDataset(root=data_path, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader