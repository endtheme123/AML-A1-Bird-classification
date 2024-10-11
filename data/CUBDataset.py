import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
from PIL import Image
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

DATA_ROOT = "/home/endtheme/git/bird-class/data/"

class CUBTrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        # self.data = datasets.ImageFolder(root=os.path.join(root, 'train' if train else 'test'), transform=transform)
        txt_path = os.path.join(root, "train.txt")
        # self.labels = []
        # img_paths = []
        self.root = root
        self.data = pd.read_csv(txt_path, sep=" ", names= ["path", "label"])
        # with open(txt_path, 'r') as file:
        #     for line in file:
        # # Remove any trailing characters like newlines or spaces
        #         line = line.strip().split(" ")
        #         print("img: ", line[0])
        #         print("label: ", int(line[1])) 
        #         self.labels.append(int(line[1]))
        #         img_paths.append(line[0])

        # self.img_paths = [os.path.join(root, "Train", img) for img in img_paths]
        self.transform = transform

        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


class CUBTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        # self.data = datasets.ImageFolder(root=os.path.join(root, 'train' if train else 'test'), transform=transform)
        txt_path = os.path.join(root, "test.txt")
        self.labels = []
        img_paths = []
        with open(txt_path, 'r') as file:
            for line in file:
        # Remove any trailing characters like newlines or spaces
                line = line.strip().split(" ")
                print("img: ", line[0])
                print("label: ", int(line[1])) 
                self.labels.append(int(line[1]))
                img_paths.append(line[0])

        self.img_paths = [os.path.join(root, "Test", img) for img in img_paths]
        self.transform = transform

        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Create Datasets
# train_dataset = CUBDataset(root='path_to_CUB_200_2011', transform=data_transforms['train'])
# val_dataset = CUBDataset(root='path_to_CUB_200_2011', transform=data_transforms['val'])

# # DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
