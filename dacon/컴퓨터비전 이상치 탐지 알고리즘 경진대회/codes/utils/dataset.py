import torch
from torch.utils.data import Dataset
from utils.augmentation import train_transform, test_transform
from utils.randaugment import randAugment

class CustomDataset(Dataset):
    def __init__(self,img_paths,labels,mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode = mode

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode=='train':
            img = train_transform(image=img)

        if self.mode=='test':
            img = test_transform(image=img)

        label = self.labels[idx]
        return img, label

class CustomRandaugDataset(Dataset):
    def __init__(self,img_paths,labels,mode='train',N=1,M=2):
        self.img_paths = img_paths
        self.labels = labels
        self.mode = mode
        self.N = N
        self.M = M

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        train_randaug_transform = randAugment(N=self.N,M=self.M,p=0.5)
        if self.mode=='train':
            img = train_randaug_transform(image=img)
        if self.mode=='test':
            img = test_transform(image=img)

        label = self.labels[idx]
        return img, label
