import rasterio, cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from rasterio.enums import Resampling


class ImageDataset(Dataset):
    """ Baseline Dataset Class from Seam-Carving & Stain Color Normalization """
    def __init__(self, df, img_dir='./', transform=None, is_test=False):
        super().__init__()
        self.df = df
        self.img_dir = img_dir  # img_dir => image path from Seam-Carving & Stain Color Normalization
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 4]

        img_path = f'{self.img_dir}/{img_id}.png'
        image = rasterio.open(img_path)
        image = image.read(resampling=Resampling.bilinear).transpose(1,2,0)
        image = image.astype(np.float32)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        label_dict = {"CE": 0, "LAA": 1}
        label = label_dict[label]
        return image, label  # 둘 다 반환


class TestImageDataset(Dataset):
    def __init__(self, df, img_dir='./', transform=None, is_test=True):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if self.is_test:
            img_id = self.df.iloc[idx, 0]
            img_path = f'{self.img_dir}/{img_id}.png'

            image = rasterio.open(img_path)
            from rasterio.enums import Resampling
            image = image.read(resampling=Resampling.bilinear).transpose(1,2,0)
            patient_ids = self.df.iloc[idx, 2]

            if self.transform is not None:
                image = self.transform(image=image)['image']

            return image, patient_ids

