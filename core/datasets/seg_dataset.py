import torch
import torch.nn as nn
import torch.utils.data as D
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class SegDataset(D.Dataset):
    def __init__(self, path, all=False, train=True, transform=None, iter_len=None, in_memory=True) -> None:
        super().__init__()
        
        self.path = path
        self.files = []
        self.transform = transform
        self.iter_len = iter_len
        self.in_memory = in_memory
        
        if not all:
            file_path = f"{self.path}/train.txt" if train else f"{self.path}/test.txt"
        else:
            file_path = f"{self.path}/all.txt"

        with open(file_path, mode="r") as f:
            lines = f.readlines()
        for line in lines:
            self.files.append(line.strip())
            
        self.items = []
        if self.in_memory:
            for file_name in self.files:
                img_path = "{}/images/{}".format(self.path, file_name)
                lbl_path = "{}/labels/{}".format(self.path, file_name)
                img = np.array(Image.open(img_path))
                lbl = np.array(Image.open(lbl_path))
                self.items.append((file_name, img, lbl))

        if self.iter_len is None:
            self.len = len(self.files)
        else:
            self.len = self.iter_len
        
        self.label2train = [
                    [0, 255],
                    [1, 0],
                    [2, 1],
                    [3, 2],
                    [4, 3],
                    [5, 4],
                    [6, 5]]
        
        self.palette = [
                    [255,0,0],
                    [255,255,255],
                    [255,255,0],
                    [0,255,0],
                    [0,255,255],
                    [0,0,255]]
        
        self.label = [
                    "Clutter background",
                    "Imprevious surfaces",
                    "Car",
                    "Tree",
                    "Low vegetation",
                    "Building"]
        
        self.get_file_name = False
        
    def __getitem__(self, index):
        if self.iter_len is not None:
            img_index= random.randint(0, len(self.files)-1)
        else:
            assert index < len(self.files)
            img_index = index
        
        if self.in_memory:
            file_name, img, lbl = self.items[img_index]
        else:
            file_name = self.files[img_index]
            img_path = "{}/images/{}".format(self.path, file_name)
            lbl_path = "{}/labels/{}".format(self.path, file_name)
            img = np.array(Image.open(img_path))
            lbl = np.array(Image.open(lbl_path))
            
        for item in self.label2train:
            lbl[lbl == item[0]] = item[1]
        if self.transform is not None:
            trans = self.transform(image=img, mask=lbl)
            img = trans["image"]
            lbl = trans["mask"]

        if not self.get_file_name:
            return transforms.ToTensor()(img).float(), lbl.squeeze()
        return file_name, transforms.ToTensor()(img).float(), lbl.squeeze()
        
        
    def __len__(self):
        return self.len