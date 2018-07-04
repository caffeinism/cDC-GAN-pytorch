# coding: utf-8

import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data as data
import torch

class ImageFeatureFolder(dset.ImageFolder):
    def __init__(self, image_root, landmark_file, transform):
        super(ImageFeatureFolder, self).__init__(root=image_root, transform=transform)

        with open(landmark_file, 'r') as f:
            data = f.read()
        data = data.strip().split('\n')
        self.attrs = torch.FloatTensor([list(map(float, line.split()[1:])) for line in data[2:]])
        
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
                
        return img, self.attrs[index]