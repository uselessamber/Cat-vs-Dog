import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

label_data = {
    'Cat' : 0,
    'Dog' : 1
}

class CatDogDataset(Dataset):
    def __init__(self, path, augmentation = False):
        self.labels = []
        self.images = []

        self.transformation = transforms.Compose([
            transforms.Resize(size = 250),
            transforms.CenterCrop(size = 250),
            transforms.ToTensor()
        ])
        if augmentation:
            self.transformation = transforms.Compose([
                transforms.Resize(size = 250),
                transforms.RandomCrop(size = 250),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(20),
                transforms.ToTensor()
            ])

        failed_load = 0
        lost_color_channel = 0
        what_the_fuck = 0

        for lb_type in list(label_data.keys()):
            label_index = label_data[lb_type]
            image_path = f"{path}\{lb_type}"
            print(f"Load in data for label {lb_type}")
            for file in tqdm(os.listdir(image_path)):
                file_name = image_path + "\\" + file

                # Integrity check:
                try:
                    image_test = self.transformation(Image.open(file_name))
                    if image_test.shape != torch.Size([3, 250, 250]):
                        failed_load += 1
                        lost_color_channel += 1
                        continue
                except UnidentifiedImageError:
                    failed_load += 1
                    what_the_fuck += 1
                    continue

                self.images.append(file_name)
                self.labels.append(label_index)
                # self.labels.append(torch.Tensor([1 if label_index == i else 0 for i in range(2)]))
        
        print(f"Number of failed load: {failed_load}")
        print(f"    - Due to missing color channel somehow: {lost_color_channel}")
        print(f"    - What the fuck: {what_the_fuck}")
    

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transformation(Image.open(self.images[idx]))
        # print(f"{image.shape} - {self.images[idx]}")
        label = self.labels[idx]
        return image, label
