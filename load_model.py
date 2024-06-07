from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import tqdm.notebook as tq
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from PIL import Image
import torch.nn.functional as nnf

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from library.CatDogDataset import *
from library.models import *
from library.train import *

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

transformation = transforms.Compose([
    transforms.Resize(size = 250),
    transforms.CenterCrop(size = 250),
    transforms.ToTensor()
])

CLASS_NAME = ["Cat", "Dog"]
torch.manual_seed(69)

model = BatchNormConvNet(device, 2)
model.load_state_dict(torch.load("BatchNormConvNet.txt"))
model.eval()


def play_around_test():
    os.system("cls")
    file_name = input("Input the name of the image file (has to be in jpg format): ")

    loaded_image = Image.open(f"./data/toy/{file_name}.jpg")
    target = transformation(loaded_image)
    target = torch.unsqueeze(target, 0)
    output = model(target)
    prediction = CLASS_NAME[torch.argmax(output)]

    prob = nnf.softmax(output, dim = 1)
    top_p, top_class = prob.topk(2, dim = 1)
    output_text = ""
    for class_name, probability in zip(top_class[0], top_p[0]):
        output_text = output_text + f"    {CLASS_NAME[class_name]} : {(probability * 100):.2f}%\n"
    
    plt.imshow(loaded_image)
    plt.title(f"Model Prediction: {prediction}")
    plt.text(x = 0, y = 10, s = output_text)
    plt.axis("off")

    plt.show()

    input_check = input("Do you want to continue? (y/n): ").lower()
    if input_check == 'n':
        return False
    return True

while play_around_test():
    pass