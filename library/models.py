import torch
import torch.nn as nn
import torchvision
import torchvision.models as mdls

class MLP(nn.Module):
    def __init__(self, device, input_size, output_size):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
        self.to(device)

    def forward(self, x):
        return self.seq(x)
    
class BatchNormConvNet(nn.Module):
    def __init__(self, device, output_size):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        self.magic_constant = 3200

        self.mlp = MLP(device, self.magic_constant, output_size)

        self.to(device)

    def forward(self, x):
        feature = self.feature_extractor(x)
        return self.mlp(feature)
    
class DropoutMLP(nn.Module):
    def __init__(self, device, input_size, output_size):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.45),
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
        self.to(device)

    def forward(self, x):
        return self.seq(x)
    
class DropoutBNConv(nn.Module):
    def __init__(self, device, output_size):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size = 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        self.magic_constant = 3200

        self.mlp = DropoutMLP(device, self.magic_constant, output_size)

        self.to(device)

    def forward(self, x):
        feature = self.feature_extractor(x)
        return self.mlp(feature)