import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, num_channels=1, z_dim=32):
        super(Autoencoder, self).__init__()

        self.input_channels = num_channels
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3)  # 46x46
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 44x44
        self.maxpool1 = nn.MaxPool2d(2)  # 22x22
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # 20x20
        self.maxpool2 = nn.MaxPool2d(2)  # 10x10
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(64 * 16 * 16, z_dim)

        # decoder
        self.fc2 = nn.Linear(z_dim, 64 * 22 * 22)
        self.convT1 = nn.ConvTranspose2d(16, 16, kernel_size=3)
        self.convT2 = nn.ConvTranspose2d(16, num_channels, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.decode(x)

    def decode(self, x):
        embedding = x
        x = self.fc2(x)
        x = x.view(x.size(0), 16, 44, 44)
        x = self.convT1(x)
        x = F.relu(x)
        x = self.convT2(x)
        x = torch.sigmoid(x)
        return x, embedding


class BasicAutoencoder(nn.Module):
    def __init__(self, num_channels: int = 1, z_dim: int = 8, hw=48):
        """Basic autoencoder - default for simages.

        Args:
           num_channels (int): grayscale = 1, color = 3
           z_dim (int): number of embedding units to compress image to
           hw (int): height and width for input/output image

        """
        super(BasicAutoencoder, self).__init__()

        self.input_channels = num_channels
        self.z_dim = z_dim
        self.hw = hw
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3)  # 46x46
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 44x44
        self.maxpool1 = nn.MaxPool2d(2)  # 22x22
        self.fc1 = nn.Linear(64 * 22 * 22, z_dim)

        # decoder
        self.fc2 = nn.Linear(z_dim, 256)
        self.fc3 = nn.Linear(256, num_channels * hw * hw)
        # self.convT1 = nn.ConvTranspose2d(16, num_channels, kernel_size=5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.decode(x)

    def decode(self, x):
        embedding = x
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = x.view(x.size(0), self.input_channels, self.hw, self.hw)
        x = torch.sigmoid(x)
        return x, embedding
