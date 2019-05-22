import torch
import torch.nn as nn
import torch.functional as F

class Autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(Autoencoder, self).__init__()

        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 46x46
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 44x44
        self.maxpool1 = nn.MaxPool2d(2)  # 22x22
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # 20x20
        self.maxpool2 = nn.MaxPool2d(2)  # 10x10
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(64 * 16 * 16, z_dim)

        # decoder
        self.fc2 = nn.Linear(z_dim, 64 * 22 * 22)
        self.convT1 = nn.ConvTranspose2d(16, 16, kernel_size=3)
        self.convT2 = nn.ConvTranspose2d(16, 1, kernel_size=3)

    def forward(self, x):
        print("A", x.shape)
        x = self.conv1(x)
        print("B", x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        print("C", x.shape)
        x = F.relu(x)
        x = self.maxpool1(x)
        print("D", x.shape)
        x = self.conv3(x)
        x = F.relu(x)
        print("E", x.shape)
        x = self.maxpool2(x)
        print("F", x.shape)
        x = self.conv4(x)
        print("G", x.shape)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        print("H", x.shape)
        x = self.fc1(x)
        print("I", x.shape)
        return self.decode(x)

    def decode(self, x):
        embedding = x
        x = self.fc2(x)
        print("J", x.shape)
        x = x.view(x.size(0), 16, 44, 44)
        x = self.convT1(x)
        print("K", x.shape)
        x = F.relu(x)
        x = self.convT2(x)
        print("L", x.shape)
        x = torch.sigmoid(x)
        return x, embedding
