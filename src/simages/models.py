import torchvision

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


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def pretrained_resnet(num_channels, hw, zdim, feature_extract=True):
    model_ft = torchvision.models.resnet34(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    # feature_model = list(model_ft.fc.children())
    # feature_model.append(nn.Linear(num_ftrs, hw * hw * num_channels))
    model_ft.fc = nn.Linear(num_ftrs, zdim)
    return model_ft


class PretrainedModel(nn.Module):
    def __init__(self, hw, num_channels, zdim=8):
        super(PretrainedModel, self).__init__()
        self._hw = hw
        self._num_channels = num_channels
        self._zdim = zdim
        self.model = pretrained_resnet(num_channels=num_channels, hw=hw, zdim=zdim)
        self.fc_out = nn.Linear(zdim, hw * hw * num_channels)

    def forward(self, input):
        embedding = self.model.forward(input)
        x, embedding = self.decode(embedding)
        return x, embedding

    def decode(self, embedding):
        x = self.fc_out(embedding)
        x = x.view(x.size(0), self._num_channels, self._hw, self._hw)
        return x, embedding


class Unflatten(nn.Module):
    def __init__(self, hw, num_channels):
        super(Unflatten, self).__init__()
        self._hw = hw
        self._num_channels = num_channels

    def forward(self, input, embedding):
        x = input.view(input.size(0), self._num_channels, self._hw, self._hw)
        return x, embedding


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
