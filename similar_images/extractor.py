import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as utils


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


class EmbeddingExtractor:
    def __init__(self, data_dir=None, array=None, num_channels=1, num_epochs=2, batch_size=32, show_train=True):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Normalize((0.5), (0.25))])

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.show_train = show_train

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_channels = num_channels

        data_transforms = transforms.Compose([transforms.Resize(48),
                                              transforms.CenterCrop(48),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5] * num_channels, [0.25] * num_channels)])

        img_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.gif', '.octet-stream']
        if isinstance(data_dir, str):
            self.image_datasets = datasets.DatasetFolder(data_dir, self.pil_loader, img_extensions,
                                                         data_transforms)
            self.dataloader = torch.utils.data.DataLoader(self.image_datasets, batch_size=batch_size, shuffle=False,
                                                          num_workers=4)
        elif array is not None:
            assert isinstance(array, np.ndarray)
            self.dataloader = self.tensor_dataloader(array)

        self.model = Autoencoder()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.distance = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)

        self.train()

    def tensor_dataloader(self, array):
        print(f"INFO: data shape: {array.shape} (Target: N x C x H x W)")
        if array.ndim == 3:
            print(f"Converting to grayscale dataset of dims {array.shape[0]} x 1 x {array.shape[1]} x {array.shape[2]}")
            array = array[:, np.newaxis, ...]
            print(f"New shape: {array.shape}")
        tensors = torch.stack([torch.Tensor(arr) for arr in array])  # transform to torch tensors
        dataset = utils.TensorDataset(tensors)  # create your datset
        dataloader = utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  # create your dataloader
        return dataloader

    def show(self, img, epoch):
        npimg = img.numpy()
        plt.subplots()
        plt.title(f"Epoch: {epoch}")
        plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze(), interpolation='nearest')
        plt.show()

    def pil_loader(self, path):
        print("loading {}".format(path))
        channels = "RGB" if self.num_channels >= 3 else "L"
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert(channels)
        except:
            print('fail to load {} using PIL'.format(img))

    def train(self):
        for epoch in range(self.num_epochs):
            embeddings = []
            for data, in self.dataloader:
                img = data.to(self.device)
                # ===================forward=====================
                output, embedding = self.model(img)
                embeddings.append(embedding.cpu())
                loss = self.distance(output, img)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.show_train:
                try:
                    img_array = img.cpu()[0]
                    output_array = output.detach().cpu()[0]

                    grid_img = torchvision.utils.make_grid([img_array, output_array])
                    self.show(grid_img, title=f"Epoch {epoch}")
                except Exception as e:
                    print(f"{e}")

            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.num_epochs, loss))

        self.embeddings = torch.cat(embeddings).detach().cpu().numpy()

    def show(self, img, title=""):
        if isinstance(img, torch.Tensor):
            npimg = img.numpy()
        else:
            raise NotImplementedError(f"{type(img)}")
        plt.subplots()
        plt.title(title)
        plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze(), interpolation='nearest')
        plt.show()

    def decode(self, embedding=None, index=None):
        if embedding is None:
            embedding = self.embeddings[index]

        emb = np.expand_dims(embedding, 0)  # add batch axis
        output, _ = self.model.module.decode(torch.Tensor(emb).to(self.device))

        grid_img = torchvision.utils.make_grid(output)
        self.show(grid_img, title=index)