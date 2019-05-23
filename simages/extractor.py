import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.utils.data as utils
from torch.utils.data.dataset import Dataset

from .models import BasicAutoencoder, Autoencoder

class PILDataset(Dataset):
    """PIL dataset."""

    def __init__(self, pil_list, transform=None):
        """
        Args:
            pil_list (list of PIL images)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pil_list = pil_list
        self.transform = transform

    def __len__(self):
        return len(self.pil_list)

    def __getitem__(self, idx):
        sample = self.pil_list[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

class EmbeddingExtractor:
    def __init__(self, data_dir=None, array=None, num_channels=1, num_epochs=2, batch_size=32, show_train=True):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.show_train = show_train

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_channels = num_channels

        data_transforms = transforms.Compose([transforms.Resize(50),
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
            self.dataloader = self.tensor_dataloader(array, data_transforms)

        if not torch.cuda.is_available():
            print("Note: No GPU found, using CPU. Performance is improved on a CUDA-device.")
            self.model = BasicAutoencoder()
        else:
            self.model = Autoencoder()

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.distance = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)

        self.train()

    def tensor_dataloader(self, array, transforms):
        print(f"INFO: data shape: {array.shape} (Target: N x C x H x W)")
        if array.ndim == 3:
            print(f"Converting to grayscale dataset of dims {array.shape[0]} x 1 x {array.shape[1]} x {array.shape[2]}")
            array = array[:, np.newaxis, ...]
            print(f"New shape: {array.shape}")
        tensor = torch.Tensor(array)
        pil_list = [TF.to_pil_image(array) for array in tensor]
        dataset = PILDataset(pil_list, transform=transforms)
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
            for data in self.dataloader:
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
