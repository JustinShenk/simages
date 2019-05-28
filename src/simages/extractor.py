import os
from typing import Union

import closely
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

from .dataset import PILDataset
from .models import BasicAutoencoder, Autoencoder


class EmbeddingExtractor:
    def __init__(
            self,
            data_dir=None,
            array=None,
            num_channels=3,
            num_epochs=2,
            batch_size=32,
            show_train=True,
            **kwargs
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.show_train = show_train

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_channels = num_channels

        data_transforms = transforms.Compose(
            [transforms.Resize(50), transforms.CenterCrop(48), transforms.ToTensor()]
        )

        img_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".gif",
            ".octet-stream",
        ]

        def is_valid(path):
            _, file_extension = os.path.splitext(path)
            valid_ext = file_extension.lower() in img_extensions
            if not valid_ext:
                return False
            try:
                Image.open(path).verify()
            except Exception as e:
                print(f"Skipping {path}: {e}")
                return False
            return True

        if isinstance(data_dir, str):
            self.image_dataset = datasets.ImageFolder(
                root=data_dir, transform=data_transforms,
                is_valid_file=is_valid
            )
            self.dataloader = torch.utils.data.DataLoader(
                self.image_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
        elif array is not None:
            assert isinstance(array, np.ndarray)
            self.dataloader = self.tensor_dataloader(array, data_transforms)

        if not torch.cuda.is_available():
            print(
                "Note: No GPU found, using CPU. Performance is improved on a CUDA-device."
            )
            self.model = BasicAutoencoder(num_channels=num_channels)
        else:
            self.model = Autoencoder(num_channels=num_channels)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.distance = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)

        self.train()
        self.eval()

    def tensor_dataloader(self, array, transforms):
        print(f"INFO: data shape: {array.shape} (Target: N x C x H x W)")
        if array.ndim == 3:
            print(
                f"Converting to grayscale dataset of dims {array.shape[0]} x 1 x {array.shape[1]} x {array.shape[2]}"
            )
            array = array[:, np.newaxis, ...]
            print(f"New shape: {array.shape}")

        tensor = torch.Tensor(array)
        pil_list = [TF.to_pil_image(array.squeeze()) for array in tensor]
        dataset = PILDataset(pil_list, transform=transforms)
        dataloader = utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )  # create your dataloader
        return dataloader

    def show(self, img, epoch):
        npimg = img.numpy()
        plt.subplots()
        plt.title(f"Epoch: {epoch}")
        plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze(), interpolation="nearest")
        plt.show()

    def pil_loader(self, path):
        print("loading {}".format(path))
        channels = "RGB" if self.num_channels >= 3 else "L"
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert(channels)
        except:
            print("fail to load {} using PIL".format(img))

    def train(self):
        for epoch in range(self.num_epochs):
            for data in self.dataloader:
                if isinstance(data, tuple):
                    data = data[0]
                img = data.to(self.device)
                # ===================forward=====================
                output, embedding = self.model(img)

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
            print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, self.num_epochs, loss))

    def eval(self):
        embeddings = []
        imgs = []
        self.model.eval()
        for data in self.dataloader:
            if isinstance(data, tuple):
                data = data[0]
            img = data.to(self.device)
            imgs.append(img)
            # ===================forward=====================
            output, embedding = self.model(img)
            embeddings.append(embedding)

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
        print("eval, loss:{:.4f}".format(loss))

        self.embeddings = torch.cat(embeddings).detach().cpu().numpy()
        self.evalimgs = imgs

    def show(self, img, title=""):
        if isinstance(img, torch.Tensor):
            npimg = img.numpy()
        else:
            raise NotImplementedError(f"{type(img)}")

        plt.subplots()
        plt.title(title)
        plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze(), interpolation="nearest")
        plt.show()

    def duplicates(self, n:int=10):
        pairs, distances = closely.solve(self.embeddings,n=n)
        return pairs, distances

    def show(self, img, title=""):
        if isinstance(img, torch.Tensor):
            npimg = img.numpy()
        else:
            raise NotImplementedError(f"{type(img)}")

        plt.subplots()
        plt.title(f"{title}")
        plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze(), interpolation="nearest")
        plt.show(block=False)

    def show_images(self, indices: Union[list, int], title=""):
        if isinstance(indices, int):
            indices = [indices]
        tensors = [self.extractor.dataloader.dataset[idx].cpu() for idx in indices]
        self.show(torchvision.utils.make_grid(tensors), title=title)

    def show_duplicates(self, n=5):
        pairs, distances = self.duplicates(n=n)

        # Plot pairs
        for idx, pair in enumerate(pairs):
            img0 = self.dataloader.dataset[pair[0]].cpu()
            img1 = self.dataloader.dataset[pair[1]].cpu()
            self.show(
                torchvision.utils.make_grid([img0, img1]),
                title=f"{pair}, dist={distances[idx]:.2f}",
            )

    def decode(self, embedding=None, index=None):
        if embedding is None:
            embedding = self.embeddings[index]

        emb = np.expand_dims(embedding, 0)  # add batch axis

        self.model.eval()

        # Check if has direct access to `decode` method
        if not hasattr(self.model, "decode"):
            output, _ = self.model.module.decode(torch.Tensor(emb).to(self.device))
        else:
            output, _ = self.model.decode(torch.Tensor(emb).to(self.device))

        grid_img = torchvision.utils.make_grid(output)
        self.show(grid_img, title=index)
