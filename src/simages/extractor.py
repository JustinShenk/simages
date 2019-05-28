import logging
import os
from typing import Union
import warnings

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

from .dataset import PILDataset, SingleFolderDataset
from .models import BasicAutoencoder, Autoencoder

warnings.filterwarnings("ignore", message="Palette images with Transparency")
log = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extract embeddings from data with models and allow visualization.

    Attributes:
        trainloader (torch loader)
        evalloader (torch loader
        model (torch.nn.Module)
        embeddings (np.ndarray)

    """
    def __init__(
        self,
        input:Union[str,np.ndarray],
        num_channels=None,
        num_epochs=2,
        batch_size=32,
        show_train=True,
        show=False,
        z_dim=8,
        **kwargs,
    ):
        """Inits EmbeddingExtractor with input, either `str` or `np.nd.array`, performs training and validation.

        Args:
            input (np.ndarray or str): data
            num_channels (int): grayscale = 1, color = 3
            num_epochs (int): more is better (generally)
            batch_size (int): number of images per batch
            show_train (bool): show intermediate training results
            show (bool): show closest pairs
            z_dim (int): compression size
            kwargs (dict)

        """
        self.num_epochs = num_epochs
        self._batch_size = batch_size
        self._show_train = show_train
        self._show = show

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._num_channels = num_channels

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
                log.info(f"Skipping {os.path.basename(path)}: {e}")
                return False
            return True

        if isinstance(input, str):
            data_dir = os.path.abspath(input)
            hasdir = any([os.path.isdir(path) for path in os.listdir(data_dir)])
            if not hasdir:
                self.image_dataset = SingleFolderDataset(
                    input, transform=data_transforms, is_valid_file=is_valid
                )
            else:
                self.image_dataset = datasets.ImageFolder(
                    root=data_dir, transform=data_transforms, is_valid_file=is_valid
                )
            self.trainloader = torch.utils.data.DataLoader(
                self.image_dataset, batch_size=batch_size, shuffle=True, num_workers=4
            )
            self.evalloader = torch.utils.data.DataLoader(
                self.image_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
        elif isinstance(input, np.ndarray):
            self.trainloader = self.tensor_dataloader(
                input, data_transforms, shuffle=True
            )
            self.evalloader = self.tensor_dataloader(
                input, data_transforms, shuffle=False
            )

        if not torch.cuda.is_available():
            log.info(
                "Note: No GPU found, using CPU. Performance is improved on a CUDA-device."
            )

        self.model = BasicAutoencoder(num_channels=num_channels, z_dim=z_dim)

        if torch.cuda.device_count() > 1:
            log.info("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(self.model)

        self.model.to(self._device)
        self._distance = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)

        self.train()
        self.eval()

    def get_image(self, index: int):
        result = self.evalloader.dataset[index]
        if isinstance(result, tuple):
            return result[0].cpu()
        else:
            return result.cpu()

    def tensor_dataloader(self, array, transforms, shuffle=True):
        log.debug(f"INFO: data shape: {array.shape} (Target: N x C x H x W)")
        if array.ndim == 3:
            log.debug(
                f"Converting to grayscale dataset of dims {array.shape[0]} x 1 x {array.shape[1]} x {array.shape[2]}"
            )
            array = array[:, np.newaxis, ...]
            log.debug(f"New shape: {array.shape}")

        tensor = torch.Tensor(array)
        pil_list = [TF.to_pil_image(array.squeeze()) for array in tensor]
        dataset = PILDataset(pil_list, transform=transforms)
        dataloader = utils.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=shuffle
        )  # create your dataloader
        return dataloader

    def pil_loader(self, path):
        log.info("loading {}".format(path))
        channels = "RGB" if self._num_channels >= 3 else "L"
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert(channels)
        except:
            log.error("fail to load {} using PIL".format(img))

    def train(self):
        for epoch in range(self.num_epochs):
            for data in self.trainloader:
                if isinstance(data, list):
                    data = data[0]
                img = data.to(self._device)
                # ===================forward=====================
                output, embedding = self.model(img)

                loss = self._distance(output, img)
                # ===================backward====================
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            if self._show_train:
                try:
                    img_array = img.cpu()[0]
                    output_array = output.detach().cpu()[0]

                    grid_img = torchvision.utils.make_grid([img_array, output_array])
                    self.show(grid_img, title=f"Epoch {epoch}", block=False)
                except Exception as e:
                    log.error(f"{e}")

            # ===================log========================
            log.info("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, self.num_epochs, loss))

    def eval(self):
        embeddings = []
        imgs = []
        self.model.eval()

        for data in self.evalloader:
            if isinstance(data, list):
                data = data[0]
            img = data.to(self._device)
            imgs.append(img)
            # ===================forward=====================
            output, embedding = self.model(img)
            embeddings.append(embedding)

            loss = self._distance(output, img)
            # ===================backward====================
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        if self._show_train:
            try:
                img_array = img.cpu()[0]
                output_array = output.detach().cpu()[0]

                grid_img = torchvision.utils.make_grid([img_array, output_array])
                self.show(grid_img, title=f"Reconstruction")
            except Exception as e:
                log.error(f"{e}")

        # ===================log========================
        log.info("eval, loss:{:.4f}".format(loss))

        self.embeddings = torch.cat(embeddings).detach().cpu().numpy()

    def duplicates(self, n: int = 10):
        pairs, distances = closely.solve(self.embeddings, n=n)
        return pairs, distances

    def show(self, img, title="", block=True):
        if isinstance(img, torch.Tensor):
            npimg = img.detach().numpy()
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise NotImplementedError(f"{type(img)}")

        plt.subplots()
        plt.title(f"{title}")
        plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze(), interpolation="nearest")
        plt.show(block=block)

    def show_images(self, indices: Union[list, int], title=""):
        if isinstance(indices, int):
            indices = [indices]
        tensors = [self.get_image(idx) for idx in indices]
        self.show(torchvision.utils.make_grid(tensors), title=title)

    def show_duplicates(self, n=5):
        pairs, distances = self.duplicates(n=n)

        # Plot pairs
        for idx, pair in enumerate(pairs):
            img0 = self.get_image(pair[0])
            img1 = self.get_image(pair[1])
            img0_reconst = self.decode(index=pair[0])[0]
            img1_reconst = self.decode(index=pair[1])[0]
            self.show(
                torchvision.utils.make_grid(
                    [img0, img1, img0_reconst, img1_reconst], nrow=2
                ),
                title=f"{pair}, dist={distances[idx]:.2f}",
            )

        return pairs, distances

    def decode(self, embedding=None, index=None, show=False):
        self.model.eval()

        if embedding is None:
            embedding = self.embeddings[index]

        emb = np.expand_dims(embedding, 0)  # add batch axis

        # Check if has direct access to `decode` method
        if not hasattr(self.model, "decode"):
            output, _ = self.model.module.decode(torch.Tensor(emb).to(self._device))
        else:
            output, _ = self.model.decode(torch.Tensor(emb).to(self._device))

        if show:
            grid_img = torchvision.utils.make_grid(output)
            self.show(grid_img, title=index)

        return output
