import logging
import os
from typing import Union, Optional
import warnings

import closely
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.utils.data as utils

from .dataset import PILDataset, SingleFolderDataset
from .models import BasicAutoencoder

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
        input: Union[str, np.ndarray],
        num_channels=3,
        num_epochs=2,
        batch_size=32,
        show=False,
        show_path=False,
        show_train=False,
        z_dim=8,
        **kwargs,
    ):
        """Inits EmbeddingExtractor with input, either `str` or `np.nd.array`, performs training and validation.

        Args:
            input (np.ndarray or str): data
            num_channels (int): grayscale = 1, color = 3
            num_epochs (int): more is better (generally)
            batch_size (int): number of images per batch
            show (bool): show closest pairs
            show_path (bool): show path of duplicates
            show_train (bool): show intermediate training results
            z_dim (int): compression size
            kwargs (dict)

        """
        self.num_epochs = num_epochs
        self._batch_size = batch_size
        self._show = show
        self._show_path = show_path
        self._show_train = show_train

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._num_channels = num_channels

        train_transforms = transforms.Compose(
            [
                transforms.Resize(50),
                transforms.CenterCrop(48),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        basic_transforms = transforms.Compose(
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
                self.train_dataset = SingleFolderDataset(
                    input, transform=train_transforms, is_valid_file=is_valid
                )
                self.eval_dataset = SingleFolderDataset(
                    input, transform=basic_transforms, is_valid_file=is_valid
                )
            else:
                self.train_dataset = datasets.ImageFolder(
                    root=data_dir, transform=train_transforms, is_valid_file=is_valid
                )
                self.eval_dataset = datasets.ImageFolder(
                    root=data_dir, transform=basic_transforms, is_valid_file=is_valid
                )
            self.trainloader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
            )
            self.evalloader = torch.utils.data.DataLoader(
                self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
        elif isinstance(input, np.ndarray):
            self.trainloader = self._tensor_dataloader(
                input, train_transforms, shuffle=True
            )
            self.evalloader = self._tensor_dataloader(
                input, basic_transforms, shuffle=False
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

    def _truncate_middle(self, string: str, n: int):
        if len(string) <= n:
            # string is already short-enough
            return string
        # half of the size, minus the 3 .'s
        n_2 = int(int(n) / 2 - 3)
        # whatever's left
        n_1 = int(n - n_2 - 3)
        return f"{string[:n_1]}...{string[-n_2:]}"

    def get_image(self, index: int):
        result = self.evalloader.dataset[index]
        if isinstance(result, tuple):
            return result[0].cpu()
        else:
            return result.cpu()

    def _tensor_dataloader(
        self,
        array: np.ndarray,
        transforms: torchvision.transforms.Compose,
        shuffle: bool = True,
    ) -> utils.DataLoader:
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
                    self.show(
                        grid_img,
                        title=f"Building embeddings: epoch [{epoch+1}/{self.num_epochs}]",
                        block=False,
                    )
                except Exception as e:
                    log.error(f"{e}")

            # ===================log========================
            log.info(
                "epoch [{}/{}], loss:{:.4f}".format(epoch + 1, self.num_epochs, loss)
            )

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

    def duplicates(self, n: int = 10) -> (np.ndarray, np.ndarray):
        pairs, distances = closely.solve(self.embeddings, n=n)
        return pairs, distances

    def show(
        self, img: Union[torch.Tensor, np.ndarray], title: str = "", block: bool = True
    ):
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

    def image_path(self, index, short=True):
        """Get path to image at `index` of eval/embedding"""
        path = self.evalloader.dataset.samples[index]
        if short:
            return self._truncate_middle(os.path.basename(path), 30)
        return path

    def show_duplicates(self, n=5, path=False):
        show_path = path or self._show_path
        pairs, distances = self.duplicates(n=n)

        # Plot pairs
        for idx, pair in enumerate(pairs):
            img0 = self.get_image(pair[0])
            img1 = self.get_image(pair[1])
            img0_reconst = self.decode(index=pair[0])[0]
            img1_reconst = self.decode(index=pair[1])[0]
            pair_details = (
                f"{self.image_path(pair[0])}\n{self.image_path(pair[1])}"
                if show_path
                else pair
            )
            title = f"{pair_details}, dist={distances[idx]:.2f}"
            self.show(
                torchvision.utils.make_grid(
                    [img0, img1, img0_reconst, img1_reconst], nrow=2
                ),
                title=title,
            )

        return pairs, distances

    def decode(
        self,
        embedding: Optional[np.ndarray] = None,
        index: Optional[int] = None,
        show: bool = False,
    ) -> np.ndarray:
        """Decode `embedding`"""
        self.model.eval()

        if embedding is None:
            embedding = self.embeddings[index]

        emb = np.expand_dims(embedding, 0)  # add batch axis

        # Check if has direct access to `decode` method
        if not hasattr(self.model, "decode"):
            image, _ = self.model.module.decode(torch.Tensor(emb).to(self._device))
        else:
            image, _ = self.model.decode(torch.Tensor(emb).to(self._device))

        if show:
            grid_img = torchvision.utils.make_grid(image)
            self.show(grid_img, title=index)

        return image.detach().cpu().numpy()
