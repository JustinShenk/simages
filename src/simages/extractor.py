"""
.. module:: extractor
   :synopsis: Embedding extractor module

.. moduleauthor:: Justin Shenk <shenkjustin@gmail.com>


"""

import logging
import os
from typing import Union, Optional, Tuple
import warnings

import closely
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.utils.data as utils

from .dataset import PILDataset, ImageFolder
from .models import BasicAutoencoder, UnNormalize

warnings.filterwarnings("ignore", message="Palette images with Transparency")
log = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extract embeddings from data with models and allow visualization.

    Attributes:
        trainloader (torch loader)
        evalloader (torch loader)
        model (torch.nn.Module)
        embeddings (np.ndarray)

    """

    def __init__(
        self,
        input: Union[str, np.ndarray],
        num_channels: int = 3,
        num_epochs: int = 2,
        batch_size: int = 32,
        show: bool = False,
        plot_embeddings=False,
        show_path: bool = False,
        show_train: bool = False,
        z_dim: int = 8,
        metric: str = "cosine",
        model: Optional[torch.nn.Module] = None,
        db: Optional = None,
        embeddings_path: Optional[str] = False,
        save_embeddings: Optional[bool] = False
    ):
        """Inits EmbeddingExtractor with input, either `str` or `np.ndarray`, performs training and validation.

        Args:
            input (np.ndarray or str): data
            num_channels (int): grayscale = 1, color = 3
            num_epochs (int): more is better (generally)
            batch_size (int): number of images per batch
            show (bool): show closest pairs
            show_path (bool): show path of duplicates
            show_train (bool): show intermediate training results
            z_dim (int): compression size
            metric (str): distance metric for :meth:`scipy.spatial.distance.cdist` (eg, euclidean, cosine, hamming, etc.)
            model (torch.nn.Module, optional): class implementing same methods as :class:`~simages.BasicAutoencoder`
            db_conn_string (str): Mongodb connection string
            embeddings_path (str): path to load embeddings
            save_embeddings (str): saves embeddings in current directory

        """
        self.num_epochs = num_epochs
        self._batch_size = batch_size
        self._show = show
        self._show_path = show_path
        self._show_train = show_train
        self._db = db

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._num_channels = num_channels

        self._metric = metric
        self._z_dim = z_dim
        self._hw = 48

        self._mean = [0.5] * self._num_channels
        self._std = [0.25] * self._num_channels

        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(self._hw),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self._mean, self._std),
            ]
        )
        basic_transforms = transforms.Compose(
            [
                transforms.Resize(self._hw),
                transforms.CenterCrop(self._hw),
                transforms.ToTensor(),
                transforms.Normalize(self._mean, self._std),
            ]
        )

        def is_valid(path):
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
            self.root = data_dir
            self.train_dataset = ImageFolder(
                data_dir, transform=train_transforms, is_valid_file=is_valid
            )
            self.eval_dataset = ImageFolder(
                data_dir, transform=basic_transforms, is_valid_file=is_valid
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

        if model is not None:
            self.model = model
        else:
            self.model = BasicAutoencoder(num_channels=num_channels, z_dim=z_dim)

        if torch.cuda.device_count() > 1:
            log.info("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(self.model)

        self.model.to(self._device)
        self._distance = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)

        if embeddings_path:
            log.info("skipping training, loading embeddings from file")
            self.embeddings = np.load(embeddings_path)
            log.info(f"Embeddings shape {self.embeddings.shape}")
        else:
            embeddings_path = f'embeddings_epochs-{num_epochs}.npy'
            self.train()
            self.eval()
        if save_embeddings:
            log.info(f"saving embeddings to {embeddings_path}")
            np.save(embeddings_path, self.embeddings)



    def _truncate_middle(self, string: str, n: int) -> str:
        if len(string) <= n:
            # string is already short-enough
            return string
        # half of the size, minus the 3 .'s
        n_2 = int(int(n) / 2 - 3)
        # whatever's left
        n_1 = int(n - n_2 - 3)
        return f"{string[:n_1]}...{string[-n_2:]}"

    def get_image(self, index: int) -> torch.Tensor:
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
        )
        return dataloader

    def train(self):
        """Train autoencoder to build embeddings of dataset. Final embeddings are created in
        :meth:`~simages.extractor.EmbeddingExtractor.eval`.

        """
        log.info(
            f"Building embeddings for {len(self.evalloader.dataset)} images. This may take some time..."
        )

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

                    grid_img = torchvision.utils.make_grid(
                        [img_array, output_array], nrow=1
                    )
                    self.show(
                        grid_img,
                        title=f"Building embeddings: epoch [{epoch+1}/{self.num_epochs}]",
                        block=False,
                        y_labels=[(2, "Original"), (5, "Reconstruction")],
                    )
                except Exception as e:
                    log.error(f"{e}")

            # ===================log========================
            log.info(
                "epoch [{}/{}], loss:{:.4f}".format(epoch + 1, self.num_epochs, loss)
            )

    def eval(self):
        """Evaluate reconstruction of embeddings built in `train`."""
        embeddings = []
        imgs = []

        # Change model to `eval` mode so weights are frozen
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

                grid_img = torchvision.utils.make_grid(
                    [img_array, output_array], nrow=1
                )
                self.show(
                    grid_img,
                    title=f"Reconstruction",
                    y_labels=[(2, "Original"), (5, "Reconstruction")],
                )
            except Exception as e:
                log.error(f"{e}")

        # ===================log========================
        log.info("eval, loss:{:.4f}".format(loss))

        self.embeddings = torch.cat(embeddings).detach().cpu().numpy()

    def duplicates(
        self, n: int = 10, quantile: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Identify `n` closest pairs of images, or quantile (for example, closest 0.05).

        Args:
            n (int): number of pairs
            quantile (float): quantile of total combination (suggested range: 0.001 - 0.01)
        """
        nr_embeddings = len(self.embeddings)
        min_pairs = (nr_embeddings * (nr_embeddings - 1)) // 2

        n = min(n, min_pairs)

        if quantile is not None:
            pairs, distances = closely.solve(
                self.embeddings, quantile=quantile, metric=self._metric
            )
        else:
            pairs, distances = closely.solve(self.embeddings, n=n, metric=self._metric)

        return pairs, distances

    @staticmethod
    def channels_last(img: np.ndarray) -> np.ndarray:
        """Move channels from first to last by swapping axes."""
        img_t = np.transpose(img, (1, 2, 0))
        return img_t

    def show(
        self,
        img: Union[torch.Tensor, np.ndarray],
        title: str = "",
        block: bool = True,
        y_labels=None,
        unnormalize=True,
    ):
        """Plot `img` with `title`.

        Args:
            img (torch.Tensor or np.ndarray): Image to plot
            title (str): plot title
            block (bool): block matplotlib plot until window closed
        """
        if unnormalize:
            img = self.unnormalize(img)
        if isinstance(img, torch.Tensor):
            npimg = img.detach().numpy()
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise NotImplementedError(f"{type(img)}")

        if img.shape[0] in [1, 2, 3]:
            npimg = self.channels_last(npimg).squeeze()

        fig, ax = plt.subplots(1, 1)
        plt.title(f"{title}")
        ax.imshow(npimg, interpolation="nearest")
        if y_labels is not None:
            labels = [item.get_text() for item in ax.get_xticklabels()]
            for idx, label in y_labels:
                labels[idx] = label
            ax.set_yticklabels(labels)

        plt.show(block=block)

    def show_images(self, indices: Union[list, int], title=""):
        """Plot images (from validation data) at `indices` with `title`"""
        if isinstance(indices, int):
            indices = [indices]
        tensors = [self.get_image(idx) for idx in indices]
        self.show(torchvision.utils.make_grid(tensors), title=title)


    def color_embeddings(self, image_paths, path_colors='train-blue_val-green'):
        """Color embeddings by path, eg, 'train' or 'val'"""
        colors = []
        path_color_mapping = {}
        for path_color in path_colors.split('_'):
            path, color = path_color.split('-')
            path_color_mapping[path] = color

        # Iterate over each image path and assign a color based on the presence of 'train' or 'val'
        for path in image_paths:
            color = 'gray'
            for directory, directory_color in path_color_mapping.items():
                if directory in path:
                    color = directory_color
                    break

            colors.append(color)
        return colors

    def plot_embeddings(self, title="", path_colors=None):
        """Plot embeddings (from validation data), hover to see image using bokeh"""
        from bokeh.models import ColumnDataSource, HoverTool
        from bokeh.plotting import figure, show
        embeddings = self.embeddings
        validation_image_paths = self.evalloader.dataset.samples

        colors = ['gray'] * len(validation_image_paths)
        if path_colors:
            colors = self.color_embeddings(validation_image_paths, path_colors=path_colors)

        # Create scatter plot of embeddings
        source = ColumnDataSource(
        data=dict(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            image_path=validation_image_paths,
            image_path_relative=[os.path.relpath(p, self.root) for p in validation_image_paths],
            colors=colors
            )
        )
        hover = HoverTool(
            tooltips="""
            <div>
                <div>
                    <img src="@image_path" height="128" alt="@image_path" width="128" style="float: left; margin: 0px 15px 15px 0px; image-rendering: pixelated;"></img>
                    @image_path_relative
                </div>
            </div>
            """
        )
        p = figure(
            plot_width=800,
            plot_height=800,
            title=title,
            tools=[hover, "pan", "wheel_zoom", "box_zoom", "reset", "save"],
        )
        p.circle("x", "y", size=5, source=source, fill_alpha=0.8, line_color='colors', fill_color='colors')
        show(p)


    def get_embedding(self, index: int) -> torch.Tensor:
        """Get embedding at `index` of eval/embedding"""
        return torch.Tensor(self.embeddings[index])

    def image_paths(self, indices, short=True):
        """Get path to image at `index` of eval/embedding

        Args:
            indices Union[int,list]: indices of embeddings in dataset
            short (bool): truncate filepath to 30 charachters

        Returns:
            paths (str or list of str): paths to images in image folder

        """
        if isinstance(indices, (int, np.int_, np.int64)):  # <-- add np.int64 here
            indices = [indices]

        paths = []
        for index in indices:
            path = self.evalloader.dataset.samples[index]
            if short:
                path = self._truncate_middle(os.path.basename(path), 30)
            paths.append(path)

        if len(paths) == 1:
            return paths[0]  # backward compatibility
        return paths

    @torch.no_grad()
    def show_duplicates(self, n=5, path=False) -> (np.ndarray, np.ndarray):
        """Show duplicates from comparison of embeddings. Uses `closely` package to get pairs.

        Args:
            n (int): how many closest pairs to identify
            path (bool): Plot pairs of images with abbreviated paths

        Returns:
            pairs (np.ndarray): pairs as indices
            distances (np.ndarray): distances of pairs

        """
        show_path = path or self._show_path
        pairs, distances = self.duplicates(n=n)

        # Plot pairs
        for idx, pair in enumerate(pairs):
            img0 = self.get_image(pair[0])
            img1 = self.get_image(pair[1])
            img0_reconst = self.decode(index=pair[0], astensor=True)[0]
            img1_reconst = self.decode(index=pair[1], astensor=True)[0]
            pair_details = (
                f"{self.image_paths(pair[0])}\n{self.image_paths(pair[1])}"
                if show_path
                else pair
            )
            title = f"{pair_details}, dist={distances[idx]:.2f}"
            self.show(
                torchvision.utils.make_grid(
                    [img0, img1, img0_reconst, img1_reconst], nrow=2
                ),
                title=title,
                y_labels=[(2, "Original"), (5, "Reconstruction")],
            )

        return pairs, distances

    def unnormalize(self, image: torch.Tensor) -> torch.Tensor:
        """Unnormalize an image.

        Args:
            image (:class:`torch.Tensor`)

        Returns:
            image (:class:`torch.Tensor`)

        """
        unorm = UnNormalize(mean=self._mean, std=self._std)
        return unorm(image)

    def decode(
        self,
        embedding: Optional[np.ndarray] = None,
        index: Optional[int] = None,
        show: bool = False,
        astensor: bool = False,
    ) -> np.ndarray:
        """Decode embeddings at `index` or pass `embedding` directly

        Args:
            embedding (np.ndarray, optional): embedding of image
            index (int): index (of validation set / embeddings) to decode
            show (bool): plot the results
            astensor (bool): keep as torch.Tensor

        Returns:
            image (np.ndarray or torch.Tensor): reconstructed image from embedding

        """
        self.model.eval()

        if embedding is None:
            embedding = self.embeddings[index]

        emb = np.expand_dims(embedding, 0)  # add batch axis

        # Check if has direct access to `decode` method
        if not hasattr(self.model, "decode"):
            image, _ = self.model.module.decode(torch.Tensor(emb).to(self._device))
        else:
            image, _ = self.model.decode(torch.Tensor(emb).to(self._device))

        image = self.unnormalize(image)

        if show:
            grid_img = torchvision.utils.make_grid(image)
            self.show(grid_img, title=index)

        if astensor:
            return image.detach().cpu()
        return image.detach().cpu().numpy()
