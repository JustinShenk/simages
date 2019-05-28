from torch.utils.data.dataset import Dataset


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