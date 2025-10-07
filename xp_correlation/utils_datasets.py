import torch
import torchvision
import torchvision.datasets as datasets

from torch.utils.data import DataLoader, Dataset
from PIL import Image


def get_dataset(str_dataset, path_data, batch_w2=1000):
    """
        Parameters
        ----------
        str_dataset: str (MNIST or CIFAR10)
        path_data: str
        batch_w2: int, batch size to compute 1D wasserstein distances (for s-OTDD)

        Returns
        ------
        dataset: torchvision.datasets
        transform: torchvision.transforms.Compose
        kwargs: dictionary with metadata ("dimension", "num_channels")
    """
    if str_dataset == "MNIST":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        dataset = datasets.MNIST(root=path_data, train=True, download=True)

        kwargs_sotdd = {
            "dimension": 784,
            "num_channels": 1,
            "num_moments": 5,
            "use_conv": False,
            "precision": "float",
            "p": 2,
            "chunk": batch_w2
        }

    elif str_dataset == "FMNIST":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        dataset = datasets.FashionMNIST(root=path_data, train=True, download=True)

        kwargs_sotdd = {
            "dimension": 784,
            "num_channels": 1,
            "num_moments": 5,
            "use_conv": False,
            "precision": "float",
            "p": 2,
            "chunk": batch_w2
        }

    elif str_dataset == "CIFAR10":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
        ])

        dataset = datasets.CIFAR10(root=path_data, train=True, download=True)
        dataset.targets = torch.tensor(dataset.targets)
        dataset.data = torch.tensor(dataset.data)

        kwargs_sotdd = {
            "dimension": 32,
            "num_channels": 3,
            "num_moments": 5,
            "use_conv": True,
            "precision": "float",
            "p": 2,
            "chunk": batch_w2
        }

    else:
        print(str_dataset + " not implemented")

    return dataset, transform, kwargs_sotdd


class Subset(Dataset):
    def __init__(self, dataset, original_indices, transform):
        self._dataset = dataset
        self._original_indices = original_indices
        self.transform = transform
        self.indices = torch.arange(start=0, end=len(self._original_indices), step=1)
        self.data = self._dataset.data[self._original_indices]
        self.targets = self._dataset.targets[self._original_indices].clone()
        self.classes = sorted(torch.unique(self._dataset.targets).tolist())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx].numpy())
        return self.transform(img), self.targets[idx]


def get_loaders(dataset, idx1, idx2, transform, batch_size=128):
    """
        Parameters
        ----------
        dataset: Dataset object
        idx1: list of indices
        idx2: list of indices
        transform: Transform object

        Return
        ------
        dataloaders: list of Dataloaders
        distr_imgs: list of size 2, with each element being the elements of all the data
        distr_labels: list of labels of each dataset
    """
    sub1 = Subset(dataset=dataset, original_indices=idx1, transform=transform)
    sub2 = Subset(dataset=dataset, original_indices=idx2, transform=transform)

    dataloader1 = DataLoader(sub1, batch_size=batch_size, shuffle=True)
    dataloader2 = DataLoader(sub2, batch_size=batch_size, shuffle=True)

    dataloaders = [dataloader1, dataloader2]

    all_imgs1, all_imgs2 = [], []
    all_labels1, all_labels2 = [], []

    for x, y in sub1:   # no DataLoader needed
        all_imgs1.append(x)
        all_labels1.append(y)

    for x, y in sub2:   # no DataLoader needed
        all_imgs2.append(x)
        all_labels2.append(y)

    all_imgs1 = torch.stack(all_imgs1)   # shape: [N, C, H, W]
    all_labels1 = torch.tensor(all_labels1)

    all_imgs2 = torch.stack(all_imgs2)   # shape: [N, C, H, W]
    all_labels2 = torch.tensor(all_labels2)

    distr_imgs = [all_imgs1, all_imgs2]
    distr_labels = [all_labels1, all_labels2]

    return dataloaders, distr_imgs, distr_labels
