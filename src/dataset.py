import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_datasets():
    training_data = datasets.CIFAR10(
    root="data",         # Directory where data is stored
    train=True,          # Target the training split
    download=True,       # Download from the internet if missing
    transform=ToTensor() # Convert images to PyTorch Tensors
    )

    train_data, val_data = random_split(training_data, [45000, 5000])

    test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
    )

    return train_data, val_data, test_data
