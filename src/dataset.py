import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

def get_datasets():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load twice with different transforms
    train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=eval_transform)
    test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=eval_transform)

    # Split using same indices so train/val don't overlap
    indices = torch.randperm(50000)
    train_data = Subset(train_dataset, indices[:45000])
    val_data = Subset(val_dataset, indices[45000:])

    return train_data, val_data, test_data