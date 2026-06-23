import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from model import CifarCNN
from dataset import get_datasets
from train import train_one_epoch
from evaluate import evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CifarCNN().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()

    train_data, val_data, test_data = get_datasets()
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128)
    test_loader = DataLoader(test_data, batch_size=128)

    epochs = 2
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        evaluate(model, val_loader, criterion, device)

    # Optionally evaluate on test data after training
    # evaluate(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()
