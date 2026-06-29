import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

from src.model import CifarCNN
from src.dataset import get_datasets
from src.train import train_one_epoch
from src.evaluate import evaluate
from src.utils import plot_training_curves


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CifarCNN().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = CrossEntropyLoss()

    train_data, val_data, test_data = get_datasets()
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128)
    test_loader = DataLoader(test_data, batch_size=128)

    train_losses = []
    val_losses = []
    val_accs = []

    epochs = 60
    best_val_acc = 0.0
    os.makedirs("outputs", exist_ok=True)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if val_acc> best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print(f"✓ Best model saved! Accuracy: {val_acc:.2f}%")
        scheduler.step()
    plot_training_curves(train_losses, val_losses, val_accs)

    # Optionally evaluate on test data after training
    # evaluate(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()
