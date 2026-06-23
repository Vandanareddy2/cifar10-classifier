from torch.utils.data import DataLoader

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    # your loop goes here
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(f"Loss: {epoch_loss:.4f}")


