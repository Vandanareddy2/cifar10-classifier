import torch

def evaluate(model, val_loader, criterion, device):
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset) * 100
    print(f"Val Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")