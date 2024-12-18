import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import mlflow



def initialize_model(weights, dropout, class_names):
    """Initialize the ResNet-18 model with custom classification head."""
    model = resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False  # Freeze feature extractor layers
    for param in model.layer4.parameters():
        param.requires_grad = True  # Unfreeze Layer 4

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, len(class_names))
    )

    for param in model.fc.parameters():
        param.requires_grad = True

    return model



def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training and validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.savefig("learning_curves.png")
    mlflow.log_artifact("learning_curves.png", artifact_path="plots")
    plt.show()
