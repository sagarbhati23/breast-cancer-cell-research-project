import torch.nn as nn
from torchvision import models

NUM_CLASSES = 8


def create_model(num_classes=NUM_CLASSES):
    """Create ResNet-18 with random initialization for subtype classification."""
    model = models.resnet18(weights=None)

    # Replace the original 1000-class output layer with our 8-class layer.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
