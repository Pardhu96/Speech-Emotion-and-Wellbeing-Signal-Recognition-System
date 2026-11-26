"""
Convolutional Neural Network (CNN) for Speech Emotion Recognition using Mel Spectrograms.
Implemented in PyTorch with 2D convolutions, batch normalization, and dropout.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, List, Tuple


class EmotionCNN(nn.Module):
    """
    2D Deep CNN for Mel Spectrogram emotion classification.
    Input shape: (Batch, 1, 128, Frames)
    """
    def __init__(self, num_classes: int = 8, dropout_rate: float = 0.25):
        super(EmotionCNN, self).__init__()

        # Conv Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1)
        )

        # Conv Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.15)
        )

        # Conv Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        # Conv Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(dropout_rate)
        )

        # Fully Connected Classifier
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


def train_cnn(
    model: EmotionCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Tuple[EmotionCNN, Dict[str, List[float]]]:
    """
    Train PyTorch CNN model with CrossEntropyLoss and AdamW optimizer.
    Returns: (best_model, history_dict)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_data in train_loader:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data in val_loader:
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step(epoch_val_acc)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = model.state_dict().copy()

    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"Loaded best checkpoint with Validation Accuracy: {best_val_acc:.4f}")

    return model, history


def save_cnn(model: EmotionCNN, filepath: str, emotion_classes: List[str]):
    """Save model checkpoint including architecture params and emotion classes."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "num_classes": len(emotion_classes),
        "emotion_classes": emotion_classes
    }
    torch.save(checkpoint, filepath)
    print(f"PyTorch CNN saved to {filepath}")


def load_cnn(filepath: str, device: str = "cpu") -> Tuple[EmotionCNN, List[str]]:
    """Load model checkpoint and instantiate EmotionCNN."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CNN model checkpoint not found: {filepath}")

    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filepath, map_location=device)
    emotion_classes = checkpoint["emotion_classes"]
    model = EmotionCNN(num_classes=len(emotion_classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, emotion_classes


def predict_single_spectrogram(
    model: EmotionCNN,
    spectrogram: np.ndarray,
    emotion_classes: List[str],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Generate CNN prediction and probabilities for a single Mel Spectrogram.
    spectrogram shape: (128, Frames)
    """
    model.eval()
    # Add batch and channel dimensions: (1, 1, 128, Frames)
    tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    prob_dict = {cls: float(prob) for cls, prob in zip(emotion_classes, probs)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    top_emotion = list(sorted_probs.keys())[0]
    top_confidence = sorted_probs[top_emotion]

    return {
        "predicted_emotion": top_emotion,
        "confidence": top_confidence,
        "probabilities": sorted_probs
    }
