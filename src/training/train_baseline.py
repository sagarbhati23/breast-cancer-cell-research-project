
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.data.breakhis_dataset import BreakHisDataset, create_transform
from src.models.resnet18 import create_model

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "metadata" / "splits"
RESULTS = ROOT / "results" / "metrics"
CHECKPOINTS = ROOT / "models"

BATCH_SIZE = 4
EPOCHS = 5
TRAIN_SAMPLES = None
VAL_SAMPLES = None
LEARNING_RATE = 0.0001

torch.manual_seed(42)
torch.set_num_threads(4)
device = torch.device("cpu")

train_df = pd.read_csv(SPLITS / "fold_1_train.csv")
val_df = pd.read_csv(SPLITS / "fold_1_val.csv")

if TRAIN_SAMPLES is not None:
    train_df = train_df.sample(
        n=min(TRAIN_SAMPLES, len(train_df)), random_state=42
    ).reset_index(drop=True)

if VAL_SAMPLES is not None:
    val_df = val_df.sample(
        n=min(VAL_SAMPLES, len(val_df)), random_state=42
    ).reset_index(drop=True)

print(f"Training images: {len(train_df)}", flush=True)
print(f"Validation images: {len(val_df)}", flush=True)
print(f"Device: {device}", flush=True)

transform = create_transform()

train_loader = DataLoader(
    BreakHisDataset(train_df, transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    BreakHisDataset(val_df, transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = create_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

RESULTS.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
history = []

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    train_loss, train_true, train_pred = 0.0, [], []

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_true.extend(labels.tolist())
        train_pred.extend(outputs.argmax(1).tolist())

        print(
            f"\rEpoch {epoch + 1}: training batch "
            f"{batch_idx}/{len(train_loader)}",
            end="",
            flush=True,
        )

    print("\nValidating...", flush=True)
    model.eval()
    val_loss, val_true, val_pred = 0.0, [], []

    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels.to(device)
            outputs = model(images.to(device))
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            val_true.extend(labels.tolist())
            val_pred.extend(outputs.argmax(1).tolist())

    row = {
        "epoch": epoch + 1,
        "train_loss": train_loss / len(train_df),
        "train_accuracy": accuracy_score(train_true, train_pred),
        "val_loss": val_loss / len(val_df),
        "val_accuracy": accuracy_score(val_true, val_pred),
        "seconds": round(time.time() - start, 1),
    }
    history.append(row)
    print(row, flush=True)

pd.DataFrame(history).to_csv(
    RESULTS / "baseline_pilot.csv", index=False
)
torch.save(model.state_dict(), CHECKPOINTS / "resnet18_baseline.pth")

print("Baseline training complete. Metrics and checkpoint saved.", flush=True)
