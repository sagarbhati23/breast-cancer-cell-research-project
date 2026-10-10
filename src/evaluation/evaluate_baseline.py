
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data.breakhis_dataset import BreakHisDataset, create_transform
from src.models.resnet18 import create_model

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "metadata" / "splits"
RESULTS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
CHECKPOINT = ROOT / "models" / "resnet18_baseline.pth"

CLASSES = [
    "adenosis", "ductal_carcinoma", "fibroadenoma",
    "lobular_carcinoma", "mucinous_carcinoma",
    "papillary_carcinoma", "phyllodes_tumor", "tubular_adenoma",
]

torch.set_num_threads(4)
device = torch.device("cpu")

test_df = pd.read_csv(SPLITS / "fold_1_test.csv")
loader = DataLoader(
    BreakHisDataset(test_df, create_transform()),
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

model = create_model().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(device))
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(1).tolist())

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

report = classification_report(
    y_true, y_pred, labels=list(range(len(CLASSES))),
    target_names=CLASSES, output_dict=True, zero_division=0,
)
pd.DataFrame(report).transpose().to_csv(RESULTS / "baseline_results.csv")

print("Test accuracy:", round(accuracy_score(y_true, y_pred), 4))
print(pd.DataFrame(report).transpose().round(3).to_string())

matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(
    RESULTS / "confusion_matrix.csv"
)

print("Evaluation saved to results/metrics/")
