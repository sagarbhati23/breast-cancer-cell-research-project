
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "metadata" / "dataset.csv"
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "BreakHis_v1"

SUBTYPES = [
    "adenosis",
    "ductal_carcinoma",
    "fibroadenoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
    "phyllodes_tumor",
    "tubular_adenoma",
]

LABEL_TO_INDEX = {name: index for index, name in enumerate(SUBTYPES)}


def resolve_image_path(image_path):
    relative_path = image_path.replace("\\", "/")
    prefix = "BreaKHis_v1/"

    if relative_path.startswith(prefix):
        relative_path = relative_path[len(prefix):]

    return IMAGE_ROOT / relative_path


class BreakHisDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = resolve_image_path(row["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        label = LABEL_TO_INDEX[row["subtype"]]

        return image, label


def load_fold_data(fold=1):
    if fold not in range(1, 6):
        raise ValueError("fold must be between 1 and 5")

    dataframe = pd.read_csv(METADATA_PATH)

    group_column = f"fold_{fold}_group"
    train_df = dataframe[dataframe[group_column] == "train"].copy()
    test_df = dataframe[dataframe[group_column] == "test"].copy()

    return train_df, test_df


def create_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_dataloader(dataframe, batch_size=8, shuffle=False):
    dataset = BreakHisDataset(
        dataframe=dataframe,
        transform=create_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
