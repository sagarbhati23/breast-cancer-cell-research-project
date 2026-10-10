
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "metadata" / "dataset.csv"
IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "BreakHis_v1"

FOLD = 1


def resolve_image_path(image_path):
    """Convert the CSV path to the local image path."""
    relative_path = image_path.replace("\\", "/")

    prefix = "BreaKHis_v1/"
    if relative_path.startswith(prefix):
        relative_path = relative_path[len(prefix):]

    return IMAGE_ROOT / relative_path


def main():
    df = pd.read_csv(METADATA_PATH)

    required_columns = [
        "image_path",
        "patient_id",
        "diagnosis",
        "subtype",
        "magnification",
        *[f"fold_{i}_group" for i in range(1, 6)],
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if df["image_path"].duplicated().any():
        raise ValueError("Duplicate image paths found.")

    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values found in required metadata.")

    paths = df["image_path"].map(resolve_image_path)
    missing_files = [str(path) for path in paths if not path.is_file()]

    train = df[df[f"fold_{FOLD}_group"] == "train"]
    test = df[df[f"fold_{FOLD}_group"] == "test"]

    train_patients = set(train["patient_id"])
    test_patients = set(test["patient_id"])
    overlapping_patients = train_patients & test_patients

    print(f"Total unique images: {len(df)}")
    print(f"Missing image files: {len(missing_files)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Unique subtypes: {df['subtype'].nunique()}")
    print(f"Magnifications: {sorted(df['magnification'].unique())}")
    print(f"\nFold {FOLD} train images: {len(train)}")
    print(f"Fold {FOLD} test images: {len(test)}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")
    print(f"Overlapping patients: {len(overlapping_patients)}")
    print("\nTraining subtype counts:")
    print(train["subtype"].value_counts().sort_index())
    print("\nTesting subtype counts:")
    print(test["subtype"].value_counts().sort_index())

    if missing_files or overlapping_patients:
        raise RuntimeError("Dataset validation failed.")

    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()
