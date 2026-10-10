from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "metadata" / "dataset.csv")
fold = "fold_1_group"

official_train = df[df[fold] == "train"].copy()
test = df[df[fold] == "test"].copy()

# Each patient has exactly one subtype; split patients within each subtype.
patients = official_train[["patient_id", "subtype"]].drop_duplicates()
train_patients, val_patients = [], []

for subtype, group in patients.groupby("subtype"):
    if len(group) < 2:
        raise ValueError(f"Not enough patients for subtype: {subtype}")
    tr, va = train_test_split(
        group["patient_id"].tolist(),
        test_size=0.2,
        random_state=42,
    )
    train_patients.extend(tr)
    val_patients.extend(va)

train = official_train[official_train["patient_id"].isin(train_patients)]
val = official_train[official_train["patient_id"].isin(val_patients)]

assert set(train["patient_id"]).isdisjoint(val["patient_id"])
assert set(train["patient_id"]).isdisjoint(test["patient_id"])
assert set(val["patient_id"]).isdisjoint(test["patient_id"])
assert set(train["subtype"]) == set(val["subtype"]) == set(test["subtype"])

out = ROOT / "metadata" / "splits"
out.mkdir(parents=True, exist_ok=True)
train.to_csv(out / "fold_1_train.csv", index=False)
val.to_csv(out / "fold_1_val.csv", index=False)
test.to_csv(out / "fold_1_test.csv", index=False)

for name, part in [("Train", train), ("Validation", val), ("Official test", test)]:
    print(f"{name}: {len(part)} images, {part['patient_id'].nunique()} patients")
    print(part["subtype"].value_counts().sort_index().to_string())
    print()
print("PASS: all 8 subtypes present; no patient overlap.")
