from pathlib import Path
import hashlib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv" 


def md5_hash(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def add_hashes(df, split_name):
    print(f"Hashing {split_name}...")

    
    df = df.copy()
    df["hash"] = df["image_path"].apply(md5_hash)

    return df


train_df = add_hashes(
    pd.read_csv(TRAIN_CSV),
    "train"
)

val_df = add_hashes(
    pd.read_csv(VAL_CSV),
    "validation"
)

test_df = add_hashes(
    pd.read_csv(TEST_CSV),
    "test"
)


train_hashes = set(train_df["hash"])
val_hashes = set(val_df["hash"])
test_hashes = set(test_df["hash"])


print("\n=== Cross-split exact duplicates ===")

print(
    "Train-Val:",
    len(train_hashes & val_hashes)
)

print(
    "Train-Test:",
    len(train_hashes & test_hashes)
)

print(
    "Val-Test:",
    len(val_hashes & test_hashes)
)


print("\n=== Internal exact duplicates ===")

print(
    "Train:",
    train_df["hash"].duplicated().sum()
)

print(
    "Validation:",
    val_df["hash"].duplicated().sum()
)

print(
    "Test:",
    test_df["hash"].duplicated().sum()
)
