from pathlib import Path
import hashlib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

files = [
    PROJECT_ROOT / "data" / "splits" / "train.csv",
    PROJECT_ROOT / "data" / "splits" / "val.csv",
    PROJECT_ROOT / "data" / "splits" / "test.csv",

]


def md5_hash(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

all_df = pd.concat(
    dfs,
    ignore_index=True
)

print("Total files:", len(all_df))

print("\nCalculating hashes...")

all_df["hash"] = all_df["image_path"].apply(md5_hash)

print(
    "Total unique hashes:",
    all_df["hash"].nunique()
)


# ----------------------------------
# Check whether identical images
# have different class labels
# ----------------------------------

label_counts = (
    all_df
    .groupby("hash")["class_name"]
    .nunique()
)

conflicting_hashes = label_counts[
    label_counts > 1
].index


print(
    "\nHashes with conflicting labels:",
    len(conflicting_hashes)
)


if len(conflicting_hashes) > 0:

    conflicts = all_df[
        all_df["hash"].isin(conflicting_hashes)
    ].sort_values("hash")

    print("\nConflicting examples:")

    print(
        conflicts[
            [
                "hash",
                "class_name",
                "image_path"
            ]
        ].to_string(index=False)
    )


# ----------------------------------
# Distribution of unique images 
# ----------------------------------

unique_df = all_df.drop_duplicates(
    subset="hash"
)

print("\nUnique image class distribution:")

print(
    unique_df["class_name"]
    .value_counts()
)
