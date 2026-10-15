from pathlib import Path

from torch.utils.data import DataLoader

from src.dataset import ECGDataset
from src.preprocessing import train_transform, val_test_transform


PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "splits" / "val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"


print("Train CSV:", TRAIN_CSV)
print("Val CSV:", VAL_CSV)
print("Test CSV:", TEST_CSV)


train_dataset = ECGDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = ECGDataset(
    VAL_CSV,
    transform=val_test_transform
)

test_dataset = ECGDataset(
    TEST_CSV,
    transform=val_test_transform
)

print("\nDataset sizes")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


image, label = train_dataset[0]

print("\nSingle sample")
print("Image shape:", image.shape)
print("Label:", label)
print("Image dtype:", image.dtype)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)


images, labels = next(iter(train_loader))

print("\nBatch test")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)
