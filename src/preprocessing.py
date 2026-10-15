from torchvision import transforms


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomRotation(
        degrees=3
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.02, 0.02)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])