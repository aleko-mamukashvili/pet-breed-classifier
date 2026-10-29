import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ImageNet-ის სტატისტიკა: pretrained მოდელები ამ ნორმალიზაციით ვარჯიშობდნენ
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool, augment: bool = True, size: int = 224):
    """train=True და augment=True: შემთხვევითი ცვლილებები.
    სხვა შემთხვევაში: დეტერმინისტული resize + normalize."""
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_dataloaders(root="data", batch_size=32, val_frac=0.2,
                    augment=True, size=224, seed=42, num_workers=2):
    # ერთი და იგივე სურათები, ოღონდ სხვადასხვა transform-ით
    train_ds = datasets.OxfordIIITPet(
        root, split="trainval", target_types="category",
        transform=get_transforms(True, augment, size), download=True)
    val_ds = datasets.OxfordIIITPet(
        root, split="trainval", target_types="category",
        transform=get_transforms(False, size=size), download=True)
    test_ds = datasets.OxfordIIITPet(
        root, split="test", target_types="category",
        transform=get_transforms(False, size=size), download=True)

    # trainval-ის გაყოფა train/val-ად ფიქსირებული seed-ით
    n = len(train_ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = Subset(train_ds, train_idx)
    val_set = Subset(val_ds, val_idx)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **kw)
    val_loader = DataLoader(val_set, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, train_ds.classes
