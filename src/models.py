import torch.nn as nn


def conv_block(in_ch, out_ch):
    """Conv -> BatchNorm -> ReLU -> MaxPool: ქსელის განმეორებადი ბლოკი."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """ნულიდან დაწერილი baseline: 4 conv ბლოკი + კლასიფიკატორი."""

    def __init__(self, num_classes=37, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),     # 224 -> 112
            conv_block(32, 64),    # 112 -> 56
            conv_block(64, 128),   # 56  -> 28
            conv_block(128, 256),  # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # 14x14 -> 1x1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


from torchvision import models


def build_resnet18(num_classes=37, freeze_backbone=True):
    """ImageNet-ზე გაწვრთნილი ResNet18, ბოლო ფენა ჩვენი 37 ჯიშისთვის."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class MediumCNN(nn.Module):
    """SimpleCNN-ზე უფრო დიდი: 5 conv ბლოკი (512 არხამდე) + 2-ფენიანი კლასიფიკატორი."""

    def __init__(self, num_classes=37, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),     # 224 -> 112
            conv_block(32, 64),    # 112 -> 56
            conv_block(64, 128),   # 56  -> 28
            conv_block(128, 256),  # 28  -> 14
            conv_block(256, 512),  # 14  -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class MediumCNN(nn.Module):
    """SimpleCNN-ზე უფრო დიდი: 5 conv ბლოკი (512 არხამდე) + 2-ფენიანი კლასიფიკატორი."""

    def __init__(self, num_classes=37, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),     # 224 -> 112
            conv_block(32, 64),    # 112 -> 56
            conv_block(64, 128),   # 56  -> 28
            conv_block(128, 256),  # 28  -> 14
            conv_block(256, 512),  # 14  -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ResBlock(nn.Module):
    """რეზიდუალური ბლოკი: out = ReLU( F(x) + shortcut(x) )."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # თუ ზომა ან არხები იცვლება, shortcut-საც უნდა მივცეთ იგივე ფორმა
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNetCNN(nn.Module):
    """ნულიდან დაწერილი ResNet-ის მსგავსი ქსელი: stem + 3 რეზიდუალური ბლოკი."""

    def __init__(self, num_classes=37, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  # 224 -> 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            ResBlock(32, 64, stride=2),    # 112 -> 56
            ResBlock(64, 128, stride=2),   # 56  -> 28
            ResBlock(128, 256, stride=2),  # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x)
        return self.classifier(x)
