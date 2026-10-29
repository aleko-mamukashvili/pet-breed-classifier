import torch
import torch.nn as nn
from sklearn.metrics import f1_score

def train_one_epoch(model, loader, criterion, optimizer, device):
    """ერთი ეპოქის ვარჯიში (Training Loop)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    """ვალიდაციის/ტესტირების ფუნქცია (Validation Loop)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss, val_acc, val_f1

import time


def fit(model, train_loader, val_loader, epochs, lr, device, ckpt_path,
        weight_decay=1e-4, label_smoothing=0.0, use_wandb=False):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(v)

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                       "val_loss": va_loss, "val_acc": va_acc,
                       "lr": optimizer.param_groups[0]["lr"]})

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)

        print(f"epoch {epoch:02d}/{epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - t0:.0f}s")

    print(f"საუკეთესო val acc: {best_val_acc:.3f}")
    return history
