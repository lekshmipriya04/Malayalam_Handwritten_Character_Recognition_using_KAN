import argparse
import time

import pandas as pd
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import MalayalamOCRNet


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100 * correct / total


def run_training(
    head_type="kan2",
    epochs=40,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    img_size=64,
    kan_hidden=128,
    kan_grid=8,
    kan_order=3,
    dropout=0.3,
    label_smoothing=0.05,
    use_sampler=True,
    patience=12,
    device=None,
    num_workers=8,
    log_prefix="",
    verbose=True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, classes, class_weight = get_dataloaders(
        img_size=img_size, batch_size=batch_size, num_workers=num_workers, use_sampler=use_sampler
    )
    num_classes = len(classes)

    model = MalayalamOCRNet(
        num_classes=num_classes,
        head_type=head_type,
        kan_hidden=kan_hidden,
        kan_grid=kan_grid,
        kan_order=kan_order,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.15
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = []
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "time_s": time.time() - t0,
            }
        )

        if verbose:
            print(
                f"{log_prefix}epoch {epoch+1}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
                f"({time.time()-t0:.1f}s)"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"{log_prefix}early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "classes": classes,
        "num_params": sum(p.numel() for p in model.parameters()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-type", default="kan2", choices=["linear", "kan1", "kan2"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--out-prefix", default="run")
    args = parser.parse_args()

    result = run_training(
        head_type=args.head_type,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        img_size=args.img_size,
        patience=args.patience,
    )
    print(f"\nBest val acc: {result['best_val_acc']:.2f}%")
    print(f"Test acc: {result['test_acc']:.2f}%")
    print(f"Num params: {result['num_params']:,}")

    result["history"].to_csv(f"{args.out_prefix}_history.csv", index=False)
    torch.save(result["model"].state_dict(), f"{args.out_prefix}_best_model.pth")
