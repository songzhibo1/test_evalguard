"""
EvalGuard — Teacher Pre-Training Script

Trains teacher models for datasets without standard pretrained weights
(e.g., Tiny-ImageNet-200). Saves weights to pretrained/<name>.pt.

Usage:
  python train_teacher.py --dataset tinyimagenet --epochs 100 --device cuda
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from evalguard.configs import tinyimagenet_data, _build_resnet18_64


def train(model, trainloader, testloader, epochs, lr, device):
    model.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
        scheduler.step()
        train_acc = correct / total * 100

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                correct += model(x).argmax(1).eq(y).sum().item()
                total += y.size(0)
        test_acc = correct / total * 100

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("Epoch {:3d}/{}: loss={:.4f}, train_acc={:.1f}%, test_acc={:.1f}% (best={:.1f}%)".format(
                epoch + 1, epochs, total_loss / len(trainloader),
                train_acc, test_acc, best_acc))

    return best_state, best_acc


def main():
    pa = argparse.ArgumentParser(description="Train teacher for EvalGuard")
    pa.add_argument("--dataset", default="tinyimagenet",
                    choices=["tinyimagenet"])
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--lr", type=float, default=0.1)
    pa.add_argument("--batch_size", type=int, default=128)
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = pa.parse_args()

    out_dir = Path("pretrained")
    out_dir.mkdir(exist_ok=True)

    if a.dataset == "tinyimagenet":
        trainset, testset, trainloader, testloader = tinyimagenet_data(a.batch_size)
        model = _build_resnet18_64(num_classes=200)
        out_path = out_dir / "tinyimagenet_resnet18.pt"
        print("Training ResNet-18 on Tiny-ImageNet-200 ({} train, {} test)".format(
            len(trainset), len(testset)))
    else:
        raise ValueError("Unknown dataset: {}".format(a.dataset))

    best_state, best_acc = train(model, trainloader, testloader,
                                  a.epochs, a.lr, a.device)

    torch.save(best_state, out_path)
    print("\nSaved best model (acc={:.1f}%) to {}".format(best_acc, out_path))


if __name__ == "__main__":
    main()
