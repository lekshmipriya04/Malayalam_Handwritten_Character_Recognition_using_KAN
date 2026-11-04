import os
import random

import numpy as np
import torch
from PIL import ImageFilter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

DATA_ROOT = "C:\\Users\\angel\\Downloads\\kan_ocr_v2\\output"


class RescaleBinaryPNG:
    """Handle both legacy {0,1} PNGs and viewable {0,255} PNGs.

    Legacy pixels are scaled to {0,1} by ToTensor(), so restore that range
    after resize while preserving antialiasing. Viewable PNGs already have
    the expected [0,1] tensor range and must not be multiplied again.
    """

    def __call__(self, tensor):
        if tensor.max() <= (1.0 / 255.0):
            return (tensor * 255.0).clamp(0.0, 1.0)
        return (tensor >= 0.5).to(dtype=tensor.dtype)


class RandomMorphology:
    """Randomly dilate or erode strokes — a standard handwriting/OCR
    augmentation for stroke-width invariance that plain affine transforms
    don't provide."""

    def __init__(self, p=0.25, sizes=(3, 5)):
        self.p = p
        self.sizes = sizes

    def __call__(self, img):
        if random.random() > self.p:
            return img
        size = random.choice(self.sizes)
        if random.random() < 0.5:
            return img.filter(ImageFilter.MaxFilter(size))  # dilate (strokes are bright)
        return img.filter(ImageFilter.MinFilter(size))  # erode


def build_transforms(img_size, train):
    ops = [transforms.Grayscale(num_output_channels=1)]
    if train:
        ops.append(RandomMorphology(p=0.25))
        ops.append(
            transforms.RandomAffine(
                degrees=8, translate=(0.08, 0.08), shear=5, scale=(0.9, 1.1)
            )
        )
    ops.append(transforms.Resize((img_size, img_size)))
    ops.append(transforms.ToTensor())
    ops.append(RescaleBinaryPNG())
    if train:
        ops.append(
            transforms.RandomErasing(p=0.15, scale=(0.01, 0.04), ratio=(0.5, 2.0), value=0)
        )
    return transforms.Compose(ops)


def make_sample_weights(dataset, power=0.5):
    """Inverse-frequency**power per-sample weights for a WeightedRandomSampler.
    power<1 softens the imbalance so the 4-image class isn't oversampled into
    dominating minibatches; also returns per-class weights for an optional
    loss-side weighting."""
    targets = np.array([s[1] for s in dataset.samples])
    counts = np.bincount(targets)
    class_weight = 1.0 / np.power(counts, power)
    sample_weights = class_weight[targets]
    return (
        torch.as_tensor(sample_weights, dtype=torch.double),
        torch.as_tensor(class_weight / class_weight.mean(), dtype=torch.float),
    )


def get_dataloaders(img_size=64, batch_size=128, num_workers=8, root=DATA_ROOT, use_sampler=True):
    train_ds = datasets.ImageFolder(
        os.path.join(root, "train"), transform=build_transforms(img_size, True)
    )
    val_ds = datasets.ImageFolder(
        os.path.join(root, "val"), transform=build_transforms(img_size, False)
    )
    test_ds = datasets.ImageFolder(
        os.path.join(root, "test"), transform=build_transforms(img_size, False)
    )

    sample_weights, class_weight = make_sample_weights(train_ds)

    if use_sampler:
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader, train_ds.classes, class_weight
