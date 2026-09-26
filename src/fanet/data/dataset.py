"""Dataset loading for FANet training/evaluation."""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DATASET(Dataset):
    """Image/mask pair dataset with optional albumentations transform."""

    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = image.astype(np.float32)

        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0)
        mask = (mask > 127).astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples


def load_data(path):
    """Load dataset split names from a Kvasir-style folder with
    train.txt/val.txt files.

    Returns ((train_x, train_y), (valid_x, valid_y)) where each entry is a
    list of file paths.
    """

    def load_names(split_file):
        with open(split_file, "r") as f:
            data = f.read().split("\n")[:-1]
        images = [os.path.join(path, "images", name) + ".jpg" for name in data]
        masks = [os.path.join(path, "masks", name) + ".jpg" for name in data]
        return images, masks

    train_x, train_y = load_names(f"{path}/train.txt")
    valid_x, valid_y = load_names(f"{path}/val.txt")
    return (train_x, train_y), (valid_x, valid_y)


def rle_batch_to_tensor(mask_list, idx, batch, size):
    """Decode a slice of RLE-encoded feedback masks into a float tensor.

    mask_list: list of RLE runs (as produced by utils.rle_encode)
    idx, batch: slice bounds
    size: (H, W) target size
    Returns tensor of shape (batch, 1, H, W)
    """
    m = []
    for edata in mask_list[idx: idx + batch]:
        from ..utils import rle_decode
        decoded = rle_decode(str(" ".join(str(d) for d in edata)), size)
        decoded = np.expand_dims(decoded, axis=0)
        m.append(decoded)
    m = np.array(m, dtype=np.int32)
    m = np.transpose(m, (0, 1, 3, 2))
    return torch.from_numpy(m).float()
