from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
import torch

COLOR_MAP = {
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "red": (255, 0, 0),
}


class HnE(Dataset):
    def __init__(self, img_dir, msk_dir, n_classes, transform=None, validation=None):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.n_classes = n_classes
        self.transform = transform
        self.validation = validation
        self.images = [
            f
            for f in os.listdir(img_dir)
            if not f.startswith(".") and f.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.images)

    def rgb_to_label(self, mask):
        label_mask = np.zeros(
            (
                mask.shape[0],
                mask.shape[1],
                self.n_classes,
            ),
            dtype=np.uint8,
        )
        if self.n_classes == 1:
            is_white = np.all(mask == np.array(COLOR_MAP["white"]), axis=-1)
            is_gray = np.all(mask == np.array(COLOR_MAP["gray"]), axis=-1)
            is_green = np.all(mask == np.array(COLOR_MAP["green"]), axis=-1)
            cancer_pixels = ~(is_white | is_gray | is_green)
            label_mask[cancer_pixels] = 1
        else:
            for idx, rgb in enumerate(list(COLOR_MAP.values())[1:]):
                color_mask = np.all(mask == np.array(rgb), axis=-1)
                label_mask[color_mask, idx] = 1

        return label_mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])

        if self.validation is not None:
            mask_path = os.path.join(
                self.msk_dir, self.validation + self.images[idx].replace(".jpg", ".png")
            )
        else:
            mask_path = os.path.join(
                self.msk_dir, "mask_" + self.images[idx].replace(".jpg", ".png")
            )

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)

        mask = self.rgb_to_label(mask)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask


def dataset_loader(
    train_img_dir,
    train_msk_dir,
    valid_img_dir,
    valid_msk_dir,
    n_workers,
    pin_memory,
    batch_size,
    img_height,
    img_width,
    n_classes,
):
    train_transform = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
            A.Rotate(limit=90, p=0.5),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
        ]
    )

    valid_transform = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
        ]
    )

    train_dataset = HnE(
        train_img_dir, train_msk_dir, transform=train_transform, n_classes=n_classes
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    valid_dataset = HnE(
        valid_img_dir,
        valid_msk_dir,
        validation="mask1_",
        transform=valid_transform,
        n_classes=n_classes,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader
