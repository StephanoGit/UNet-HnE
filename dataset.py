from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import os
import numpy as np
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
import torch

COLOR_RULES = {
    4: {  # Red color
        "ranges": [
            (0.75, 1.0, 5),  # Above 50%, 5 augmentations
            (0.5, 0.75, 4),  # Between 20% and 50%, 3 augmentations
            (0.2, 0.5, 3),  # Between 20% and 50%, 3 augmentations
        ],
    },
    3: {
        "ranges": [
            (0.5, 1.0, 4),
            (0.2, 0.5, 3),  # Between 20% and 50%, 3 augmentations
        ],
    },
    2: {
        "ranges": [
            (0.5, 1.0, 3),  # Above 50%, 5 augmentations
            (0.2, 0.5, 2),  # Between 20% and 50%, 3 augmentations
        ],
    },
    1: {
        "ranges": [
            (0.75, 1.0, 3),  # Above 50%, 5 augmentations
            (0.5, 0.75, 2),
        ],
    },
}

COLOR_MAP = {
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "red": (255, 0, 0),
}

COLOR_MAP2 = {
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "red": (255, 0, 0),
}


def show_image_and_mask(dataset, label, idx):
    # Get the image and mask
    image, mask = dataset[idx]

    # Convert the image from torch tensor to numpy array for visualization
    image_np = image.permute(1, 2, 0).numpy()  # Convert CHW -> HWC

    # Reduce mask from (H, W, n_classes) to (H, W) using argmax to get class labels
    mask_np = mask.permute(1, 2, 0).numpy()  # Convert CHW -> HWC
    mask_np = np.argmax(mask_np, axis=-1)  # Get class index at each pixel

    # Plot image and mask side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display image
    ax[0].imshow(image_np)
    ax[0].set_title("Image")
    ax[0].axis("off")

    # Display mask
    ax[1].imshow(mask_np, cmap="jet")  # Use a colormap to visualize class regions
    ax[1].set_title(f"{label}")
    ax[1].axis("off")

    plt.show()


class HnE(Dataset):
    def __init__(self, img_dir, msk_dir, n_classes, transform=None, validation=None):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.n_classes = n_classes
        self.transform = transform
        self.validation = validation
        self.labels = []
        self.class_distribution = {
            i: {"original": 0, "augmented": 0}
            for i, color in enumerate(COLOR_MAP2.values())
        }

        self.images = [
            f
            for f in os.listdir(self.img_dir)
            if not f.startswith(".") and (f.endswith(".png") or f.endswith(".jpg"))
        ][:400]

        if self.validation is None:
            self.image_augmentations = self._determine_augmentations()

    def _determine_augmentations(self):
        image_augmentations = []

        for img_name in self.images:
            mask_path = os.path.join(
                self.msk_dir, "mask_" + img_name.replace(".jpg", ".png")
            )
            mask = np.array(Image.open(mask_path).convert("RGB"))

            total_pixels = mask.shape[0] * mask.shape[1]
            color_proportions = {
                i: np.all(mask == rgb, axis=-1).sum() / total_pixels
                for i, (color, rgb) in enumerate(COLOR_MAP2.items())
            }

            dominant_color = max(color_proportions, key=color_proportions.get)
            self.class_distribution[dominant_color]["original"] += 1

            num_augmentations = 1
            for color, rule in COLOR_RULES.items():
                proportion = color_proportions.get(color, 0)
                for min_val, max_val, aug_count in rule["ranges"]:
                    if min_val <= proportion <= max_val:
                        num_augmentations = aug_count
                        self.class_distribution[color]["augmented"] += aug_count
                        break  # Use the first matching range
                if num_augmentations > 1:  # first matching color
                    break
                else:
                    self.class_distribution[dominant_color]["augmented"] += 1

            image_augmentations.extend(
                [(img_name, i) for i in range(num_augmentations)]
            )

        print(f"Total image augmentations: {len(image_augmentations)}")
        print(f"Class Distribution: {self.class_distribution}")
        return image_augmentations

    def __len__(self):
        return (
            len(self.image_augmentations)
            if self.validation is None
            else len(self.images)
        )

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
        if self.validation is None:
            img_name, _ = self.image_augmentations[idx]
        else:
            img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        if self.validation is not None:
            mask_path = os.path.join(
                self.msk_dir, self.validation + img_name.replace(".jpg", ".png")
            )
        else:
            mask_path = os.path.join(
                self.msk_dir, "mask_" + img_name.replace(".jpg", ".png")
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
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(alpha=1.5, sigma=10, p=0.3),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.2,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
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

    augmented_values = [
        color_data["augmented"]
        for color_data in train_dataset.class_distribution.values()
    ]

    total_augmented = sum(augmented_values)

    weights = {
        color: total_augmented / (color_data["augmented"] + 1e-8)
        for color, color_data in train_dataset.class_distribution.items()
    }

    w_list = list(weights.values())
    w_list[0] = 1.0

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

    print(train_dataset.class_distribution)
    print(w_list)

    # for i in range(0, 15):
    # show_image_and_mask(train_dataset, train_dataset.labels[i], i)

    return train_loader, valid_loader, w_list
