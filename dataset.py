from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import os
import numpy as np
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
import torch
from collections import Counter
from albumentations.pytorch import ToTensorV2

COLOR_MAP = {
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "red": (255, 0, 0),
}


CLASS_LABELS = {
    "gray": 0,
    "white": 1,
    "green": 2,
    "blue": 3,
    "yellow": 4,
    "red": 5,
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
        self.images = []
        self.labels = []
        self.class_distribution = Counter()
        self.filter_images()

    def is_valid_mask(self, mask_path):
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        center_patch = mask[
            mask.shape[0] // 2 - 256 : mask.shape[0] // 2 + 256,
            mask.shape[1] // 2 - 256 : mask.shape[1] // 2 + 256,
        ]

        total_pixels = center_patch.shape[0] * center_patch.shape[1]
        color_counts = {}

        for color_name, rgb in COLOR_MAP.items():
            color_mask = np.all(center_patch == rgb, axis=-1)
            color_counts[color_name] = np.sum(color_mask)

        background_colors = ["white", "gray"]
        non_background_colors = set(COLOR_MAP.keys()) - set(background_colors)

        present_non_background_colors = [
            color for color in non_background_colors if color_counts[color] > 0
        ]

        if len(present_non_background_colors) > 1:
            return None

        for color in non_background_colors:
            if color_counts[color] / total_pixels >= 0.4:
                return CLASS_LABELS[color]

        if (color_counts["white"] + color_counts["gray"]) / total_pixels >= 0.5:
            if color_counts["gray"] > color_counts["white"]:
                return None
            else:
                return CLASS_LABELS["white"]

        return None

    def filter_images(self):
        for image in os.listdir(self.img_dir):
            if not image.startswith("."):
                if self.validation is not None:
                    mask_path = os.path.join(
                        self.msk_dir, self.validation + image.replace(".jpg", ".png")
                    )
                else:
                    mask_path = os.path.join(
                        self.msk_dir, "mask_" + image.replace(".jpg", ".png")
                    )
                label = self.is_valid_mask(mask_path)

                if label is not None:
                    self.images.append(image)
                    self.labels.append(label)
                    self.class_distribution[label] += 1

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

    def get_class_distribution(self):
        return dict(self.class_distribution)


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

    # Calculate class weights based on class distribution
    class_counts = train_dataset.get_class_distribution()
    total_samples = sum(class_counts.values())

    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Assign weights to each sample in the dataset
    # sample_weights = [
    #     class_weights[train_dataset.labels[i]] for i in range(len(train_dataset))
    # ]

    # # Create a WeightedRandomSampler
    # weighted_sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True,  # With replacement for random sampling
    # )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory,
        shuffle=True,
        # sampler=weighted_sampler,
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

    print(train_dataset.get_class_distribution())

    # for i in range(0, 15):
    # show_image_and_mask(train_dataset, train_dataset.labels[i], i)

    return train_loader, valid_loader, class_weights
