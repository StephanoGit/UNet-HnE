import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

COLOR_MAP = torch.tensor(
    [
        [225, 255, 255],
        [0, 255, 0],  # 1 - Green
        [0, 0, 255],  # 2 - Blue
        [255, 255, 0],  # 3 - Yellow
        [255, 0, 0],  # 4 - Red
    ],
    dtype=torch.uint8,
    device="cuda",
)


def save_checkpoint(state, file_name="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, file_name)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def plot_single_metric(epochs, metric_values, labels, ylabel, title, save_path, colors):
    plt.figure(figsize=(12, 8))
    for i, (label, color) in enumerate(zip(labels, colors)):
        plt.plot(
            range(1, epochs + 1),
            metric_values[i],
            label=label,
            color=color,
            linewidth=2,
        )
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics(
    train_loss_all,
    train_acc_all,
    train_dice_all,
    valid_loss_all,
    valid_acc_all,
    valid_dice_all,
    save_dir="plots/",
):
    print("=> Saving Plots")
    epochs = len(train_loss_all)
    class_names = ["Background", "Benign", "G3", "G4", "G5"]
    colors = ["black", "green", "blue", "yellow", "red"]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Common style settings
    plt.style.use("seaborn-v0_8")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.titlesize"] = 20

    # Plot Training and Validation Loss
    plot_single_metric(
        epochs,
        [train_loss_all, valid_loss_all],
        ["Train Loss", "Val Loss"],
        "Loss",
        "Training and Validation Loss",
        os.path.join(save_dir, "train_valid_loss.png"),
        ["blue", "orange"],
    )

    # Plot Training and Validation Accuracy
    plot_single_metric(
        epochs,
        [train_acc_all, valid_acc_all],
        ["Train Accuracy", "Val Accuracy"],
        "Accuracy",
        "Training and Validation Accuracy",
        os.path.join(save_dir, "train_valid_accuracy.png"),
        ["blue", "orange"],
    )
    # Plot Dice Scores
    plot_single_metric(
        epochs,
        list(
            zip(*train_dice_all)
        ),  # Transpose the list to get values for each class across epochs
        class_names,
        "Dice Score",
        "Training Dice Scores",
        os.path.join(save_dir, "train_dice_scores.png"),
        colors,
    )

    # Plot Dice Scores
    plot_single_metric(
        epochs,
        list(
            zip(*valid_dice_all)
        ),  # Transpose the list to get values for each class across epochs
        class_names,
        "Dice Score",
        "Validation Dice Scores",
        os.path.join(save_dir, "valid_dice_scores.png"),
        colors,
    )


def save_heatmaps(tensor, folder, batch_idx, normalize=True):
    tensor = tensor.cpu().detach().numpy()

    if not os.path.exists(folder):
        os.makedirs(folder)

    num_samples, num_channels, height, width = tensor.shape

    for s in range(min(num_samples, 8)):
        image_tensor = tensor[s]  # Shape (C, H, W)

        fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
        for c in range(num_channels):
            channel_data = image_tensor[c]

            if normalize:
                channel_data = (channel_data - np.min(channel_data)) / (
                    np.max(channel_data) - np.min(channel_data)
                )

            im = axes[c].imshow(channel_data, cmap="hot", interpolation="sinc")
            axes[c].set_title(f"Channel {c}")
            axes[c].axis("off")

        cbar = plt.colorbar(
            im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1
        )
        cbar.set_label("Intensity")

        file_path = os.path.join(folder, f"{s}.png")
        plt.savefig(file_path)
        plt.close()


def create_composite_image(pred, truth):
    pred_binary = (pred > 0.5).float()
    truth_binary = (truth > 0.5).float()

    comp_imgs = torch.zeros(
        (pred.shape[0], 3, pred.shape[2], pred.shape[3]),
        dtype=torch.float32,
        device=pred.device,
    )

    # True Positive (TP): Green
    tp = (pred_binary == 1) & (truth_binary == 1)
    comp_imgs[:, 1, :, :][tp.squeeze(1)] = 1.0

    # False Positive (FP): Red
    fp = (pred_binary == 1) & (truth_binary == 0)
    comp_imgs[:, 0, :, :][fp.squeeze(1)] = 1.0

    # False Negative (FN): Blue
    fn = (pred_binary == 0) & (truth_binary == 1)
    comp_imgs[:, 2, :, :][fn.squeeze(1)] = 1.0

    return comp_imgs


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx > 10:
            break
        x = x.to(device)
        y = y.float().to(device)

        with torch.no_grad():
            if model.out_channels == 1:
                preds = torch.sigmoid(model(x))

                comp_img = create_composite_image(preds, y)
                comp_with_original = torch.cat((x, comp_img), dim=0)
                torchvision.utils.save_image(comp_with_original, f"{folder}/{idx}.png")
            else:
                preds = torch.softmax(model(x), dim=1)
                class_preds = torch.argmax(preds, dim=1)  # B H W
                class_label = torch.argmax(y, dim=1)  # B H W

                save_heatmaps(preds.clone(), f"{folder}/heatmaps/", idx)

                color_preds = COLOR_MAP[class_preds]  # idx to color 0 -> [0, 255, 0]
                color_preds = color_preds.permute(0, 3, 1, 2).float()

                color_label = COLOR_MAP[class_label]
                color_label = color_label.permute(0, 3, 1, 2).float()

                x_y_p = torch.cat((x, color_label, color_preds), dim=3)
                x_y_p = x_y_p.view(-1, 3, x.shape[2], x.shape[3] * 3)
                torchvision.utils.save_image(x_y_p, f"{folder}/{idx}.png")
