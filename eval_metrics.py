from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class Tversky_Focal_Loss(nn.Module):
    def __init__(
        self, device, weight=None, alpha=0.5, beta=0.5, gamma=2.0, epsilon=1e-7
    ):
        super(Tversky_Focal_Loss, self).__init__()
        if weight is not None:
            self.weight = weight.to(device=device)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

    def forward(self, activations, annotations):
        activations = activations.float().to(self.device)
        annotations = annotations.float().to(self.device)

        # Softmax across channels to get class probabilities
        probabilities = torch.softmax(activations, dim=1)  # Shape: [B, C, H, W]
        probabilities = torch.clamp(
            probabilities, min=self.epsilon, max=1 - self.epsilon
        )

        # Tversky loss computation
        true_pos = (probabilities * annotations).sum(
            dim=(2, 3)
        )  # Sum over spatial dimensions (H, W), resulting in [B, C]
        false_neg = ((1 - probabilities) * annotations).sum(
            dim=(2, 3)
        )  # Sum over spatial dimensions (H, W), resulting in [B, C]
        false_pos = (probabilities * (1 - annotations)).sum(
            dim=(2, 3)
        )  # Sum over spatial dimensions (H, W), resulting in [B, C]

        # Compute Tversky index per class
        tversky_index = (true_pos + self.epsilon) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.epsilon
        )  # Shape: [B, C]

        # Tversky loss per class, averaged over batch and classes
        tversky_loss = 1 - tversky_index  # Shape: [B, C]
        t_loss = tversky_loss.mean()  # Average over batch and classes

        # Focal loss computation
        pt = torch.where(
            annotations > 0, probabilities, 1 - probabilities
        )  # Shape: [B, C, H, W]
        focal_loss = -torch.pow(1 - pt, self.gamma) * torch.log(pt)

        # Apply class weights if provided
        if self.weight is not None:
            focal_loss *= self.weight.view(1, -1, 1, 1)  # Shape: [1, C, 1, 1]

        # Focal loss averaged over spatial dimensions and batch
        f_loss = focal_loss.mean(
            dim=(2, 3)
        )  # Average over spatial dims (H, W) resulting in [B, C]
        f_loss = f_loss.mean()  # Average over batch and classes

        # Final loss combination
        total_loss = f_loss * 0.2 + t_loss * 0.8
        return total_loss


def custom_cross_entropy(predictions, targets):
    return F.cross_entropy(predictions, targets, reduction="mean")


def accuracy_fn(predictions, targets):
    if targets.shape[1] == 1:
        predictions = torch.sigmoid(predictions)
        predicted_classes = predictions > 0.5
        target_classes = targets > 0.5
    else:
        predictions = torch.softmax(predictions, dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)  # (B, H, W)
        target_classes = torch.argmax(targets, dim=1)  # (B, H, W)

    num_classes = targets.shape[1]
    accuracies = []

    for class_idx in range(num_classes):
        if num_classes == 1:
            class_pred = predicted_classes
            class_target = target_classes
        else:
            class_pred = predicted_classes == class_idx
            class_target = target_classes == class_idx

        correct = (class_pred == class_target).float()
        class_acc = correct.sum() / correct.numel()
        accuracies.append(class_acc.item())

    return np.array(accuracies)


def class_acc_fn(predictions, targets):
    # predictions, targets: (B, C, H, W)

    predictions = torch.softmax(predictions, dim=1)
    predicted_classes = torch.argmax(predictions, dim=1)  # (B, H, W)

    accs = []
    for idx in range(targets.shape[1]):
        class_target = targets[:, idx]
        class_correct = ((predicted_classes == idx) & (class_target == 1)).float().sum()
        class_total = class_target.sum()
        if class_total > 0:
            accs.append((class_correct / class_total).item())
        else:
            accs.append(0.0)
    return accs


def dice_fn(predictions, targets, epsilon=1e-6):
    if targets.shape[1] == 1:
        predictions = torch.sigmoid(predictions)
        predicted_classes = predictions > 0.5
        target_classes = targets > 0.5
    else:
        predictions = torch.softmax(predictions, dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)  # (B, H, W)
        target_classes = torch.argmax(targets, dim=1)  # (B, H, W)

    num_classes = targets.shape[1]
    dice_scores = []

    for class_idx in range(num_classes):
        if num_classes == 1:
            pred_flat = predicted_classes.view(-1)
            target_flat = target_classes.view(-1)
        else:
            pred_flat = (predicted_classes == class_idx).view(-1)
            target_flat = (target_classes == class_idx).view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice_coeff = (2.0 * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice_coeff.item())

    return np.array(dice_scores)


def valid_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    per_class_acc = np.zeros(model.out_channels)
    per_class_dice = np.zeros(model.out_channels)
    loop = tqdm(loader, desc="Validation")

    for idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.float().to(device)

        with torch.no_grad():
            predictions = model(images)
            loss = loss_fn(predictions, labels)

            batch_class_acc = accuracy_fn(predictions, labels)
            batch_class_dice = dice_fn(predictions, labels)

        per_class_acc += batch_class_acc
        per_class_dice += batch_class_dice

        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    avg_per_class_acc = per_class_acc / len(loader)
    avg_per_class_dice = per_class_dice / len(loader)

    return (
        total_loss / len(loader),
        avg_per_class_acc,
        avg_per_class_dice,
    )


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0
    per_class_acc = np.zeros(model.out_channels)
    per_class_dice = np.zeros(model.out_channels)
    loop = tqdm(loader, desc="Training")

    for idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.float().to(device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            batch_class_acc = accuracy_fn(predictions, labels)
            batch_class_dice = dice_fn(predictions, labels)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        per_class_acc += batch_class_acc
        per_class_dice += batch_class_dice

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    avg_per_class_acc = per_class_acc / len(loader)
    avg_per_class_dice = per_class_dice / len(loader)

    return (
        total_loss / len(loader),
        avg_per_class_acc,
        avg_per_class_dice,
    )
