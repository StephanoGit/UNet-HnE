import argparse
import torch
from model import UNet
from dataset import dataset_loader
import torch.optim
import torch.nn
from eval_metrics import train_fn, valid_fn, custom_cross_entropy, Tversky_Focal_Loss

from utils import save_predictions_as_imgs, plot_metrics, save_checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau


def parse_args():
    parser = argparse.ArgumentParser(description="UNet Training Script")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_height", type=int, default=256, help="Image height")
    parser.add_argument("--img_width", type=int, default=256, help="Image width")
    parser.add_argument(
        "--train_img_dir", type=str, required=True, help="Directory of training images"
    )
    parser.add_argument(
        "--train_mask_dir", type=str, required=True, help="Directory of training masks"
    )
    parser.add_argument(
        "--val_img_dir", type=str, required=True, help="Directory of validation images"
    )
    parser.add_argument(
        "--val_mask_dir", type=str, required=True, help="Directory of validation masks"
    )
    parser.add_argument("--n_classes", type=int, default=1, help="Batch size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--pin_memory", type=bool, default=True, help="Pin memory")
    parser.add_argument(
        "--load_model", type=bool, default=False, help="Load pre-trained model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    parser.add_argument("--gamma", type=float, default=3.0, help="LossFn Gamma")
    parser.add_argument(
        "--weights",
        type=str,
        default="1.0,1.0,1.0,1.0,1.0",
        help="Comma-separated list of weights for the LossFn",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    WEIGHTS = [float(w) for w in args.weights.split(",")]
    LR = args.lr
    IMG_HEIGHT = args.img_height
    IMG_WIDTH = args.img_width
    TRAIN_IMG_DIR = args.train_img_dir
    TRAIN_MASK_DIR = args.train_mask_dir
    VALID_IMG_DIR = args.val_img_dir
    VALID_MASK_DIR = args.val_mask_dir
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    PIN_MEMORY = args.pin_memory
    LOAD_MODEL = args.load_model
    DEVICE = args.device
    GAMMA = args.gamma
    N_CLASSES = args.n_classes

    train_loader, valid_loader = dataset_loader(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VALID_IMG_DIR,
        VALID_MASK_DIR,
        16,
        PIN_MEMORY,
        BATCH_SIZE,
        IMG_HEIGHT,
        IMG_WIDTH,
        N_CLASSES,
    )

    print(f"Train Images: {len(train_loader.dataset)}")
    print(f"Valid Images: {len(valid_loader.dataset)}")

    model = UNet(3, N_CLASSES, dropout_rate=0.5).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    if N_CLASSES == 1:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        # loss_fn = custom_cross_entropy
        loss_fn = Tversky_Focal_Loss(weight=torch.tensor(WEIGHTS), device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    best_loss = 999.0
    train_loss_all, train_acc_all, train_dice_all = [], [], []
    valid_loss_all, valid_acc_all, valid_dice_all = [], [], []

    for e in range(EPOCHS):
        print(f"Epoch {e+1}/{EPOCHS}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.8f}")

        train_loss, train_acc, train_dice = train_fn(
            train_loader, model, optimizer, loss_fn, scaler, DEVICE
        )
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        train_dice_all.append(train_dice)

        print(f"Train AVG. Loss: {train_loss:.5f}")
        print(f"Train AVG. Accuracy: {train_acc}")
        print(f"Train AVG. Dice Score: {train_dice}")

        valid_loss, valid_acc, valid_dice = valid_fn(
            valid_loader, model, loss_fn, DEVICE
        )
        valid_loss_all.append(valid_loss)
        valid_acc_all.append(valid_acc)
        valid_dice_all.append(valid_dice)

        print(f"Valid AVG. Loss: {valid_loss:.5f}")
        print(f"Valid AVG. Accuracy: {valid_acc}")
        print(f"Valid AVG. Dice Score: {valid_dice}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, file_name=f"model_{e+1}.pth.tar")

        save_predictions_as_imgs(valid_loader, model, folder=f"predictions_epoch{e+1}/")
        scheduler.step(valid_loss)
        print("\n")

    plot_metrics(
        train_loss_all,
        train_acc_all,
        train_dice_all,
        valid_loss_all,
        valid_acc_all,
        valid_dice_all,
    )
