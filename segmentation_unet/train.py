import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import os
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 0
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 720
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

RESULTS_SAVE_DIR = "saved_images/"
MODEL_SAVE_DIR = "saved_models/"

print('Init model...')
print('DEVICE: ', DEVICE)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training")

    epoch_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(DEVICE), targets.to(DEVICE).float().unsqueeze(1)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())
        epoch_loss += loss.item()

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / num_batches
    return avg_epoch_loss


def calculate_metrics(predictions, targets):
    # Применяем сигмоиду к предсказаниям
    predictions = torch.sigmoid(predictions)
    # Бинаризуем предсказания (например, порог 0.5)
    predictions = (predictions > 0.5).float()

    # Вычисляем TP, FP, FN
    TP = (predictions * targets).sum()
    FP = (predictions * (1 - targets)).sum()
    FN = ((1 - predictions) * targets).sum()

    # IoU
    iou = TP / (TP + FP + FN + 1e-8)  # Добавляем 1e-8 для избежания деления на 0

    # Dice Coefficient
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

    # Precision
    precision = TP / (TP + FP + 1e-8)

    # Recall
    recall = TP / (TP + FN + 1e-8)

    return iou.item(), dice.item(), precision.item(), recall.item()


def save_metrics_to_txt(epoch_metrics, save_dir):
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write("Epoch\tIoU\tDice\tPrecision\tRecall\n")
        for epoch, (iou, dice, precision, recall) in enumerate(epoch_metrics):
            f.write(f"{epoch + 1}\t{iou:.4f}\t{dice:.4f}\t{precision:.4f}\t{recall:.4f}\n")

def plot_loss(epoch_losses, save_dir):
    plt.figure()
    plt.plot(epoch_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_over_epochs.png"))
    plt.close()


def plot_metrics(epoch_metrics, save_dir):
    epochs = range(1, len(epoch_metrics) + 1)
    iou = [metrics[0] for metrics in epoch_metrics]
    dice = [metrics[1] for metrics in epoch_metrics]
    precision = [metrics[2] for metrics in epoch_metrics]
    recall = [metrics[3] for metrics in epoch_metrics]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, iou, label="IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU over Epochs")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, dice, label="Dice Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice Coefficient over Epochs")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, precision, label="Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision over Epochs")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, recall, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_over_epochs.png"))
    plt.close()


def main():
    # Data augmentations and preprocessing
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # Initialize model, loss, optimizer
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Data loaders
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
        BATCH_SIZE, train_transform, val_transforms, NUM_WORKERS, PIN_MEMORY
    )

    # Lists to store metrics and losses
    epoch_losses = []
    epoch_metrics = []

    # Load model if required
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        avg_epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save average loss for the epoch
        epoch_losses.append(avg_epoch_loss)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_iou, val_dice, val_precision, val_recall = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE).float().unsqueeze(1)
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                val_loss += loss.item()

                # Calculate metrics
                iou, dice, precision, recall = calculate_metrics(predictions, targets)
                val_iou += iou
                val_dice += dice
                val_precision += precision
                val_recall += recall

        # Average metrics over validation set
        num_val_batches = len(val_loader)
        val_loss /= num_val_batches
        val_iou /= num_val_batches
        val_dice /= num_val_batches
        val_precision /= num_val_batches
        val_recall /= num_val_batches

        # Save metrics for the epoch
        epoch_metrics.append((val_iou, val_dice, val_precision, val_recall))

        # Print metrics and save
        print(f"Validation Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        save_metrics_to_txt(epoch_metrics, MODEL_SAVE_DIR)

        # Save model checkpoint
        save_checkpoint({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, epoch, MODEL_SAVE_DIR)

    # Plot and save loss and metrics over epochs
    plot_loss(epoch_losses, MODEL_SAVE_DIR)
    plot_metrics(epoch_metrics, MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()