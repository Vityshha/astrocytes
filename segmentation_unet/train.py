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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 2
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


def plot_loss(epoch_losses, save_dir):
    plt.figure()
    plt.plot(epoch_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_over_epochs.png"))
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

    # List to store average loss per epoch
    epoch_losses = []

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

        # Save model checkpoint
        save_checkpoint({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, epoch, MODEL_SAVE_DIR)

        # Check accuracy after each epoch
        # check_accuracy(val_loader, model, device=DEVICE)

        # Save predictions
        # save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

    # Plot and save loss over epochs
    plot_loss(epoch_losses, MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()