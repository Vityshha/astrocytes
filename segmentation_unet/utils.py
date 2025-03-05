import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, epoch, folder, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    filename = f'my_checkpoint_{epoch}.pth.tar'
    torch.save(state, folder + filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
    val_ds = CarvanaDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    return train_loader, val_loader



def get_val_loader(val_dir, val_maskdir, batch_size, val_transform, num_workers, pin_memory):
    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return val_loader



def check_accuracy(loader, model, doorstep=0.5, device="cuda"):
    model.eval()
    num_correct, num_pixels, dice_score = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > doorstep).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Accuracy: {num_correct}/{num_pixels} = {num_correct / num_pixels * 100:.2f}%")
    print(f"Dice score: {dice_score / len(loader):.4f}")
    model.train()

def save_predictions_as_imgs(loader, model, doorstep=0.5, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > doorstep).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")

    model.train()
