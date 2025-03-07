import re
import matplotlib.pyplot as plt


'''
Лучшая эпоха: 47 (46 модель) (максимальные IoU и Dice, сбалансированные Precision и Recall, низкие потери).
'''

def parse_logs(log_file):
    train_loss_pattern = re.compile(r'loss=([\d.]+)')
    val_loss_pattern = re.compile(r'Validation Loss: ([\d.]+)')
    iou_pattern = re.compile(r'IoU: ([\d.]+)')
    dice_pattern = re.compile(r'Dice: ([\d.]+)')
    precision_pattern = re.compile(r'Precision: ([\d.]+)')
    recall_pattern = re.compile(r'Recall: ([\d.]+)')

    train_losses = []
    val_losses = []
    ious = []
    dices = []
    precisions = []
    recalls = []

    with open(log_file, 'r') as file:
        for line in file:
            train_loss_match = train_loss_pattern.search(line)
            val_loss_match = val_loss_pattern.search(line)
            iou_match = iou_pattern.search(line)
            dice_match = dice_pattern.search(line)
            precision_match = precision_pattern.search(line)
            recall_match = recall_pattern.search(line)

            if train_loss_match:
                train_losses.append(float(train_loss_match.group(1)))
            if val_loss_match:
                val_losses.append(float(val_loss_match.group(1)))
            if iou_match:
                ious.append(float(iou_match.group(1)))
            if dice_match:
                dices.append(float(dice_match.group(1)))
            if precision_match:
                precisions.append(float(precision_match.group(1)))
            if recall_match:
                recalls.append(float(recall_match.group(1)))

    return train_losses, val_losses, ious, dices, precisions, recalls

log_file = 'saved_models/LOGS.txt'
train_losses, val_losses, ious, dices, precisions, recalls = parse_logs(log_file)

print("Train Losses:", train_losses)
print("Validation Losses:", val_losses)
print("IoUs:", ious)
print("Dices:", dices)
print("Precisions:", precisions)
print("Recalls:", recalls)



def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metrics(ious, dices, precisions, recalls):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(ious, label='IoU', marker='o', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(dices, label='Dice', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Dice over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(precisions, label='Precision', marker='o', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(recalls, label='Recall', marker='o', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_losses(train_losses, val_losses)
plot_metrics(ious, dices, precisions, recalls)