from collections import Counter
import seaborn as sns
import os
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import numpy as np
import matplotlib.pyplot as plt

# Загрузка модели
def load_model(model_path, device):
    model = models.vgg19()
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Подготовка изображения
def preprocess_image(image, transform, device):
    image = transform(image).unsqueeze(0)  # Добавляем батч-размер
    return image.to(device)

# Классификация изображения
def classify_image(model, image, device):
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]  # Вероятности для двух классов
    return probabilities

# Обработка всех изображений в папке
def process_folder(model, folder_path, class_names, transform, device):
    all_labels = []
    all_probs = []
    all_preds = []

    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.exists(class_folder):
            print(f"Folder {class_folder} does not exist. Skipping.")
            continue

        print(f"Processing folder: {class_folder}")
        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)
            try:
                # Открытие и предобработка изображения
                image = Image.open(image_path).convert("RGB")
                image = preprocess_image(image, transform, device)

                # Классификация
                probabilities = classify_image(model, image, device)
                predicted_class_idx = np.argmax(probabilities)
                predicted_class = class_names[predicted_class_idx]

                # Сохранение меток и предсказаний
                true_label = class_names.index(class_name)
                all_labels.append(true_label)
                all_preds.append(predicted_class_idx)
                all_probs.append(probabilities[1])  # Вероятность для класса "sick"

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Построение графиков
def plot_metrics(conf_matrix, labels, probs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # График матрицы ошибок
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = ["Healthy", "Sick"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Добавление текста в ячейки
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ROC-кривая
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--')  # Диагональная линия
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

# Настройки
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_path = "models/astrocyte_classifier_model.pth"
class_names = ["healthy", "sick"]

# Преобразования для изображения
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
])

# Загрузка модели
model = load_model(model_path, device)

# Путь к папке с изображениями
folder_path = "val_data_for_metrics"

# Обработка всех изображений
all_labels, all_preds, all_probs = process_folder(model, folder_path, class_names, data_transform, device)

# Вычисление метрик
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds,average="binary")
recall = recall_score(all_labels, all_preds,average="binary")
f1 = f1_score(all_labels, all_preds,average="binary")
roc_auc = roc_auc_score(all_labels, all_probs)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Метрики для класса "healthy"
precision_healthy = precision_score(all_labels, all_preds, labels=[0], average="macro")
recall_healthy = recall_score(all_labels, all_preds, labels=[0], average="macro")
f1_healthy = f1_score(all_labels, all_preds, labels=[0], average="macro")

print(f"Precision (Healthy): {precision_healthy:.4f}")
print(f"Recall (Healthy): {recall_healthy:.4f}")
print(f"F1 Score (Healthy): {f1_healthy:.4f}")

class_counts = Counter(all_labels)
total_samples = len(all_labels)
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
print("Class Weights:", class_weights)



def plot_class_metrics(precision, recall, f1, output_dir):
    metrics = {
        "Metric": ["Precision", "Recall", "F1 Score"],
        "Value": [precision, recall, f1],
    }
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Metric", y="Value", data=metrics)
    plt.title("Metrics for Binary Classification")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, "class_metrics.png"))
    plt.close()

# Построение графиков
output_dir = "../../../../Library/Application Support/JetBrains/PyCharmCE2023.2/scratches/temp-files/classification_metrics_plots_31.03.25_100_valmet"
plot_metrics(conf_matrix, all_labels, all_probs, output_dir)

# Вызов функции
plot_class_metrics(precision, recall, f1, output_dir)