import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from collections import Counter
from sklearn.metrics import accuracy_score
import numpy as np




class VGGTrainer:
    def __init__(
            self,
            epochs = 100,
            train_dir = "train_data/train",
            val_dir = "train_data/val",
            test_dir = "train_data/test",
            output_dir = "models",
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ):

        self.epochs = epochs
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.device = device
        self.output_dir = output_dir
        print(f"DEVICE: {device}")

        self._init_transforms()
        self._load_dataset()
        self._init_vgg_model()

    def _init_transforms(self):
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(35),
                transforms.ToTensor(),
                transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            ])
        }

    def _load_dataset(self):
        image_datasets = {
            'train': ImageFolder(self.train_dir, self.data_transforms['train']),
            'val': ImageFolder(self.val_dir, self.data_transforms['val']),
            'test': ImageFolder(self.test_dir, self.data_transforms['test'])
        }

        self.dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=16, shuffle=True),
            'val': DataLoader(image_datasets['val'], batch_size=16, shuffle=False),
            'test': DataLoader(image_datasets['test'], batch_size=16, shuffle=False)
        }

        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        class_names = image_datasets['train'].classes

    def _calculate_weighted_accuracy(self, true_labels, predicted_labels):
        """
        Рассчитывает взвешенную точность, учитывая распределение классов.
        """
        class_counts = Counter(true_labels)
        total_samples = len(true_labels)

        weighted_accuracy = 0.0
        for cls in class_counts:
            cls_mask = true_labels == cls
            cls_true = true_labels[cls_mask]
            cls_pred = predicted_labels[cls_mask]
            cls_accuracy = accuracy_score(cls_true, cls_pred)
            weighted_accuracy += cls_accuracy * (class_counts[cls] / total_samples)

        return weighted_accuracy

    def _init_vgg_model(self):
        self.model = models.vgg19(weights="IMAGENET1K_V1")

        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, 2)

        model = self.model.to(self.device)

        # Расчет весов классов для компенсации дисбаланса
        num_healthy_train = 26
        num_sick_train = 44
        weights = torch.tensor([1.0 / num_healthy_train, 1.0 / num_sick_train])
        weights = weights / weights.sum()
        weights = weights.to(self.device)

        # Критерий и оптимизатор
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

    def train(self):
        best_acc = 0.0
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        best_model_wts = None

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print("-" * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                all_labels = []
                all_preds = []

                for inputs, labels in self.dataloaders[phase]:
                    inputs, labels = inputs.to(self.device).float(), labels.long().to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        preds = torch.argmax(outputs, dim=1)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = self._calculate_weighted_accuracy(np.array(all_labels), np.array(all_preds))

                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Weighted Acc: {epoch_acc:.4f}")

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
                elif phase == 'val':
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = self.model.state_dict()

        if best_model_wts is not None:
            print(f"Best val Weighted Acc: {best_acc:.4f}")
            self.model.load_state_dict(best_model_wts)

            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), f"{self.output_dir}/astrocyte_classifier_model.pth")
            print(f'model saved: {self.output_dir}/astrocyte_classifier_model.pth"')

            return self.model, train_loss_history, train_acc_history, val_loss_history, val_acc_history


trainer = VGGTrainer(epochs=2)
trainer.train()



