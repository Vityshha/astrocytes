import torch
from torchvision import models, transforms
from PIL import Image
from matplotlib import pyplot as plt


class AstrocytesClassificator:
    def __init__(
            self,
            weights_path: str = "./models/astrocyte_classifier_model.pth",
            device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    ):
        self.weights_path = weights_path
        self.device = device
        self.class_names = ["healthy", "sick"]

        self._init_model()
        self._init_transform()

    def _init_model(self):
        self.model = models.vgg19()
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(num_features, 2)
        self.model.load_state_dict(torch.load( self.weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _init_transform(self):
        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ])

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.data_transform(image)
        image = image.unsqueeze(0)
        return image.to(self.device)

    def _model_inference(self, image):
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            print('probabilities: ', probabilities)

        predicted_class_idx = probabilities.argmax()
        predicted_class = self.class_names[predicted_class_idx]
        probability = probabilities[predicted_class_idx]

        return predicted_class, probability

    def classify(self):
        image_path = "train_data/val/sick/croped_mask_60.png"
        image = self.load_and_preprocess_image(image_path)

        predicted_class, probability = self._model_inference(image)
        print(f"Predicted Class: {predicted_class} (Probability: {probability:.4f})")


test = AstrocytesClassificator()
test.classify()