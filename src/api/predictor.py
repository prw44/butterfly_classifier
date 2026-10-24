from PIL import Image
import torch
from torchvision.models import resnet50
import torchvision.transforms as transforms
import torch.nn as nn
import yaml



class ImageClassifier:
    def __init__(self, model_path: str = "models/butterfly_classifier.pt", classes_path="models/classes.txt"):
        self.model_path = model_path
        with open(classes_path) as f:
            self.classes = f.read().splitlines()
        self.num_classes = len(self.classes)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    @classmethod    
    def init_from_config_path(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)

        model = cls(model_path = config["model_path"],
                    classes_path = config["classes_path"])

        return model

    
    def predict_from_file(self, file_object, top_k=3):
        image = Image.open(file_object).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]

        top_probs, top_idx = probs.topk(min(top_k, self.num_classes))
        return [
            {"species": self.classes[i], "confidence": round(p, 4)}
            for p, i in zip(top_probs.tolist(), top_idx.tolist())
        ]

    



        
       