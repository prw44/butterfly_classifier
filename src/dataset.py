import os
from PIL import Image
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset





class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.data.iloc[:, 1].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image name and label from CSV
        img_name = self.data.iloc[idx, 0]
        label_str = self.data.iloc[idx, 1]
        label = self.class_to_idx[label_str]
        # Construct full path to image
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label



def create_dataloader(train = True):

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    if train:
        csv_file = "data/Training_set.csv"
        img_dir = "data/train"

    else:
        csv_file = "data/Testing_set.csv"
        img_dir = "data/test"
    
    dataset = CustomDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=train)

    return dataloader


def get_classes():
    dataset = CustomDataset(csv_file = "data/Training_set.csv", img_dir = "data/train")
    return dataset.classes
    






