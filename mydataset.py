import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
            
        return image, label

    def get_classes(self):
        return self.classes

if __name__ == '__main__':
    # Test the dataset
    data_dir = 'dataset/train'  # Change this to your dataset path
    
    # Create dataset instance
    dataset = CustomDataset(data_dir)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test loading some samples
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"Image shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Classes: {[dataset.classes[label] for label in labels]}")
        
        if i >= 2:  # Print first 3 batches only
            break
    
    print("\nDataset Summary:")
    print(f"Total number of images: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")