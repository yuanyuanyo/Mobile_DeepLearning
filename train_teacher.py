import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from tqdm import tqdm
import os
import logging
from mydataset import CustomDataset

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, save_dir='ResNet_checkpoint'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.criterion = nn.CrossEntropyLoss()
        # Using SGD instead of Adam as it often works better with ResNet
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, verbose=True
        )
        
        # Setup logging
        logging.basicConfig(
            filename=os.path.join(save_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Print and log model structure
        model_summary = str(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        framework_versions = (
            f"PyTorch version: {torch.__version__}, "
            f"Torchvision version: {torchvision.__version__}"
        )
        
        logging.info("Model Structure:")
        logging.info(model_summary)
        logging.info(f"Total Parameters: {total_params}")
        logging.info(f"Trainable Parameters: {trainable_params}")
        logging.info(framework_versions)
        
        print("Model Structure:")
        print(model_summary)
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print(framework_versions)


    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{train_loss/total:.3f}', 
                            'acc': f'{100.*correct/total:.2f}%'})
        
        return train_loss/len(self.train_loader), 100.*correct/total

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return val_loss/len(self.val_loader), 100.*correct/total

    def train(self, num_epochs=200, val_freq=5):
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch + 1)
            
            log_msg = f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%'
            
            if (epoch + 1) % val_freq == 0:
                val_loss, val_acc = self.validate()
                log_msg += f', Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(epoch, val_acc, is_best=True)
                
                self.scheduler.step(val_acc)
            
            logging.info(log_msg)
            print(log_msg)
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_acc)

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc
        }
        
        if is_best:
            path = os.path.join(self.save_dir, 'model_best.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            
        torch.save(checkpoint, path)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = CustomDataset('Dataset/train')
    val_dataset = CustomDataset('Dataset/val')
    
    # Create dataloaders with a larger batch size for ResNet
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Initialize model - using ResNet18 with pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify the final fully connected layer for 4-class classification
    num_classes = 4  # Since you mentioned 4-class classification
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.train(num_epochs=200, val_freq=1)

if __name__ == '__main__':
    print("Starting model training...")
    try:
        main()
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")