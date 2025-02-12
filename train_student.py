import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import torchvision
import logging
from tqdm import tqdm
import os
from mydataset import CustomDataset
from model_student import MobileNetV2Student

class DistillationTrainer:
    def __init__(self, teacher_model, student_model, train_loader, val_loader, 
                 device, temperature=4.0, alpha=0.5, save_dir='Student_checkpoint'):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.criterion_ce = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(
            self.student.parameters(), 
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, verbose=True)
        
        logging.basicConfig(
            filename=os.path.join(save_dir, 'distillation.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )


        model_summary = str(student_model)
        total_params = sum(p.numel() for p in student_model.parameters())
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
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
        
    def distillation_loss(self, student_logits, teacher_logits, labels):

        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size(0)
        

        hard_loss = self.criterion_ce(student_logits, labels)
        

        l1_norm = 0
        for param in self.student.parameters():
            l1_norm += torch.sum(torch.abs(param))
        

        lambda_l1 = 1e-5
        loss = (self.alpha * self.temperature * self.temperature * soft_loss + 
                (1 - self.alpha) * hard_loss + lambda_l1 * l1_norm)
        return loss, l1_norm.item()
        
    def train_epoch(self, epoch):
        self.teacher.eval()  
        self.student.train()
        train_loss = 0
        total_l1_norm = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            

            with torch.no_grad():
                teacher_logits = self.teacher(images)
                

            self.optimizer.zero_grad()
            student_logits = self.student(images)
            
            loss, l1_norm = self.distillation_loss(student_logits, teacher_logits, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            total_l1_norm += l1_norm
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{train_loss/total:.3f}', 
                            'acc': f'{100.*correct/total:.2f}%'})
            
        avg_l1_norm = total_l1_norm / len(self.train_loader)
        return train_loss/len(self.train_loader), 100.*correct/total, avg_l1_norm
        
    def validate(self):
        self.student.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.student(images)
                loss = self.criterion_ce(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return val_loss/len(self.val_loader), 100.*correct/total
        
    def train(self, num_epochs=200, val_freq=1):
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc, avg_l1_norm = self.train_epoch(epoch + 1)
            
            log_msg = f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Avg L1 Norm: {avg_l1_norm:.4f}'
            
            if (epoch + 1) % val_freq == 0:
                val_loss, val_acc = self.validate()
                log_msg += f', Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(epoch, val_acc, is_best=True)
                    
                self.scheduler.step(val_acc)
            
            logging.info(log_msg)
            print(log_msg)
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_acc)
                
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc
        }
        
        if is_best:
            path = os.path.join(self.save_dir, 'student_best.pth')
        else:
            path = os.path.join(self.save_dir, f'student_checkpoint_epoch_{epoch+1}.pth')
            
        torch.save(checkpoint, path)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    train_dataset = CustomDataset('Dataset/train')
    val_dataset = CustomDataset('Dataset/val')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    num_classes = 4 
    

    teacher = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
    

    teacher_checkpoint = torch.load('Teacher/Teacher_Resnet_checkpoint/model_best.pth', map_location=device)
    teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher.eval()  
    

    student = MobileNetV2Student(num_classes=num_classes)
    

    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        temperature=4.0,  
        alpha=0.5        
    )
    
    print("Starting knowledge distillation training...")
    trainer.train(num_epochs=200, val_freq=1)
    print("Knowledge distillation completed!")

if __name__ == '__main__':
    main()