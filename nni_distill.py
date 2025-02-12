import torch
from model_student import MobileNetV2Student
from mydataset import CustomDataset
from torch.utils.data import DataLoader
from nni.contrib.compression.pruning import L1NormPruner
from nni.contrib.compression.utils import auto_set_denpendency_group_ids
from nni.compression.pytorch.speedup.v2 import ModelSpeedup
from torchvision import models
import torch.nn as nn
from tqdm import tqdm
import logging

def setup_logger():
   logging.basicConfig(
       filename='epochs50_Prune_30/pruning.log',
       level=logging.INFO,
       format='%(asctime)s - %(message)s'
   )

def print_model_info(model, msg=""):
   info = f"\n{msg}\n{'-'*50}\n"
   info += str(model)
   info += f"\n{'-'*50}\n"
   logging.info(info)
   print(info)

def train(model, teacher_model, optimizer, train_loader, criterion, device, temperature=4.0, alpha=0.5):
   model.train()
   teacher_model.eval()  
   total_loss = 0
   correct = 0
   total = 0
   
   pbar = tqdm(train_loader, desc='Training')
   for images, labels in pbar:
       images, labels = images.to(device), labels.to(device)
       
       with torch.no_grad():
           teacher_logits = teacher_model(images)
           
       optimizer.zero_grad()
       student_logits = model(images)
       
      
       soft_targets = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
       soft_prob = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
       soft_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size(0)
       
      
       hard_loss = criterion(student_logits, labels)
       
      
       loss = (alpha * temperature * temperature * soft_loss + 
               (1 - alpha) * hard_loss)
               
       loss.backward()
       optimizer.step()
       total_loss += loss.item()
       
      
       _, predicted = student_logits.max(1)
       total += labels.size(0)
       correct += predicted.eq(labels).sum().item()
       acc = 100. * correct / total
       
       pbar.set_postfix({
           'loss': f'{total_loss/len(pbar):.4f}',
           'acc': f'{acc:.2f}%'
       })

def evaluate(model, val_loader, device):
   model.eval()
   correct = 0
   total = 0
   pbar = tqdm(val_loader, desc='Evaluating')
   with torch.no_grad():
       for images, labels in pbar:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           _, predicted = outputs.max(1)
           total += labels.size(0)
           correct += predicted.eq(labels).sum().item()
           acc = 100. * correct / total
           pbar.set_postfix({'acc': f'{acc:.2f}%'})
   return acc

def main():
   setup_logger()
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   

   train_dataset = CustomDataset('Dataset/train')
   val_dataset = CustomDataset('Dataset/val')
   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
   val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)


   model = MobileNetV2Student(num_classes=4) 
   checkpoint = torch.load('student_best.pth', map_location=device)
   model.load_state_dict(checkpoint['student_state_dict'])
   model = model.to(device)


   print_model_info(model, "Original Model Structure")
   original_params = sum(p.numel() for p in model.parameters())
   original_acc = evaluate(model, val_loader, device)
   logging.info(f'Original params: {original_params:,}')
   logging.info(f'Original acc: {original_acc:.2f}%')
   print(f'Original params: {original_params:,}')
   print(f'Original acc: {original_acc:.2f}%')


   config_list = [{
       'op_types': ['Conv2d'],
       'sparse_ratio': 0.3
   }]
   

   dummy_input = torch.rand(8, 3, 224, 224).to(device)
   config_list = auto_set_denpendency_group_ids(model, config_list, dummy_input)
   

   pruner = L1NormPruner(model, config_list)
   _, masks = pruner.compress()
   pruner.unwrap_model()
   

   model = ModelSpeedup(model, dummy_input, masks).speedup_model()
   

   print_model_info(model, "Pruned Model Structure")
   pruned_params = sum(p.numel() for p in model.parameters())
   pruned_acc = evaluate(model, val_loader, device)
   logging.info(f'Pruned params: {pruned_params:,}')
   logging.info(f'Pruned acc: {pruned_acc:.2f}%')
   print(f'Pruned params: {pruned_params:,}')
   print(f'Pruned acc: {pruned_acc:.2f}%')


   teacher_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
   teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 4)  
   teacher_checkpoint = torch.load('Teacher/Teacher_Resnet_checkpoint/model_best.pth', map_location=device)
   teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
   teacher_model = teacher_model.to(device)
   teacher_model.eval()


   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='max', patience=2, verbose=True
   )
   
   best_acc = pruned_acc
   print("\nStarting fine-tuning with knowledge distillation...")
   
   for epoch in range(50):
       logging.info(f'Epoch {epoch+1}/10')
       print(f'Epoch {epoch+1}/10')
       

       train(model, teacher_model, optimizer, train_loader, criterion, device,
             temperature=4.0, alpha=0.5)
       

       acc = evaluate(model, val_loader, device)
       logging.info(f'Validation acc: {acc:.2f}%')
       print(f'Validation acc: {acc:.2f}%')
       

       scheduler.step(acc)
       

       if acc > best_acc:
           best_acc = acc
           torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'accuracy': acc,
               'params': pruned_params
           }, 'epochs50_Prune_30/best_pruned_model.pth')
   

   best_checkpoint = torch.load('best_pruned_model.pth', map_location=device)
   model.load_state_dict(best_checkpoint['model_state_dict'])
   torch.save(model, 'final_pruned_model.pth')
   
   logging.info('Pruning and fine-tuning completed')
   print('Pruning and fine-tuning completed')
   print(f'Best validation accuracy: {best_acc:.2f}%')
   print(f'Parameters reduced from {original_params:,} to {pruned_params:,}')

if __name__ == '__main__':
   main()