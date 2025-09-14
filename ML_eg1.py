import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import ssl

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_CLASSES = 10  # CIFAR-10 also has 10 classes

# Data preprocessing and loading - CHANGED FOR CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 channels for RGB
])

# Download and load datasets - CHANGED TO CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    transform=transform,
    download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define the neural network architecture - NEEDS CHANGES FOR 3 CHANNELS
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten the image
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# Initialize model - CHANGED INPUT SIZE FOR CIFAR-10
input_size = 3 * 32 * 32  # CIFAR-10 images are 32x32 with 3 channels (RGB)
hidden_size = 512
model = NeuralNet(input_size, hidden_size, NUM_CLASSES).to(device)

# Loss and optimizer (same as before)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler (same as before)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training function (same as before)
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Validation function (same as before)
def validate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, all_preds, all_targets

# Training loop (same as before)
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print("Starting training...")
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    val_loss, val_acc, _, _ = validate_model(model, test_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    scheduler.step()
    
    print(f'Epoch [{epoch}/{NUM_EPOCHS}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Final evaluation
print("\nFinal Evaluation:")
final_val_loss, final_val_acc, all_preds, all_targets = validate_model(model, test_loader, criterion)
print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
print(f"Final Validation Loss: {final_val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'cifar10_model.pth')
print("Model saved as 'cifar10_model.pth'")

# Plot training history (same as before)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Confusion matrix - UPDATED CLASS NAMES
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('confusion_matrix.png')
plt.show()

# Classification report - UPDATED CLASS NAMES
print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=class_names))

# Test on some sample images - UPDATED FOR CIFAR-10
def test_sample_images(model, test_loader, num_samples=5):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    images = images.cpu().numpy()
    labels = labels.numpy()
    predicted = predicted.cpu().numpy()
    
    # Denormalize images for display
    images = images.transpose(0, 2, 3, 1)  # Change from (B, C, H, W) to (B, H, W, C)
    images = images * 0.5 + 0.5  # Unnormalize
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title(f'True: {class_names[labels[i]]}\nPred: {class_names[predicted[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

test_sample_images(model, test_loader)

# Load and test saved model
print("\nTesting loaded model...")
loaded_model = NeuralNet(input_size, hidden_size, NUM_CLASSES).to(device)
loaded_model.load_state_dict(torch.load('cifar10_model.pth'))
loaded_model.eval()

test_loss, test_acc, _, _ = validate_model(loaded_model, test_loader, criterion)
print(f"Loaded model test accuracy: {test_acc:.2f}%")

# Model summary
print("\nModel Architecture:")
print(model)

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Dataset info
print(f"\nNumber of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")