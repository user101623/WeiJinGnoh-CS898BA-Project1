import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms

from baselineCNN import BaselineCNN
from torch import nn, optim
from matplotlib import pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# PART 2: Data Preprocesssing and Augmentation
class FishDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
            
        return img, self.labels[idx]

# 2. Prepare File Paths and Labels
data_dir = Path('fish_dataset')
image_paths = list(data_dir.glob('*/*.png'))
labels = [p.parent.name for p in image_paths]

# Create label mapping
unique_labels = sorted(list(set(labels)))
label_to_idx = {l: i for i, l in enumerate(unique_labels)}
y = [label_to_idx[l] for l in labels]

# 3. Stratified Splits
train_paths, temp_paths, train_y, temp_y = train_test_split(
    image_paths, y, test_size=0.3, stratify=y, random_state=SEED
)
val_paths, test_paths, val_y, test_y = train_test_split(
    temp_paths, temp_y, test_size=0.5, stratify=temp_y, random_state=SEED
)

# 4. Define Transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 5. Initialize Datasets and Loaders
train_ds = FishDataset(train_paths, train_y, transform=train_transform)
val_ds = FishDataset(val_paths, val_y, transform=val_test_transform)
test_ds = FishDataset(test_paths, test_y, transform=val_test_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# PART 3.2: Initial training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# Initialize Model, Loss, and Optimizer
model = BaselineCNN(num_classes=len(unique_labels), img_size=224).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
BASELINE_WEIGHTS_PATH = Path("baseline_model.pth")
PLOT_PATH = Path("baseline_curves.png")

# 2. Epoch Function
def run_epoch(loader, model, optimizer, training=True):
    model.train() if training else model.eval()
    total_loss, total_correct = 0.0, 0
    
    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            if training:
                optimizer.zero_grad(set_to_none=True)
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            if training:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

# 3. Training Loop
train_losses, train_accs, val_losses, val_accs = [], [], [], []
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = run_epoch(train_loader, model, optimizer, training=True)
    val_loss, val_acc = run_epoch(val_loader, model, optimizer, training=False)
    
    train_losses.append(train_loss); train_accs.append(train_acc)
    val_losses.append(val_loss); val_accs.append(val_acc)
    
    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), BASELINE_WEIGHTS_PATH)
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss}, Val Loss: {val_loss}, Train Acc: {train_acc}, Val Acc: {val_acc}")

# 4. Plotting
def save_performance_plot(train_losses, val_losses, train_accs, val_accs, save_path="training_summary.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    epochs_range = range(1, len(train_losses) + 1)

    # Loss Plot
    axes[0].plot(epochs_range, train_losses, 'o-', label="Train Loss")
    axes[0].plot(epochs_range, val_losses, 's-', label="Val Loss")
    axes[0].set_title("Learning Progress (Loss)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Accuracy Plot
    axes[1].plot(epochs_range, train_accs, 'o-', label="Train Acc")
    axes[1].plot(epochs_range, val_accs, 's-', label="Val Acc")
    axes[1].set_title("Learning Progress (Accuracy)")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=200)
    print(f"Summary plot saved as: {save_path}")

save_performance_plot(train_losses, val_losses, train_accs, val_accs)