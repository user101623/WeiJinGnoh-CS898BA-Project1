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

import gc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

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
image_paths = list(data_dir.glob('*/*.jpg'))
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
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
LEARNING_RATE = 0.001
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

# PART 4: Hyperparameter Optimization using Grid Search
search_learning_rates = [0.01, 0.001, 0.0001]
search_batch_sizes = [32, 64]
search_dropout_rates = [0.3, 0.5]

best_val_loss = float('inf')
best_config = {}
best_weights = None

print("Starting Custom Grid Search Optimization...")

# 2. Iterate through all possible combinations of hyperparameters
for lr in search_learning_rates:
    for bs in search_batch_sizes:
        for dropout in search_dropout_rates:
            print(f"\n[Evaluating Config] LR: {lr} | Batch Size: {bs} | Dropout: {dropout}")
            
            # Re-initialize dataloaders for the current batch size
            current_train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
            current_val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
            
            # Re-initialize model with the current dropout rate
            tuning_model = BaselineCNN(
                num_classes=len(unique_labels), 
                img_size=224, 
                dropout_rate=dropout
            ).to(device)
            
            tuning_criterion = nn.CrossEntropyLoss()
            tuning_optimizer = optim.Adam(tuning_model.parameters(), lr=lr)
            
            # Train for a designated tuning epoch count (e.g., 5-8 epochs for fast evaluation)
            tuning_epochs = 6
            trial_best_loss = float('inf')
            
            for epoch in range(1, tuning_epochs + 1):
                # Train phase
                tuning_model.train()
                for imgs, labels in current_train_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    tuning_optimizer.zero_grad(set_to_none=True)
                    loss = tuning_criterion(tuning_model(imgs), labels)
                    loss.backward()
                    tuning_optimizer.step()
                
                # Validation phase
                tuning_model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for imgs, labels in current_val_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        preds = tuning_model(imgs)
                        val_loss_sum += tuning_criterion(preds, labels).item() * labels.size(0)
                
                epoch_val_loss = val_loss_sum / len(current_val_loader.dataset)
                if epoch_val_loss < trial_best_loss:
                    trial_best_loss = epoch_val_loss
            
            print(f"-> Result: Best Validation Loss for this configuration = {trial_best_loss:.4f}")
            
            # Track the global best configuration
            if trial_best_loss < best_val_loss:
                best_val_loss = trial_best_loss
                best_config = {"lr": lr, "batch_size": bs, "dropout": dropout}
                best_weights = tuning_model.state_dict()
                
            del tuning_model, tuning_optimizer, tuning_criterion, current_train_loader, current_val_loader
            gc.collect()
            torch.cuda.empty_cache()

# 3. Save the final optimized model weights
OPTIMIZED_WEIGHTS_PATH = Path("optimized_model.pth")
torch.save({
    "model_state_dict": best_weights,
    "hyperparameters": best_config,
    "validation_loss": best_val_loss
}, OPTIMIZED_WEIGHTS_PATH)

print("\nGRID SEARCH COMPLETED SUCCESSFULLY!")
print(f"Best Configuration Found: {best_config}")
print(f"Lowest Validation Loss: {best_val_loss}")
print(f"Optimized weights saved to: {OPTIMIZED_WEIGHTS_PATH.name}")
print("==============================================")

# PART 5: Evaluation and Analysis
baseline_model = BaselineCNN(num_classes=len(unique_labels), img_size=224).to(device)
baseline_model.load_state_dict(torch.load("baseline_model.pth", map_location=device))
baseline_model.eval()

optimized_checkpoint = torch.load("optimized_model.pth", map_location=device)
optimized_state_dict = optimized_checkpoint.get("model_state_dict", optimized_checkpoint)

optimized_model = BaselineCNN(num_classes=len(unique_labels), img_size=224, dropout_rate=0.5).to(device)
optimized_model.load_state_dict(optimized_state_dict)
optimized_model.eval()

def evaluate_model_predictions(loader, model, device=device):
    model.eval()
    true_targets = []
    predicted_targets = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            true_targets.append(targets.cpu())
            predicted_targets.append(outputs.argmax(dim=1).cpu())
            
    return torch.cat(true_targets).numpy(), torch.cat(predicted_targets).numpy()

# 2. Generate predictions for both models
baseline_y_true, baseline_y_pred = evaluate_model_predictions(test_loader, baseline_model)
optimized_y_true, optimized_y_pred = evaluate_model_predictions(test_loader, optimized_model)

# Verify label consistency
if not np.array_equal(baseline_y_true, optimized_y_true):
    raise ValueError("Test set labels differ between model evaluation runs.")

# 3. Print quantitative classification metrics
print("\n--- Baseline Model Test Performance ---")
print(classification_report(baseline_y_true, baseline_y_pred, target_names=unique_labels, digits=4, zero_division=0))
print(f"Baseline Accuracy: {accuracy_score(baseline_y_true, baseline_y_pred):.2%}")

print("\n--- Optimized Model Test Performance ---")
print(classification_report(optimized_y_true, optimized_y_pred, target_names=unique_labels, digits=4, zero_division=0))
print(f"Optimized Accuracy: {accuracy_score(optimized_y_true, optimized_y_pred):.2%}")

# 4. Create Custom Multi-Panel Visualization Summary
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

epochs = range(1, len(train_losses) + 1)
axes[0].plot(epochs, train_losses, label="Train Loss", linestyle="--")
axes[0].plot(epochs, val_losses, label="Validation Loss")
axes[0].set_title("Training vs Validation Loss")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, train_accs, label="Train Accuracy", linestyle="--")
axes[1].plot(epochs, val_accs, label="Validation Accuracy")
axes[1].set_title("Training vs Validation Accuracy")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

conf_matrix = confusion_matrix(optimized_y_true, optimized_y_pred)
matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=unique_labels)
matrix_display.plot(ax=axes[2], cmap="Blues", colorbar=False, values_format="d")
axes[2].set_title("Optimized Model Confusion Matrix")
axes[2].tick_params(axis="x", labelrotation=45)

plt.tight_layout()
final_evaluation_plot = Path("evaluation_summary.png")
plt.savefig(final_evaluation_plot, dpi=200)
plt.show()

print(f"\nEvaluation plot successfully generated and saved to: {final_evaluation_plot}")