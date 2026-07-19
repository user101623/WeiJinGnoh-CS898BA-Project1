import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        
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
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])

# 5. Initialize Datasets and Loaders
train_ds = FishDataset(train_paths, train_y, transform=transform)
val_ds = FishDataset(val_paths, val_y, transform=transform)
test_ds = FishDataset(test_paths, test_y, transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)