import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=6, img_size=224, dropout_rate=0.3):
        super().__init__()
        
        # 1. Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 2. Calculating Dynamic Classifier Input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            flat_size = self.features(dummy_input).flatten().shape[0]
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))