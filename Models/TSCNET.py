import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EEGModel(nn.Module):
    def __init__(self, channels, num_classes, dropout_rate):
        super(EEGModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            # Temporal + Spatial 
            nn.Conv2d(1, 64, (channels, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 2)),
            nn.Dropout2d(dropout_rate * 0.4),
            
            nn.Conv2d(64, 128, (1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 2)),
            nn.Dropout2d(dropout_rate * 0.5),
            
            nn.Conv2d(128, 128, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 2)),
            nn.Dropout2d(dropout_rate * 0.6)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.ndim == 3:
            B, C, T = x.shape
           
            x = x.reshape(B, 1, C, T)
        
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x


