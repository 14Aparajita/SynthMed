import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy classifier using MobileNetV2.
    Lightweight and suitable for laptop execution.
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
        else:
            weights = None
        
        self.backbone = mobilenet_v2(weights=weights)
        
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize new layers
        for module in self.backbone.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features