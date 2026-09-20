import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from typing import Optional  # <-- Add this import


class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy classifier using MobileNetV2.
    Can optionally fuse metadata features.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        use_metadata: bool = False,
        metadata_dim: int = 7
    ):
        super().__init__()
        self.use_metadata = use_metadata

        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
        else:
            weights = None
        self.backbone = mobilenet_v2(weights=weights)

        # Extract number of features from original classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier with identity to get features
        self.backbone.classifier = nn.Identity()

        # Build new head
        if use_metadata:
            self.meta_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 32),
                nn.ReLU()
            )
            fusion_dim = in_features + 32
        else:
            self.meta_encoder = None
            fusion_dim = in_features

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Initialize new head
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.meta_encoder is not None:
            for module in self.meta_encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.backbone(x)  # now returns pooled features
        if self.use_metadata and metadata is not None:
            meta_feats = self.meta_encoder(metadata)
            features = torch.cat([features, meta_feats], dim=1)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        return self.backbone(x)