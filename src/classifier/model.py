import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy classifier using MobileNetV2 with optional metadata late-fusion.

    When use_metadata=True, the pooled image features are concatenated with a 32-dim
    metadata embedding before the classification head. The metadata vector is 7-dim;
    there is no separate 'has_metadata' scalar (that would leak modality).
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        use_metadata: bool = False,
        metadata_dim: int = 7,
    ):
        super().__init__()
        self.use_metadata = use_metadata

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        if use_metadata:
            self.meta_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 32),
                nn.ReLU(),
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
            nn.Linear(128, num_classes),
        )

        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        if self.meta_encoder is not None:
            for module in self.meta_encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        features = self.backbone(x)
        if self.use_metadata and metadata is not None:
            meta_feats = self.meta_encoder(metadata)
            features = torch.cat([features, meta_feats], dim=1)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)