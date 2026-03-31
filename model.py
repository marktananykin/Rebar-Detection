import torch
import torch.nn as nn
import torchvision.models as models

class RebarDetectionModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(RebarDetectionModel, self).__init__()

        # Use pretrained ResNet backbone for better generalized feature extraction
        self.backbone = models.resnet18(pretrained=pretrained)

        # Replace the final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
