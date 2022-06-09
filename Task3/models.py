import torch.nn as nn
import torchvision.models as models


class M(nn.Module):
    def __init__(
        self,
        backbone_name,
        num_classes,
    ):
        super(M, self).__init__()
        if backbone_name == "vgg":
            self.backbone = models.vgg16(pretrained=True)
        elif backbone_name == "resnet":
            self.backbone = models.resnet18(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        logits = self.classifier(x)
        return logits
