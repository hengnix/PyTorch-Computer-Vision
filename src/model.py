import torch.nn as nn
import torchvision.models as models


class FruitVegClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FruitVegClassifier, self).__init__()

        # 自动检测 torchvision 版本并选择合适的权重加载方式
        try:
            # 新版本 (torchvision >= 0.13)
            from torchvision.models.resnet import ResNet18_Weights

            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V2)
        except (ImportError, AttributeError):
            # 旧版本
            self.model = models.resnet18(pretrained=True)

        # 修改分类层并添加 Dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
