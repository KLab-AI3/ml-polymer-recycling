# models/resnet18_vision.py
# 1D ResNet-18 style model for spectra: input (B, 1, L)
import torch
import torch.nn as nn
from typing import Callable, List

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def _make_layer(block: Callable[..., nn.Module], in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
    downsample = None
    if stride != 1 or in_planes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv1d(in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes * block.expansion),
        )
    layers: List[nn.Module] = [block(in_planes, planes, stride, downsample)]
    in_planes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(in_planes, planes))
    return nn.Sequential(*layers)

class ResNet18Vision(nn.Module):
    def __init__(self, input_length: int = 500, num_classes: int = 2):
        super().__init__()
        # 1D stem
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet-18: 2 blocks per layer
        self.layer1 = _make_layer(BasicBlock1D, 64,  64, blocks=2, stride=1)
        self.layer2 = _make_layer(BasicBlock1D, 64, 128, blocks=2, stride=2)
        self.layer3 = _make_layer(BasicBlock1D, 128, 256, blocks=2, stride=2)
        self.layer4 = _make_layer(BasicBlock1D, 256, 512, blocks=2, stride=2)

        # Global pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)     # (B, C, 1)
        x = torch.flatten(x, 1) # (B, C)
        x = self.fc(x)          # (B, num_classes)
        return x
