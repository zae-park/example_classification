import torch
import torch.nn as nn
from torchvision.models import *


def get_gray_resnet(num_layer: int, num_classes: int) -> nn.Module:
    # 1. 모델 선택
    if num_layer == 18:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif num_layer == 34:
        model = resnet34(weights=ResNet34_Weights.DEFAULT)
    elif num_layer == 50:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif num_layer == 101:
        model = resnet101(weights=ResNet101_Weights.DEFAULT)
    elif num_layer == 152:
        model = resnet152(weights=ResNet152_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported ResNet layer: {num_layer}")

    # 2. Conv1 수정
    pretrained_conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=pretrained_conv1.out_channels,
        kernel_size=pretrained_conv1.kernel_size,
        stride=pretrained_conv1.stride,
        padding=pretrained_conv1.padding,
        bias=False
    )

    with torch.no_grad():
        new_conv1.weight = nn.Parameter(pretrained_conv1.weight.mean(dim=1, keepdim=True))

    model.conv1 = new_conv1
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
