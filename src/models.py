import timm
import torch
import torch.nn as nn
from torchvision.models import *


def get_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Model Getter
    """
    if model_name.startswith("resnet_"):
        layer_num = int(model_name.split("_")[-1])
        return get_gray_resnet(layer_num, num_classes)
    elif model_name.startswith("efficientnet_b"):
        version = model_name.split("_")[-1]
        return get_gray_efficientnet(version, num_classes)
    elif model_name.startswith("swin_"):
        return get_gray_swin(model_name, num_classes)
    elif model_name.startswith("coatnet_"):
        return get_gray_coatnet(model_name, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def get_gray_resnet(num_layer: int, num_classes: int) -> nn.Module:
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


def get_gray_efficientnet(version: str, num_classes: int) -> nn.Module:
    version_map = {
        "b0": (efficientnet_b0, EfficientNet_B0_Weights),
        "b1": (efficientnet_b1, EfficientNet_B1_Weights),
        "b2": (efficientnet_b2, EfficientNet_B2_Weights),
        "b3": (efficientnet_b3, EfficientNet_B3_Weights),
        "b4": (efficientnet_b4, EfficientNet_B4_Weights),
        "b5": (efficientnet_b5, EfficientNet_B5_Weights),
        "b6": (efficientnet_b6, EfficientNet_B6_Weights),
        "b7": (efficientnet_b7, EfficientNet_B7_Weights),
    }

    if version not in version_map:
        raise ValueError(f"Unsupported EfficientNet version: {version}")

    builder_fn, weight_enum = version_map[version]
    model = builder_fn(weights=weight_enum.DEFAULT)

    pretrained_conv1 = model.features[0][0]
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

    model.features[0][0] = new_conv1
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


def get_gray_swin(model_name: str, num_classes: int) -> nn.Module:
    model = swin_t(weights=Swin_T_Weights.DEFAULT)

    pretrained_conv = model.features[0][0]  # Conv2d(3, 96, kernel_size=4, stride=4)
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=pretrained_conv.out_channels,
        kernel_size=pretrained_conv.kernel_size,
        stride=pretrained_conv.stride,
        padding=pretrained_conv.padding,
        bias=False
    )

    with torch.no_grad():
        new_conv.weight = nn.Parameter(pretrained_conv.weight.mean(dim=1, keepdim=True))

    model.features[0][0] = new_conv
    model.head = nn.Linear(model.head.in_features, num_classes)

    return model


def get_gray_coatnet(version: str, num_classes: int) -> nn.Module:
    if version + "_rw_224" not in timm.list_models("*coatnet*"):
        raise ValueError(f"Unsupported CoAtNet version: {version}")

    # Load pretrained model
    model = timm.create_model(version, pretrained=True)

    # 입력 채널 수정 (1채널 grayscale → 3채널 mean 복사)
    conv_stem = model.conv_stem
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv_stem.out_channels,
        kernel_size=conv_stem.kernel_size,
        stride=conv_stem.stride,
        padding=conv_stem.padding,
        bias=False
    )
    with torch.no_grad():
        new_conv.weight = nn.Parameter(conv_stem.weight.mean(dim=1, keepdim=True))
    model.conv_stem = new_conv

    # 출력 클래스 수 수정
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    return model
