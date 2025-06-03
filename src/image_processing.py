from torchvision import transforms

# 각 모델별 expected input size 정의
MODEL_INPUT_SIZE_MAP = {
    "resnet": (224, 224),
    "efficientnet_b0": (224, 224),
    "efficientnet_b1": (240, 240),
    "efficientnet_b2": (260, 260),
    "efficientnet_b3": (300, 300),
    "efficientnet_b4": (380, 380),
    "efficientnet_b5": (456, 456),
    "efficientnet_b6": (528, 528),
    "efficientnet_b7": (600, 600),
}


def get_image_transform(model_name: str, with_augmentation: bool = False):
    model_key = model_name.lower()

    # 모델 입력 크기 가져오기
    for key in MODEL_INPUT_SIZE_MAP:
        if key in model_key:
            input_size = MODEL_INPUT_SIZE_MAP[key]
            break
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # transform 구성
    if with_augmentation:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=input_size, scale=(0.8, 1.0)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
