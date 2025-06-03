import wandb
from typing import List, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src import parse_dataset_structure
from src import trainer, schedulers
from src import models
from src import summarizer


DATASET_PATH = "./dataset"
VALIDATION_RATIO = 0.2
RANDOM_SEED = 42

EPS = 1e-8
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

WEB_LOGGING = True
PROJECT_NAME = "EX-classification"
MODEL_NAME = "resnet_18"
RUN_NAME = f"model_{MODEL_NAME}-LR_{LR}"
LOG_CONFIG = {"LR": LR, "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "BACKBONE": MODEL_NAME.split("_")[0]}


class ExDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int]], label_to_idx: dict, img_transform=None):
        self.data = data
        self.label_to_idx = label_to_idx
        self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        self.img_transform = img_transform

        counts = Counter([label for _, label in self.data])
        for label_idx, count in sorted(counts.items()):
            label_name = self.idx_to_label[label_idx]
            print(f"\t{label_name:<10}: {count}개")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i]
        image = Image.open(path).convert("RGB")
        if self.img_transform:
            image = self.img_transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class ExTrainer(trainer.Trainer):
    def __init__(self, model, device, mode: str = "train", optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(ExTrainer, self).__init__(model, device, mode, optimizer, scheduler)

    def train_step(self, batch):
        x, y = batch
        proba = self.model(x).softmax(1)
        loss = F.cross_entropy(proba, y)

        n_cls = proba.shape[1]
        p = proba.argmax(1)  # prediction

        acc = (p == y).float().mean().item()

        p_hot = F.one_hot(p, num_classes=n_cls).float()
        y_hot = F.one_hot(y, num_classes=n_cls).float()

        tp = (p_hot * y_hot).sum(dim=0)
        fp = (p_hot * (1 - y_hot)).sum(dim=0)
        fn = ((1 - p_hot) * y_hot).sum(dim=0)

        precision = (tp / (tp + fp + EPS)).mean().item()
        recall = (tp / (tp + fn + EPS)).mean().item()
        f1 = (2 * precision * recall / (precision + recall + EPS))

        log_dict = {"loss": loss, "output": p, "acc": acc, "precision": precision, "recall": recall, "f1": f1}
        if WEB_LOGGING:
            prefix = "train" if self.mode == "train" else "valid" if self.mode == "test" else "inference"
            wandb.log({f"{prefix}_{k}": v for k, v in log_dict.items()})
        return log_dict

    def test_step(self, batch):
        return self.train_step(batch)


def main():
    # Parse data with labels
    parsed_train_data, parsed_test_data, label_map = parse_dataset_structure(DATASET_PATH)

    # Split Train & Validation data (Stratified)
    train_data_paths = [dp for dp, _ in parsed_train_data]
    label_indices = [label_map[label] for _, label in parsed_train_data]

    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_data_paths,
        label_indices,
        test_size=VALIDATION_RATIO,
        random_state=RANDOM_SEED,
        stratify=label_indices,
    )

    train_data = list(zip(train_paths, train_labels))
    valid_data = list(zip(valid_paths, valid_labels))
    test_data = [(dp, label_map[label]) for dp, label in parsed_test_data]

    # Preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Define Dataset
    print(f"Train samples: {len(train_data)}")
    train_dataset = ExDataset(train_data, label_map, img_transform=transform)
    print(f"Valid samples: {len(valid_data)}")
    valid_dataset = ExDataset(valid_data, label_map, img_transform=transform)
    print(f"Test samples: {len(test_data)}")
    test_dataset = ExDataset(test_data, label_map, img_transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initial WebLogger
    if WEB_LOGGING:
        wandb.init(project=PROJECT_NAME, name=RUN_NAME, config=LOG_CONFIG)
    else:
        import matplotlib
        matplotlib.use('TkAgg')

    # Define model & trainer
    model = models.get_gray_resnet(num_layer=18, num_classes=max(label_indices) + 1)
    ex_optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    ex_scheduler = schedulers.SchedulerChain(
        schedulers.WarmUpScheduler(ex_optimizer, total_iters=(warm_epoch := EPOCHS//10)),
        schedulers.CosineAnnealingScheduler(ex_optimizer, total_iters=EPOCHS-warm_epoch),
    )
    ex_trainer = ExTrainer(model, device=DEVICE, optimizer=ex_optimizer, scheduler=ex_scheduler)

    # Training & Validation
    ex_trainer.run(n_epoch=EPOCHS, loader=train_loader, valid_loader=valid_loader)

    # Inference
    ex_trainer.toggle("inference")
    test_result = torch.cat(ex_trainer.inference(loader=test_loader), dim=0)
    y_true = [label for _, label in test_dataset]
    y_pred = test_result.tolist()

    print("Summary")
    summary = summarizer.Summary(y_true=y_true, y_pred=y_pred, label_map=train_dataset.idx_to_label)
    summary.summary(log_to_wandb=WEB_LOGGING)

    if WEB_LOGGING:
        wandb.finish()
    print("Done")


if __name__ == "__main__":
    main()
