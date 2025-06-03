from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


class Summary:
    def __init__(self, y_true, y_pred, label_map: Dict[int, str]):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.label_map = label_map
        self.labels = list(label_map.keys())
        self.label_names = [label_map[i] for i in self.labels]

    def print_metrics(self, log_to_wandb=False):
        acc = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)

        print("\n📊 [Test Evaluation Metrics]")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f1:.4f}")

        if log_to_wandb:
            import wandb
            wandb.log({"test_accuracy": acc, "test_precision": precision, "test_recall": recall, "test_f1": f1})

    def show_confusion_matrix(self, log_to_wandb: bool = False):
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=self.label_names, yticklabels=self.label_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        if log_to_wandb:
            import wandb
            wandb.log({"confusion_matrix": wandb.Image(plt.gcf())})
        else:
            plt.show()

    def summary(self, log_to_wandb: bool = False):
        self.print_metrics(log_to_wandb=log_to_wandb)
        self.show_confusion_matrix(log_to_wandb=log_to_wandb)
