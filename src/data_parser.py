import os
import glob
from typing import List, Tuple, Dict


def parse_dataset_structure(
    data_root: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, int]]:

    def collect_paths(subdir: str) -> List[Tuple[str, str]]:
        pattern = os.path.join(data_root, subdir, "*", "*")
        filepaths = glob.glob(pattern)
        return [(path, os.path.basename(os.path.dirname(path))) for path in filepaths]

    train_data = collect_paths("train")
    test_data = collect_paths("test")

    all_labels = sorted({label for _, label in train_data + test_data})
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    return train_data, test_data, label_to_idx
