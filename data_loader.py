"""Data loader for multi-hospital histopathology classification.

Clinical Center & Hospital Mapping:
- Hospital 0 (Source / Center 0 - RadboudUMC):
    * train: Supervised source training (supports 10,000, 20,000, and 30,000 balanced samples).
    * val: In-distribution validation from validation-00000 (5,284 patches).
    * test: Held-out in-distribution testing from train-00007 (10,902 patches).
- Hospital 1 (Target 1 / Center 1 - CWZ Nijmegen):
    * test / target: Out-of-distribution evaluation and test-time adaptation from validation-00002 (22,821 patches).
- Hospital 2 (Target 2 / Center 2 - UMC Utrecht):
    * test / target: Official Camelyon17-WILDS benchmark test domain from test-00000..00003 (85,054 patches).

Dataset: Camelyon17-WILDS H&E stained lymph node patches (224x224x3).
Classes: 0 = Normal tissue, 1 = Metastatic breast carcinoma (Tumor).
"""

from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
from typing import Tuple, List, Union, Optional
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class HistopathologyDataset(Dataset):
    """PyTorch Dataset for histopathology image patches.
    
    Supports both pre-stacked in-memory tensors (for small test sets <= 2,000 samples)
    and memory-efficient on-the-fly streaming (for large training sets up to 30,000+ samples).
    """

    def __init__(
        self,
        images: Union[torch.Tensor, List[bytes]],
        labels: torch.Tensor,
        hospital_id: int,
        transform=None
    ):
        self._is_tensor = isinstance(images, torch.Tensor)
        if self._is_tensor:
            self._images = images
            self._raw_bytes = None
        else:
            self._images = None
            self._raw_bytes = images

        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        self.hospital_id = hospital_id
        self.transform = transform

    @property
    def images(self) -> torch.Tensor:
        """Provides direct tensor access with caching if requested as tensor."""
        if self._images is None and self._raw_bytes is not None:
            self._images = torch.stack(_parallel_decode(self._raw_bytes, self.transform))
        return self._images

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, bytes], torch.Tensor]:
        if self._images is not None:
            return self._images[idx], self.labels[idx]
        return self._raw_bytes[idx], self.labels[idx]


def _parallel_decode(raw_list: List[bytes], transform, max_workers: int = 8) -> List[torch.Tensor]:
    """Decodes a list of raw byte streams in parallel using thread workers."""
    if not raw_list:
        return []
    workers = min(max_workers, os.cpu_count() or 4)

    def _task(raw):
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return transform(img)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_task, raw_list))


def parallel_collate_fn(batch):
    """Decodes a batch of image patches in parallel using thread workers."""
    if isinstance(batch[0][0], torch.Tensor):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return images, labels

    raw_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    workers = min(8, os.cpu_count() or 4)
    def _task(raw):
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return transform(img)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        tensors = list(executor.map(_task, raw_list))

    return torch.stack(tensors), labels


def load_hospital_data(
    data_dir: Union[str, Path],
    hospital: Union[int, str] = 0,
    split: str = "train",
    num_samples: int = 1000,
    balanced: bool = True
) -> HistopathologyDataset:
    """Loads image patches for a specific hospital partition from parquet archives.

    Args:
        data_dir: Path to directory containing Camelyon17 parquet files.
        hospital: 0 or 'radboud' for RadboudUMC (Center 0),
                  1 or 'cwz' for CWZ Nijmegen (Center 1),
                  2 or 'umcu' for UMC Utrecht (Center 2).
        split: 'train', 'val', or 'test'.
               For hospital=0 (or 1 in legacy code):
                 - 'train': Center 0 training patches (supports 10k, 20k, 30k).
                 - 'val': Center 0 validation patches (validation-00000).
                 - 'test': Center 0 held-out in-distribution test patches (train-00007).
               For hospital=1 (CWZ Nijmegen):
                 - 'test': Target domain 1 out-of-distribution patches.
               For hospital=2 (UMC Utrecht):
                 - 'test': Target domain 2 out-of-distribution patches.
        num_samples: Total number of samples to load (e.g. 10000, 20000, 30000).
        balanced: Whether to balance classes (equal normal and tumor).
    """
    data_path = Path(data_dir)
    split_lower = str(split).lower().strip()
    h_str = str(hospital).lower().strip()

    # Resolve target clinical center ID
    if h_str in ("0", "radboud", "h0", "center0"):
        target_center = 0
    elif h_str in ("1", "h1", "cwz", "center1", "cwz nijmegen"):
        target_center = 1
    elif h_str in ("2", "h2", "umcu", "center2", "umc utrecht"):
        target_center = 2
    elif h_str in ("3", "h3", "rijnstate", "center3", "rijnstate hospital"):
        target_center = 3
    elif h_str in ("4", "h4", "lpon", "center4", "lpon enschede"):
        target_center = 4
    else:
        target_center = int(hospital)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    per_class = num_samples // 2 if balanced else num_samples
    raw_normals: List[bytes] = []
    raw_tumors: List[bytes] = []

    # =========================================================================
    # Center 0 (RadboudUMC): Source Domain Training, Validation & In-Dist Test
    # =========================================================================
    if target_center == 0:
        if split_lower in ("train", "training"):
            # 1. Normals: read from train-00006 (has 19,444 Center 0 normals)
            p_norm = data_path / "train-00006-of-00014.parquet"
            if p_norm.exists():
                tab_norm = pq.read_table(p_norm, columns=["image", "label", "center"])
                c_arr = tab_norm["center"].to_numpy()
                l_arr = tab_norm["label"].to_numpy()
                idx_norm = np.where((c_arr == 0) & (l_arr == 0))[0][:per_class]
                img_col = tab_norm["image"]
                for i in idx_norm:
                    raw_normals.append(img_col[int(i)].as_py()["bytes"])

            # Fallback to train-00005 if per_class > 19,444
            if len(raw_normals) < per_class:
                p_norm_extra = data_path / "train-00005-of-00014.parquet"
                if p_norm_extra.exists():
                    tab_extra = pq.read_table(p_norm_extra, columns=["image", "label", "center"])
                    c_arr = tab_extra["center"].to_numpy()
                    l_arr = tab_extra["label"].to_numpy()
                    needed = per_class - len(raw_normals)
                    idx_extra = np.where((c_arr == 0) & (l_arr == 0))[0][:needed]
                    img_col = tab_extra["image"]
                    for i in idx_extra:
                        raw_normals.append(img_col[int(i)].as_py()["bytes"])

            # 2. Tumors: read from train-00000..00006 and train-00013 (provides over 20,500 Center 0 tumors)
            tumor_files = [
                "train-00000-of-00014.parquet",
                "train-00001-of-00014.parquet",
                "train-00002-of-00014.parquet",
                "train-00003-of-00014.parquet",
                "train-00004-of-00014.parquet",
                "train-00005-of-00014.parquet",
                "train-00006-of-00014.parquet",
                "train-00013-of-00014.parquet"
            ]
            for tf in tumor_files:
                if len(raw_tumors) >= per_class:
                    break
                p_file = data_path / tf
                if not p_file.exists():
                    continue
                tab_tum = pq.read_table(p_file, columns=["image", "label", "center"])
                c_arr = tab_tum["center"].to_numpy()
                l_arr = tab_tum["label"].to_numpy()
                idx_tum = np.where((c_arr == 0) & (l_arr == 1))[0]
                needed = per_class - len(raw_tumors)
                take_idx = idx_tum[:needed]
                img_col = tab_tum["image"]
                for i in take_idx:
                    raw_tumors.append(img_col[int(i)].as_py()["bytes"])

        elif split_lower in ("val", "validation"):
            # Center 0 Validation: from validation-00000 (5,284 Center 0 patches)
            tab = pq.read_table(data_path / "validation-00000-of-00003.parquet", columns=["image", "label", "center"])
            c_arr = tab["center"].to_numpy()
            l_arr = tab["label"].to_numpy()
            idx_norm = np.where((c_arr == 0) & (l_arr == 0))[0][:per_class]
            idx_tum = np.where((c_arr == 0) & (l_arr == 1))[0][:per_class]
            img_col = tab["image"]
            for i in idx_norm:
                raw_normals.append(img_col[int(i)].as_py()["bytes"])
            for i in idx_tum:
                raw_tumors.append(img_col[int(i)].as_py()["bytes"])

        elif split_lower in ("test", "testing", "in_dist", "indist"):
            # Center 0 Held-Out In-Distribution Test Set: from train-00007 (10,902 Center 0 patches)
            tab = pq.read_table(data_path / "train-00007-of-00014.parquet", columns=["image", "label", "center"])
            c_arr = tab["center"].to_numpy()
            l_arr = tab["label"].to_numpy()
            idx_norm = np.where((c_arr == 0) & (l_arr == 0))[0][:per_class]
            idx_tum = np.where((c_arr == 0) & (l_arr == 1))[0][:per_class]
            img_col = tab["image"]
            for i in idx_norm:
                raw_normals.append(img_col[int(i)].as_py()["bytes"])
            for i in idx_tum:
                raw_tumors.append(img_col[int(i)].as_py()["bytes"])

    # =========================================================================
    # Center 1 (CWZ Nijmegen): Target Hospital 1
    # =========================================================================
    elif target_center == 1:
        v2_path = data_path / "validation-00002-of-00003.parquet"
        if v2_path.exists():
            tab = pq.read_table(v2_path, columns=["image", "label"])
            l_arr = tab["label"].to_numpy()
            img_col = tab["image"]
            idx_norm = np.where(l_arr == 0)[0][:per_class]
            idx_tum = np.where(l_arr == 1)[0][:per_class]
            for i in idx_norm:
                raw_normals.append(img_col[int(i)].as_py()["bytes"])
            for i in idx_tum:
                raw_tumors.append(img_col[int(i)].as_py()["bytes"])

        # Fallback to validation-00001 if more normals are requested
        if len(raw_normals) < per_class and (data_path / "validation-00001-of-00003.parquet").exists():
            tab_v1 = pq.read_table(data_path / "validation-00001-of-00003.parquet", columns=["image", "label", "center"])
            c_arr = tab_v1["center"].to_numpy()
            l_arr = tab_v1["label"].to_numpy()
            needed = per_class - len(raw_normals)
            idx_extra = np.where((c_arr == 1) & (l_arr == 0))[0][:needed]
            img_col_v1 = tab_v1["image"]
            for i in idx_extra:
                raw_normals.append(img_col_v1[int(i)].as_py()["bytes"])

    # =========================================================================
    # Center 2 (UMC Utrecht): Target Hospital 2
    # =========================================================================
    elif target_center == 2:
        # Normals from test-00000 and test-00001
        for fn in ["test-00000-of-00004.parquet", "test-00001-of-00004.parquet"]:
            if len(raw_normals) >= per_class:
                break
            p_file = data_path / fn
            if not p_file.exists():
                continue
            tab = pq.read_table(p_file, columns=["image", "label"])
            l_arr = tab["label"].to_numpy()
            img_col = tab["image"]
            needed = per_class - len(raw_normals)
            idx_norm = np.where(l_arr == 0)[0][:needed]
            for i in idx_norm:
                raw_normals.append(img_col[int(i)].as_py()["bytes"])

        # Tumors from test-00002 and test-00003
        for fn in ["test-00002-of-00004.parquet", "test-00003-of-00004.parquet"]:
            if len(raw_tumors) >= per_class:
                break
            p_file = data_path / fn
            if not p_file.exists():
                continue
            tab = pq.read_table(p_file, columns=["image", "label"])
            l_arr = tab["label"].to_numpy()
            img_col = tab["image"]
            needed = per_class - len(raw_tumors)
            idx_tum = np.where(l_arr == 1)[0][:needed]
            for i in idx_tum:
                raw_tumors.append(img_col[int(i)].as_py()["bytes"])

    # =========================================================================
    # Center 3 (Rijnstate Hospital): Target Hospital 3
    # =========================================================================
    elif target_center == 3:
        # Normals from train-00008, train-00009, train-00010, train-00007
        for fn in ["train-00008-of-00014.parquet", "train-00009-of-00014.parquet", "train-00010-of-00014.parquet", "train-00007-of-00014.parquet"]:
            if len(raw_normals) >= per_class:
                break
            p_file = data_path / fn
            if not p_file.exists():
                continue
            tab = pq.read_table(p_file, columns=["image", "label", "center"])
            c_arr = tab["center"].to_numpy()
            l_arr = tab["label"].to_numpy()
            img_col = tab["image"]
            needed = per_class - len(raw_normals)
            idx_norm = np.where((c_arr == 3) & (l_arr == 0))[0][:needed]
            for i in idx_norm:
                raw_normals.append(img_col[int(i)].as_py()["bytes"])

        # Tumors from train-00000, train-00001, train-00002, train-00013
        for fn in ["train-00000-of-00014.parquet", "train-00001-of-00014.parquet", "train-00002-of-00014.parquet", "train-00013-of-00014.parquet"]:
            if len(raw_tumors) >= per_class:
                break
            p_file = data_path / fn
            if not p_file.exists():
                continue
            tab = pq.read_table(p_file, columns=["image", "label", "center"])
            c_arr = tab["center"].to_numpy()
            l_arr = tab["label"].to_numpy()
            img_col = tab["image"]
            needed = per_class - len(raw_tumors)
            idx_tum = np.where((c_arr == 3) & (l_arr == 1))[0][:needed]
            for i in idx_tum:
                raw_tumors.append(img_col[int(i)].as_py()["bytes"])

    # =========================================================================
    # Center 4 (LPON Enschede): Target Hospital 4
    # =========================================================================
    elif target_center == 4:
        # Normals from train-00011, train-00012, train-00010, train-00013
        for fn in ["train-00011-of-00014.parquet", "train-00012-of-00014.parquet", "train-00010-of-00014.parquet", "train-00013-of-00014.parquet"]:
            if len(raw_normals) >= per_class:
                break
            p_file = data_path / fn
            if not p_file.exists():
                continue
            tab = pq.read_table(p_file, columns=["image", "label", "center"])
            c_arr = tab["center"].to_numpy()
            l_arr = tab["label"].to_numpy()
            img_col = tab["image"]
            needed = per_class - len(raw_normals)
            idx_norm = np.where((c_arr == 4) & (l_arr == 0))[0][:needed]
            for i in idx_norm:
                raw_normals.append(img_col[int(i)].as_py()["bytes"])

        # Tumors from train-00003, train-00004, train-00005, train-00002
        for fn in ["train-00003-of-00014.parquet", "train-00004-of-00014.parquet", "train-00005-of-00014.parquet", "train-00002-of-00014.parquet"]:
            if len(raw_tumors) >= per_class:
                break
            p_file = data_path / fn
            if not p_file.exists():
                continue
            tab = pq.read_table(p_file, columns=["image", "label", "center"])
            c_arr = tab["center"].to_numpy()
            l_arr = tab["label"].to_numpy()
            img_col = tab["image"]
            needed = per_class - len(raw_tumors)
            idx_tum = np.where((c_arr == 4) & (l_arr == 1))[0][:needed]
            for i in idx_tum:
                raw_tumors.append(img_col[int(i)].as_py()["bytes"])

    total_loaded = len(raw_normals) + len(raw_tumors)
    all_raw = raw_normals + raw_tumors
    all_labels = [0] * len(raw_normals) + [1] * len(raw_tumors)

    # Shuffle instances
    perm = np.random.RandomState(42).permutation(total_loaded)
    shuffled_raw = [all_raw[i] for i in perm]
    shuffled_labels = torch.tensor([all_labels[i] for i in perm], dtype=torch.long)

    # For datasets <= 2000 samples (e.g. test, validation sets), decode upfront for fast direct indexing
    if total_loaded <= 2000:
        decoded_tensors = _parallel_decode(shuffled_raw, transform)
        images_tensor = torch.stack(decoded_tensors)
        return HistopathologyDataset(images_tensor, shuffled_labels, hospital_id=target_center, transform=transform)
    else:
        # For large training sets (10k, 20k, 30k), use memory-efficient streaming to prevent RAM overflow
        return HistopathologyDataset(shuffled_raw, shuffled_labels, hospital_id=target_center, transform=transform)


def get_dataloaders(
    data_dir: Union[str, Path] = r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data\data",
    batch_size: int = 1000,
    micro_batch_size: int = 100,
    train_samples: int = 10000,
    val_samples: int = 1000,
    test_samples: int = 1000,
    target_hospital: Union[int, str] = 1,
    include_train: bool = True,
    include_val: bool = True,
    include_test: bool = True
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], int]:
    """Prepares clean DataLoaders for Source Hospital 0 (Train/Val) and Target Hospital (Test).

    Args:
        data_dir: Directory containing parquet files.
        batch_size: Effective large batch size for gradient updates (e.g. 1000).
        micro_batch_size: Sub-batch size loaded per step to prevent memory issues.
        train_samples: Number of training samples from Hospital 0 (Center 0, e.g. 10000, 20000, 30000).
        val_samples: Number of validation samples from Hospital 0 (Center 0).
        test_samples: Number of evaluation samples from target hospital.
        target_hospital: Target hospital for adaptation (1 for CWZ Nijmegen, 2 for UMC Utrecht).

    Returns:
        (train_loader, val_loader, test_loader, accumulation_steps)
    """
    actual_micro = min(micro_batch_size, batch_size)
    accumulation_steps = max(1, batch_size // actual_micro)

    train_loader = None
    val_loader = None
    test_loader = None

    if include_train:
        print(f"Loading Hospital 0 (Center 0 - RadboudUMC) Training Set ({train_samples} samples)...")
        train_dataset = load_hospital_data(data_dir, hospital=0, split="train", num_samples=train_samples)
        train_loader = DataLoader(train_dataset, batch_size=actual_micro, shuffle=True, collate_fn=parallel_collate_fn)

    if include_val:
        print(f"Loading Hospital 0 (Center 0 - RadboudUMC) Validation Set ({val_samples} samples)...")
        val_dataset = load_hospital_data(data_dir, hospital=0, split="val", num_samples=val_samples)
        val_loader = DataLoader(val_dataset, batch_size=actual_micro, shuffle=False, collate_fn=parallel_collate_fn)

    if include_test:
        t_str = str(target_hospital).lower().strip()
        if t_str in ("1", "cwz", "center1", "h1"):
            target_center_code = 1
            target_name = "CWZ Nijmegen (Center 1)"
        elif t_str in ("2", "umcu", "center2", "h2"):
            target_center_code = 2
            target_name = "UMC Utrecht (Center 2)"
        elif t_str in ("3", "rijnstate", "center3", "h3"):
            target_center_code = 3
            target_name = "Rijnstate Hospital (Center 3)"
        elif t_str in ("4", "lpon", "center4", "h4"):
            target_center_code = 4
            target_name = "LPON Enschede (Center 4)"
        else:
            target_center_code = int(target_hospital)
            target_name = f"Hospital {target_center_code}"
        print(f"Loading Target Hospital ({target_name}) Test Set ({test_samples} samples)...")
        test_dataset = load_hospital_data(data_dir, hospital=target_center_code, split="test", num_samples=test_samples)
        test_loader = DataLoader(test_dataset, batch_size=actual_micro, shuffle=False, collate_fn=parallel_collate_fn)

    return train_loader, val_loader, test_loader, accumulation_steps
