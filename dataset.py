"""
Ad Image Binary Classifier - Dataset
======================================
Handles data loading from CSV files and augmentation.
Each split (train/val/test) is loaded from its own CSV file.
"""

import csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.v2 as T
from omegaconf import DictConfig


# ============================================================
# Dataset
# ============================================================

class AdImageDataset(Dataset):
    """Advertisement image dataset for binary classification.
    
    Args:
        samples: List of (image_path, label) tuples.
        transform: Image transform pipeline.
    """
    
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform=None,
    ):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, text_info = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)

        return image, label, img_path, text_info
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Return class counts."""
        dist = {}
        for _, label, *_ in self.samples:
            dist[label] = dist.get(label, 0) + 1
        return dist


# ============================================================
# CSV Loading
# ============================================================

def load_samples_from_csv(csv_path: str) -> List[Tuple[str, int]]:
    """Load samples from a CSV file.
    
    Expected CSV columns:
        - filepath: path to the image file
        - label:    integer class label (0 or 1)
    
    Example:
        filepath,label
        /data/images/img001.jpg,True
        /data/images/img002.jpg,False
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    samples = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # Validate required columns
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty: {csv_path}")
        
        required_cols = {"local_image_path", "label"}
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV missing required columns {missing}. "
                f"Found columns: {reader.fieldnames}. "
                f"Expected at least: local_image_path, label"
            )
        
        for row_idx, row in enumerate(reader):
            filepath = row["local_image_path"].strip()
            try:
                label = 1 if row["label"].strip() == "True" else 0
            except ValueError:
                raise ValueError(
                    f"Invalid label at row {row_idx + 2} in {csv_path}: "
                    f"'{row['label']}' (expected True or False)"
                )
            
            if label not in (0, 1):
                raise ValueError(
                    f"Label must be 0 or 1, got {label} at row {row_idx + 2} in {csv_path}"
                )
            
            samples.append((filepath, label, row.get('text_info', '').strip()))
    
    # Summary
    n_pos = sum(1 for _, l, *_ in samples if l == 1)
    n_neg = len(samples) - n_pos
    ratio = n_pos / len(samples) if samples else 0
    
    print(f"  Loaded {csv_path.name}: {len(samples)} samples "
          f"(class_0: {n_neg}, class_1: {n_pos}, pos_ratio: {ratio:.3f})")
    
    return samples


# ============================================================
# Transforms
# ============================================================

class ResizeWithAspectRatioPad:
    """Resize image maintaining aspect ratio, then pad to target size.
    
    Resizes the image so the longest side equals target_size,
    then pads the shorter side with the specified fill color.
    """
    
    def __init__(self, target_size: int, fill: Tuple[int, int, int] = (128, 128, 128)):
        """
        Args:
            target_size: Target size for both width and height.
            fill: RGB tuple for padding color (default: gray).
        """
        self.target_size = target_size
        self.fill = fill
    
    def __call__(self, img: Image.Image) -> Image.Image:
        # Get original size
        w, h = img.size
        
        # Calculate new size maintaining aspect ratio
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize with aspect ratio preserved
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Create padded image with fill color
        padded = Image.new("RGB", (self.target_size, self.target_size), self.fill)
        
        # Center the resized image
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        padded.paste(img, (paste_x, paste_y))
        
        return padded


def get_normalization(cfg: DictConfig) -> Tuple[List[float], List[float]]:
    """Get normalization mean and std based on model backbone.

    Args:
        cfg: Full config with model._target_ field

    Returns:
        (mean, std) lists for normalization
    """
    # Get model target to determine backbone type
    model_target = cfg.vision_encoder.get("_target_", "")

    # SigLIP models use 0.5 normalization
    if "siglip" in model_target.lower():
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # ImageNet normalization for ResNet, VGG, ViT, EfficientNet
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_train_transform(cfg: DictConfig) -> T.Compose:
    """Build training transform pipeline.

    Args:
        cfg: Full config (needs cfg.vision_encoder.image_size and cfg.augmentation).
    """
    transforms = []

    # Resize with aspect ratio preserved + center padding
    image_size = cfg.vision_encoder.image_size
    transforms.append(ResizeWithAspectRatioPad(image_size))

    if cfg.augmentation.enabled:
        aug = cfg.augmentation
        # Text-safe augmentations only
        if aug.horizontal_flip_p > 0:
            transforms.append(T.RandomHorizontalFlip(p=aug.horizontal_flip_p))
        if aug.rotation_degrees > 0:
            transforms.append(T.RandomRotation(degrees=aug.rotation_degrees))
        if aug.perspective.p > 0:
            transforms.append(T.RandomPerspective(
                distortion_scale=aug.perspective.distortion,
                p=aug.perspective.p,
            ))
        transforms.append(T.ColorJitter(
            brightness=aug.color_jitter.brightness,
            contrast=aug.color_jitter.contrast,
            saturation=aug.color_jitter.saturation,
            hue=aug.color_jitter.hue,
        ))
        if aug.grayscale_p > 0:
            transforms.append(T.RandomGrayscale(p=aug.grayscale_p))
        if aug.gaussian_blur.p > 0:
            transforms.append(T.RandomApply([
                T.GaussianBlur(
                    kernel_size=aug.gaussian_blur.kernel_size,
                    sigma=(aug.gaussian_blur.sigma_min, aug.gaussian_blur.sigma_max),
                ),
            ], p=aug.gaussian_blur.p))

    # Get backbone-specific normalization
    mean, std = get_normalization(cfg)

    transforms.extend([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(transforms)


def get_eval_transform(cfg: DictConfig) -> T.Compose:
    """Build evaluation transform (no augmentation)."""
    image_size = cfg.vision_encoder.image_size
    mean, std = get_normalization(cfg)

    return T.Compose([
        ResizeWithAspectRatioPad(image_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ])


# ============================================================
# DataLoader Factory
# ============================================================

def create_dataloaders(
    cfg: DictConfig,
    ocr_texts: Optional[Dict[str, str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from separate CSV files.
    
    Args:
        cfg: Full configuration (needs cfg.data, cfg.vision_encoder, cfg.train, cfg.augmentation).
        ocr_texts: Optional dict mapping image_path -> OCR text string.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    print("Loading datasets from CSV files:")
    train_samples = load_samples_from_csv(cfg.data.train_csv)
    val_samples = load_samples_from_csv(cfg.data.val_csv)
    test_samples = load_samples_from_csv(cfg.data.test_csv)
    
    print(f"  Total: {len(train_samples) + len(val_samples) + len(test_samples)} samples "
          f"(train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)})")
    
    # Build transforms
    train_transform = get_train_transform(cfg)
    eval_transform = get_eval_transform(cfg)
    
    # Create datasets
    train_dataset = AdImageDataset(train_samples, train_transform, ocr_texts)
    val_dataset = AdImageDataset(val_samples, eval_transform, ocr_texts)
    test_dataset = AdImageDataset(test_samples, eval_transform, ocr_texts)
    
    # Custom collate to handle OCR text strings
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        paths = [item[2] for item in batch]
        text_info = [item[3] for item in batch]
        return images, labels, paths, text_info
    
    # Seeded generator for reproducible shuffling
    seed = cfg.train.get("seed", 42)
    g = torch.Generator()
    g.manual_seed(seed)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        import random, numpy as np
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        generator=g,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    
    return train_loader, val_loader, test_loader
