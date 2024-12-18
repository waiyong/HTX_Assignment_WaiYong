import os
from pathlib import Path
import copy
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from utils.image_processing import detect_and_blur_license_plate





def stratified_split(image_label_dict, test_size=0.2, random_state=42):
    """
    Perform a stratified split on a dataset.

    Args:
        image_label_dict (dict): Dictionary of image filenames and their labels.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Validation dictionary and updated testing dictionary.
    """
    filenames = list(image_label_dict.keys())
    labels = list(image_label_dict.values())

    indices_train, indices_val = train_test_split(
        range(len(filenames)),
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    validation_dict = {filenames[i]: labels[i] for i in indices_val}
    updated_test_dict = {filenames[i]: labels[i] for i in indices_train}

    return validation_dict, updated_test_dict


def get_augmentation_pipelines(image_size=(224, 224)):
    """
    Define augmentation and basic transformation pipelines.

    Args:
        image_size (tuple): Target image size.

    Returns:
        tuple: Augmentation pipeline and basic transformation pipeline.
    """
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    basic_pipeline = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return augmentation_pipeline, basic_pipeline



class StanfordCarsDataset(Dataset):
    """Custom dataset class for Stanford Cars dataset."""
    def __init__(self, image_label_dict, root_dir, augmentation_pipeline, basic_pipeline, use_augmentation=False, enable_plate_detection=False):
        """
        Args:
            image_label_dict (dict): Dictionary mapping image filenames to labels.
            root_dir (Path): Root directory containing the images.
            augmentation_pipeline (albumentations.Compose): Augmentation pipeline.
            basic_pipeline (albumentations.Compose): Basic transformation pipeline.
            use_augmentation (bool): Whether to apply augmentation.
            enable_plate_detection (bool): Whether to enable license plate detection and blurring.
        """
        self.image_label_dict = image_label_dict
        self.root_dir = root_dir
        self.use_augmentation = use_augmentation
        self.enable_plate_detection = enable_plate_detection
        self.augmentation_pipeline = augmentation_pipeline
        self.basic_pipeline = basic_pipeline

    def __len__(self):
        return len(self.image_label_dict)

    def __getitem__(self, idx):
        image_filename = list(self.image_label_dict.keys())[idx]
        label = self.image_label_dict[image_filename]
        image_path = self.root_dir / image_filename
        image = Image.open(image_path).convert("RGB")
        img_np = np.array(image)

        # Apply license plate detection if enabled
        if self.enable_plate_detection:
            img_np = detect_and_blur_license_plate(img_np)

        # Apply the appropriate pipeline
        transformed = self.augmentation_pipeline(image=img_np) if self.use_augmentation else self.basic_pipeline(image=img_np)
        image = transformed["image"]
        return image, label

