import os
import copy
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

import cv2

def detect_and_blur_license_plate(image: np.ndarray) -> np.ndarray:
    """
    Detect and blur license plates in an image using OpenCV.
    
    Args:
        image (np.ndarray): Input image as a NumPy array.
        
    Returns:
        np.ndarray: Processed image with license plates blurred.
    """
    # Load Haar cascade for license plates
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect license plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in plates:
        # Extract and blur the region of interest
        roi = image[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (51, 51), 30)
        image[y:y+h, x:x+w] = roi
    
    return image


def parse_annotations(annotations):
    """
    Parse annotations to extract bounding boxes, class labels, and file names.
    """
    bounding_boxes = []
    class_labels = []
    file_names = []

    for anno in annotations:
        x1 = anno['bbox_x1'][0][0]
        y1 = anno['bbox_y1'][0][0]
        x2 = anno['bbox_x2'][0][0]
        y2 = anno['bbox_y2'][0][0]
        class_id = anno['class'][0][0]
        file_name = anno['fname'][0]

        bounding_boxes.append((x1, y1, x2, y2))
        class_labels.append(class_id - 1)  # Adjust to 0-based indexing
        file_names.append(file_name)

    return np.array(bounding_boxes), np.array(class_labels), np.array(file_names)


# Function to crop and save images
def crop_and_save_images(file_names, bounding_boxes, source_dir, target_dir):
    """
    Crop and save images if they don't already exist.

    Args:
        file_names (list): List of image file names.
        bounding_boxes (list): List of bounding boxes (x1, y1, x2, y2).
        source_dir (Path): Path to the source directory containing raw images.
        target_dir (Path): Path to save the cropped images.
    """
    target_dir.mkdir(parents=True, exist_ok=True)  # Ensure target directory exists

    # Check if images already exist
    if len(list(target_dir.glob("*.jpg"))) == len(file_names):
        print(f"Images already cropped and saved in {target_dir}. Skipping cropping.")
        return

    print(f"Cropping and saving images to {target_dir}...")
    for i, bbox in tqdm(enumerate(bounding_boxes), total=len(bounding_boxes)):
        file_name = file_names[i]
        x1, y1, x2, y2 = bbox

        # Load image
        img_path = source_dir / file_name
        if not img_path.exists():
            print(f"Warning: {img_path} does not exist. Skipping.")
            continue

        # Crop and save the image
        with Image.open(img_path) as img:
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img.save(target_dir / file_name)

    print(f"Cropping completed. Images saved to {target_dir}.")


# Function to display a few cropped images for verification
def display_cropped_images(cropped_dir, num_images=5):
    cropped_files = list(cropped_dir.iterdir())[:num_images]
    plt.figure(figsize=(10, 5))
    for i, file_path in enumerate(cropped_files):
        img = Image.open(file_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(f"Cropped: {file_path.name}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


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

