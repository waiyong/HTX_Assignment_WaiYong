import torch
from torchvision import transforms
from torchvision.models import resnet18
from torch import nn
import json
import cv2
import numpy as np
from PIL import Image
import io

# def load_model(model_path, num_classes):
#     """
#     Load the pre-trained model for inference.
#     """
#     model = resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     try:
#         state_dict = torch.load(model_path, map_location=torch.device('cpu'))
#         model.load_state_dict(state_dict)
#         model.eval()
#     except Exception as e:
#         raise RuntimeError(f"Error loading model: {e}")
#     return model

def load_model(model_path: str, num_classes: int, dropout: float = 0.0):
    """
    Load a trained ResNet-18 model with a custom classification head.
    
    Args:
        model_path (str): Path to the saved model state_dict.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability for the classification head.

    Returns:
        torch.nn.Module: Loaded ResNet-18 model.
    """
    try:
        # Define the model architecture to match the training setup
        model = resnet18()
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(model.fc.in_features, num_classes)
        )
        # Load the state_dict
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def load_class_names(class_names_path):
    """
    Load class names from a JSON file.
    """
    try:
        with open(class_names_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading class names: {e}")

def detect_and_blur_license_plate(image: np.ndarray) -> np.ndarray:
    """
    Detect and blur license plates in an image using OpenCV.
    """
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in plates:
        roi = image[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (51, 51), 30)
        image[y:y+h, x:x+w] = roi
    return image

def preprocess_image(image_content):
    """
    Preprocess the uploaded image content for model prediction.
    """
    image = Image.open(io.BytesIO(image_content)).convert("RGB")
    image_np = np.array(image)
    return image_np

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
