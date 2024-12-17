from fastapi import FastAPI, Request, File, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
import io
import json
import cv2
import numpy as np

# Global toggle for license plate detection
ENABLE_PLATE_DETECTION = True  # Set to False to disable license plate detection

# Initialize the rate limiter
limiter = Limiter(key_func=get_remote_address)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Define the model architecture
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
state_dict = torch.load("model.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize FastAPI app
app = FastAPI()

# Integrate the rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"error": "Rate limit exceeded. Please try again later."}
))

@app.get("/")
@limiter.limit("10/minute")  # Allow 10 requests per minute per client
async def read_root(request: Request):
    return {
        "message": "Welcome to the Stanford Cars Image Classification API. Visit /docs for documentation.",
        "swagger_ui": "http://localhost:8000/docs"
    }

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

@app.post("/predict/")
@limiter.limit("5/minute")  # Allow 5 requests per minute per client
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image_np = np.array(image)  # Convert PIL image to NumPy array

        # Perform license plate detection if enabled
        if ENABLE_PLATE_DETECTION:
            image_np = detect_and_blur_license_plate(image_np)

        # Convert the processed image back to PIL format for classification
        processed_image = Image.fromarray(image_np)
        
        # Apply transformations for classification
        input_tensor = transform(processed_image).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Map predicted class to class name, if available
        predicted_class_name = class_names[predicted_class] if class_names else predicted_class

        return {"predicted_class": predicted_class_name}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
