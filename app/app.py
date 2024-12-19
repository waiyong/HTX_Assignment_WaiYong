from fastapi import FastAPI, Request, File, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from utils import load_model, load_class_names, detect_and_blur_license_plate, preprocess_image, transform
import numpy as np
from PIL import Image
import torch
import yaml
from pathlib import Path

# Load configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

ROOT_RATE_LIMIT = config["api_rate_limits"]["ROOT_RATE_LIMIT"]
PREDICT_RATE_LIMIT = config["api_rate_limits"]["PREDICT_RATE_LIMIT"]
ENABLE_PLATE_DETECTION = config["ENABLE_PLATE_DETECTION"]
CLASS_NAMES_PATH = config["CLASS_NAMES_PATH"]
MODEL_PATH = config["MODEL_PATH"]

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"error": "Rate limit exceeded. Please try again later."}
))

# Load model and class names
class_names = load_class_names(CLASS_NAMES_PATH)
model = load_model(MODEL_PATH, len(class_names))

@app.get("/")
@limiter.limit(ROOT_RATE_LIMIT)
async def read_root(request: Request):
    return {
        "message": "Welcome to the Stanford Cars Image Classification API. Visit /docs for documentation.",
        "swagger_ui": "http://localhost:8000/docs"
    }

@app.post("/predict/")
@limiter.limit(PREDICT_RATE_LIMIT)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        content = await file.read()
        image_np = preprocess_image(content)
        
        # Blur license plates if enabled
        if ENABLE_PLATE_DETECTION:
            image_np = detect_and_blur_license_plate(image_np)

        # Convert back to PIL and apply transformations
        processed_image = Image.fromarray(image_np)
        input_tensor = transform(processed_image).unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        predicted_class_name = class_names[predicted_class]
        return {"predicted_class": predicted_class_name}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
