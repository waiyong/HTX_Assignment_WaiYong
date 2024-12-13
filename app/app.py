from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
import io
import json

with open("class_names.json", "r") as f:
    class_names = json.load(f)


# Define the model architecture
model = resnet18(weights=None)  # Start with no pre-trained weights
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Adjust final layer to match the number of classes

# Load the state dictionary
model_path = "model.pth"  # Path to your model weights
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Stanford Cars Image Classification API. Visit /docs for documentation.",
            "swagger_ui": "http://localhost:8000/docs"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        # Apply transformations
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Map predicted class to class name, if available
        predicted_class_name = class_names[predicted_class] if class_names else predicted_class

        # Return prediction result
        return {"predicted_class": predicted_class_name}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})