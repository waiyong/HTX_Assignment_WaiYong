# Use the official Python image as the base
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY app/ /app/
COPY resnet18_finetuned_Adam_lr0.001_bs32_epoch5_10Dec_v1.pth /app/model.pth 
COPY class_names.json /app/class_names.json


# Expose port 8000
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
