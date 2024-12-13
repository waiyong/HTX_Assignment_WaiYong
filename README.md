
# Advanced Vehicle Classification for Enhanced Homeland Security

Project Overview: 
In line with HTX's mission to harness cutting-edge technologies for safeguarding Singapore
MINISTRY OF HOME AFFAIRS
, this project presents a sophisticated computer vision solution designed to accurately classify vehicle types from images. This capability is pivotal for various Home Team departments, including the Singapore Police Force and the Immigration and Checkpoints Authority, in bolstering surveillance, traffic management, and security operations.

Key Features:

High-Precision Vehicle Classification: Utilizes a fine-tuned ResNet-18 model trained on the Stanford Cars Dataset, capable of distinguishing between 196 car models with high accuracy.
Real-Time API Integration: Implements a FastAPI-based service, facilitating seamless integration with existing Home Team systems for instantaneous vehicle identification.
Scalable and Secure Deployment: Employs Docker containerization to ensure the solution is easily deployable, scalable, and secure across various operational environments.
Strategic Alignment:

This initiative aligns with HTX's commitment to advancing science and technology in Home Team. By integrating this vehicle classification system, HTX can enhance situational awareness and operational efficiency, contributing to Singapore's safety and security.

 
Task: You are tasked to train an image classification model and serve the model via API.


## Installation and Setup

1. Clone the repo
```bash
    git clone <insert github link>
```

2. Install conda environment

```bash
  conda env create -f environment.yml
  cd my-project
```

3. Check if Docker Is installed. Run the following command to check if Docker is installed. 

```bash
    docker --version
```
If Docker is installed, it will display the version number (e.g., Docker version 20.10.21, build 20fd1d6).

5. Verify Docker Daemon Is Running

```bash
    docker info
```
If the command fails, they need to start Docker. On macOS or Windows, this typically means opening the Docker Desktop application.

6. Build and run the Docker container

```base
    docker build -t stanford-cars-api .
    docker run -p 8000:8000 stanford-cars-api
```

7. Open your browser and navigate to:

```base
    http://localhost:8000/docs
```
This opens the Swagger UI where you can interact with the API.
Use the /predict/ endpoint to upload an image for classification.
    
## Model and Dataset

The Stanford Cars Dataset is a highly specialized dataset designed for fine-grained image classification tasks. It contains images of 196 car models, making it an ideal benchmark for recognizing subtle differences between visually similar classes.

Source: Kaggle - Stanford Cars Dataset
Contents:
Training Set: 8,144 images with annotations.
Testing Set: 8,041 images with annotations.
Metadata: Includes class names and bounding box coordinates for each image.
Classes: 196 distinct car models, ranging from sedans to SUVs, annotated with specific makes and models.
Image Resolution: High-resolution images, varying in size, with diverse backgrounds and lighting conditions.