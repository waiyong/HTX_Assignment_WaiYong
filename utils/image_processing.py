import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
