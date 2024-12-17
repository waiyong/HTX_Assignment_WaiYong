import cv2
import numpy as np

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