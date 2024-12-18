from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib.ticker as mticker

def extract_image_dimensions(image_dir):
    """
    Extract the width and height of all images in the specified directory.

    Args:
        image_dir (str or Path): Path to the directory containing images.

    Returns:
        pd.DataFrame: A DataFrame containing image names, widths, and heights.
    """
    image_dimensions = []

    for img_path in Path(image_dir).iterdir():
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:  # Check for valid image formats
            with Image.open(img_path) as img:
                width, height = img.size
                image_dimensions.append({'file_name': img_path.name, 'width': width, 'height': height})
    
    return pd.DataFrame(image_dimensions)

def plot_image_dimension_distributions(df, dataset_type="train"):
    """
    Plot the distribution of image widths and heights.

    Args:
        df (pd.DataFrame): DataFrame containing image dimensions.
        dataset_type (str): "train" or "test", for labeling the plots.
    """
    plt.figure(figsize=(10, 5))

    # Width distribution
    plt.subplot(1, 2, 1)
    plt.hist(df['width'], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Image Width Distribution ({dataset_type.capitalize()} Set)")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Frequency")

    # Height distribution
    plt.subplot(1, 2, 2)
    plt.hist(df['height'], bins=30, color='orange', edgecolor='black')
    plt.title(f"Image Height Distribution ({dataset_type.capitalize()} Set)")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def plot_bounding_box_distributions(df, dataset_type="train"):
    """
    Plot bounding box area and area ratio distributions.
    
    Args:
        df (pd.DataFrame): DataFrame containing bounding box stats.
        dataset_type (str): "train" or "test", for labeling the plots.
    """
    plt.figure(figsize=(10, 5))

    # Bounding box area distribution
    plt.subplot(1, 2, 1)
    plt.hist(df['bbox_area'], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Bounding Box Area Distribution ({dataset_type.capitalize()} Set)")
    plt.xlabel("Bounding Box Area (pixels)")
    plt.ylabel("Frequency")
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Bounding box area ratio distribution
    plt.subplot(1, 2, 2)
    plt.hist(df['bbox_area_ratio'], bins=30, color='orange', edgecolor='black')
    plt.title(f"Bounding Box Area Ratio Distribution ({dataset_type.capitalize()} Set)")
    plt.xlabel("Bounding Box Area Ratio")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def calculate_bounding_box_stats_with_actual_dimensions(df, image_dir):
    """
    Calculate bounding box statistics using actual image dimensions and add columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing bounding box annotations.
        image_dir (str or Path): Directory containing the images.
    
    Returns:
        pd.DataFrame: Updated DataFrame with additional columns for bounding box stats.
        float: Average bounding box area ratio.
    """
    image_areas = {}
    
    # Extract actual image dimensions
    for img_path in Path(image_dir).iterdir():
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:  # Check valid image formats
            with Image.open(img_path) as img:
                width, height = img.size
                image_areas[img_path.name] = width * height  # Store image area
                
    # Convert bounding box coordinates to int
    df['x1'] = df['x1'].astype('int64')
    df['x2'] = df['x2'].astype('int64')
    df['y1'] = df['y1'].astype('int64')
    df['y2'] = df['y2'].astype('int64')

    # Calculate bounding box stats
    df['bbox_width'] = (df['x2'] - df['x1']).astype('int64')
    df['bbox_height'] = (df['y2'] - df['y1']).astype('int64')
    df['bbox_area'] = df['bbox_width'] * df['bbox_height']

    # Map image areas to rows in DataFrame
    df['image_area'] = df['file_name'].map(image_areas).astype('float64')
    df['bbox_area_ratio'] = df['bbox_area'] / df['image_area']
    
    # Handle missing images
    if df['image_area'].isnull().any():
        print("Warning: Some image areas could not be calculated.")
    
    # Calculate average bounding box area ratio
    average_bbox_ratio = df['bbox_area_ratio'].mean()
    return df, average_bbox_ratio

def calculate_bounding_box_stats(df, image_dir):
    """
    Calculate bounding box statistics such as area and area ratio.
    """
    # Extract actual image dimensions
    image_areas = {}
    for img_path in Path(image_dir).iterdir():
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            with Image.open(img_path) as img:
                width, height = img.size
                image_areas[img_path.name] = width * height

    # Add bounding box stats
    df['bbox_width'] = (df['x2'] - df['x1']).astype('int64')
    df['bbox_height'] = (df['y2'] - df['y1']).astype('int64')
    df['bbox_area'] = df['bbox_width'] * df['bbox_height']
    df['image_area'] = df['file_name'].map(image_areas).astype('float64')
    df['bbox_area_ratio'] = df['bbox_area'] / df['image_area']

    return df


def plot_samples_with_bboxes(num_samples, annotations, class_names, cars_dir, dataset_type="train"):
    """
    Plot images with bounding boxes for training or test set.

    Args:
        num_samples (int): Number of samples to display.
        annotations (list): List of annotations for the dataset.
        class_names (list): List of car class names.
        cars_dir (Path or str): Path to the directory containing images.
        dataset_type (str): "train" or "test", for labeling the dataset.
    """
    plt.figure(figsize=(12, 5))  # Adjust figure size for clarity
    
    for i in range(num_samples):
        # Get annotation details
        annotation = annotations[i]
        x1 = annotation['bbox_x1'][0][0]
        y1 = annotation['bbox_y1'][0][0]
        x2 = annotation['bbox_x2'][0][0]
        y2 = annotation['bbox_y2'][0][0]
        class_id = annotation['class'][0][0]
        file_name = annotation['fname'][0]
        car_name = class_names[class_id - 1]  # Map class ID to car name

        # Load the image
        img_path = Path(cars_dir) / file_name
        if not img_path.exists():
            print(f"Warning: {file_name} does not exist in {cars_dir}. Skipping.")
            continue
        with Image.open(img_path) as img:
            # Plot image
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img)
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                              edgecolor='red', facecolor='none', linewidth=2))
            plt.title(f"{dataset_type.capitalize()}: {car_name}", fontsize=8)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def parse_annotations_to_dataframe(annotations, class_names):
    """
    Parse annotations into a DataFrame and calculate class distribution.

    Args:
        annotations (list): List of annotations (train or test).
        class_names (list): List of class names.

    Returns:
        pd.DataFrame: DataFrame with file names, bounding box info, class IDs, and class names.
        pd.Series: Class distribution counts.
    """
    if annotations.size == 0:  # Check if annotations is empty
        raise ValueError("Annotations array is empty.")

    data = []
    for anno in annotations:
        try:
            x1 = anno['bbox_x1'][0][0]
            y1 = anno['bbox_y1'][0][0]
            x2 = anno['bbox_x2'][0][0]
            y2 = anno['bbox_y2'][0][0]
            class_id = anno['class'][0][0]
            file_name = anno['fname'][0]
            data.append([file_name, x1, y1, x2, y2, class_id, class_names[class_id - 1]])
        except Exception as e:
            print(f"Error parsing annotation: {anno}. Skipping. Error: {e}")
            continue

    df = pd.DataFrame(data, columns=['file_name', 'x1', 'y1', 'x2', 'y2', 'class_id', 'class_name'])
    class_counts = df['class_name'].value_counts()
    return df, class_counts



def plot_class_distribution(class_counts, dataset_type="train", save_path=None):
    """
    Plot class distribution as a bar chart.

    Args:
        class_counts (pd.Series): Class distribution counts.
        dataset_type (str): "train" or "test", for labeling the dataset.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    if class_counts.empty:
        raise ValueError("Class counts are empty. Cannot plot distribution.")

    plt.figure(figsize=(10, 5))
    class_counts.plot(kind='bar')
    plt.title(f"Number of Images per Car Class ({dataset_type.capitalize()} Set)")
    plt.xlabel("Car Class")
    plt.ylabel("Number of Images")
    plt.xticks([])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Class distribution plot saved to {save_path}")
    else:
        plt.show()


# Analyze and plot distribution for a given dataset
def analyze_class_distribution(annotations, class_names, dataset_type):
    """
    Wrapper function to analyze class distribution and plot the results.

    Args:
        annotations (list): List of annotations (train or test).
        class_names (list): List of class names.
        dataset_type (str): Type of dataset ("train" or "test").
    """
    df, class_counts = parse_annotations_to_dataframe(annotations, class_names)
    print(f"Top 5 {dataset_type.capitalize()} Classes and Frequency:")
    print(class_counts.head(5))
    print(f"Bottom 5 {dataset_type.capitalize()} Classes and Frequency:")
    print(class_counts.tail(5))
    plot_class_distribution(class_counts, dataset_type=dataset_type)


def visualize_low_ratio_images(df, cars_dir, ratio_threshold=0.1, num_samples=5, dataset_type="train"):
    """
    Visualize images with bounding box-to-image area ratio below a given threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing bounding box stats and area ratios.
        cars_dir (Path or str): Directory containing images.
        ratio_threshold (float): Threshold for bounding box area ratio.
        num_samples (int): Number of samples to display.
        dataset_type (str): "train" or "test", for labeling the dataset.
    """
    # Filter DataFrame for low area ratio
    low_ratio_df = df[df['bbox_area_ratio'] < ratio_threshold]
    
    # Select a few samples randomly
    sampled_rows = low_ratio_df.sample(min(num_samples, len(low_ratio_df)))
    
    plt.figure(figsize=(10, 5))  # Adjust figure size
    
    for idx, row in enumerate(sampled_rows.itertuples()):
        # Load image
        img_path = Path(cars_dir) / row.file_name
        if not img_path.exists():
            print(f"Warning: {row.file_name} does not exist in {cars_dir}. Skipping.")
            continue
        
        with Image.open(img_path) as img:
            # Plot image with bounding box
            plt.subplot(1, num_samples, idx + 1)
            plt.imshow(img)
            plt.gca().add_patch(plt.Rectangle(
                (row.x1, row.y1), row.bbox_width, row.bbox_height,
                edgecolor='red', facecolor='none', linewidth=2
            ))
            plt.title(f"{dataset_type.capitalize()} Set\nBBox Ratio: {row.bbox_area_ratio:.2f}", fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()