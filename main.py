from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import uuid
import os
import random
random.seed(42)  # Set seed for reproducibility

def create_patches(image, patch_size=800, overlap=0):
    """Create patches from a large image with specified patch size and overlap, ignoring incomplete patches."""
    height, width = image.shape[:2]
    patches = []
    coordinates = []
    
    step = patch_size - overlap
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            if y + patch_size <= height and x + patch_size <= width:
                patch = image[y:y + patch_size, x:x + patch_size]
                patches.append(patch)
                coordinates.append((y, x, y + patch_size, x + patch_size))
    
    return patches, coordinates

def predict_patches(model, patches, conf_threshold=0.25):
    """Run YOLO predictions on patches."""
    predictions = []
    
    for patch in patches:
        results = model.predict(patch, conf=conf_threshold, verbose=False)
        predictions.append(results[0])
    
    return predictions

def merge_predictions(predictions, coordinates, image_shape, conf_threshold=0.25):
    """Merge predictions from patches into the original image coordinates."""
    merged_boxes = []
    merged_scores = []
    merged_classes = []
    
    height, width = image_shape[:2]
    
    for pred, (y_start, x_start, y_end, x_end) in zip(predictions, coordinates):
        if pred.boxes:
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            classes = pred.boxes.cls.cpu().numpy()
            
            boxes[:, [0, 2]] += x_start
            boxes[:, [1, 3]] += y_start
            
            valid_mask = (boxes[:, 0] < width) & (boxes[:, 1] < height) & \
                        (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            classes = classes[valid_mask]
            
            merged_boxes.append(boxes)
            merged_scores.append(scores)
            merged_classes.append(classes)
    
    if merged_boxes:
        merged_boxes = np.vstack(merged_boxes)
        merged_scores = np.hstack(merged_scores)
        merged_classes = np.hstack(merged_classes)
    else:
        merged_boxes = np.array([])
        merged_scores = np.array([])
        merged_classes = np.array([])
    
    return merged_boxes, merged_scores, merged_classes

def save_results(image, boxes, scores, classes, output_path, class_names):
    """Draw bounding boxes on the original image and save it."""
    img = image.copy()
    
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls)]} {score:.2f}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path), img)

def main(image_path, output_dir="Yolo_v8_output", patch_size=800, overlap=0, conf_threshold=0.25):
    """Main function to process image, create patches, predict, and save results."""
    # Load YOLO model
    model = YOLO("/deac/csc/paucaGrp/golda24/dl_final/models/v8_best.pt")
    
    # Print original class names for debugging
    print("Original class names:", model.names)
    
    # Define custom class names for annotation
    class_names = {0: 'palm'}  # Use 'palm' instead of 'item'
    print("Class names used for annotation:", class_names)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create patches
    patches, coordinates = create_patches(image, patch_size, overlap)
    
    # Predict on patches
    predictions = predict_patches(model, patches, conf_threshold)
    
    # Merge predictions
    boxes, scores, classes = merge_predictions(predictions, coordinates, image.shape, conf_threshold)
    
    # Generate unique output filename
    output_filename = output_dir / f"result_{uuid.uuid4().hex[:8]}.jpg"
    
    # Save results
    save_results(image, boxes, scores, classes, output_filename, class_names)
    
    print(f"Results saved to {output_filename}")
    
    return boxes, scores, classes, class_names

def save_patches(patches, output_dir):
    """Save patches to the specified output directory."""
    for i, patch in enumerate(patches):
        patch_filename = output_dir / f"patch_{i + 1}.jpg"
        cv2.imwrite(str(patch_filename), patch)

def process_random_images(image_dir, patch_output_dir, patch_size=800, overlap=0):
    """Randomly select two images, create patches, and save them."""
    image_paths = [
        os.path.join(image_dir, file)
        for file in os.listdir(image_dir)
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    selected_images = random.sample(image_paths, 2)
    for image_path in selected_images:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image at {image_path}")
            continue
        patches, _ = create_patches(image, patch_size, overlap)
        image_name = Path(image_path).stem
        output_subdir = Path(patch_output_dir) / image_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        save_patches(patches, output_subdir)
        print(f"Patches saved to {output_subdir}")

if __name__ == "__main__":
    image_dir = "/deac/csc/paucaGrp/golda24/dl_final/test_output/20250430_011312/YOLOv8/patches/geotagged1_DSC04212_geotag_JPG.rf.166fec6da4f00df0cf64a33ea3c1aa2e"
    image_paths = [
        os.path.join(image_dir, file)
        for file in os.listdir(image_dir)
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    for image_path in image_paths:
        print(f"Processing {image_path}...", flush=True)
        main(image_path,patch_size=800, overlap=0, conf_threshold=0.10)
    process_random_images(image_dir, patch_output_dir="patch_output", patch_size=800, overlap=0)