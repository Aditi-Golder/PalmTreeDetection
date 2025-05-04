from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import uuid
import os
import random
import torchvision.ops as ops
import torch
from tqdm import tqdm
random.seed(42)  # Reproducibility

def create_patches(image, patch_size=800, stride=600):
    """Create overlapping patches using stride."""
    height, width = image.shape[:2]
    patches = []
    coordinates = []
    
    # Validate stride and set a default if it's 0
    if stride <= 0:
        # No overlap - set stride equal to patch size
        stride = patch_size
        print(f"Warning: stride was set to 0 or negative. Using stride={patch_size} instead.")
    
    # Ensure patch size is not larger than the image
    if patch_size > height or patch_size > width:
        patch_size = min(height, width)
        print(f"Warning: patch_size was larger than image dimensions. Using patch_size={patch_size} instead.")
        # If we had to adjust patch_size, also ensure stride isn't larger than patch_size
        if stride > patch_size:
            stride = patch_size
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            coordinates.append((y, x, y + patch_size, x + patch_size))

    return patches, coordinates

def predict_patches(model, patches, conf_threshold=0.25):
    predictions = []
    for patch in tqdm(patches, desc="Predicting patches", leave=False):
        results = model.predict(patch, conf=conf_threshold, verbose=False)
        predictions.append(results[0])
    return predictions

def merge_predictions(predictions, coordinates, image_shape):
    merged_boxes = []
    merged_scores = []
    merged_classes = []

    height, width = image_shape[:2]

    for pred, (y_start, x_start, _, _) in tqdm(zip(predictions, coordinates), desc="Merging predictions", leave=False):
        if pred.boxes:
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            classes = pred.boxes.cls.cpu().numpy()

            # Shift boxes to original image coordinates
            boxes[:, [0, 2]] += x_start
            boxes[:, [1, 3]] += y_start

            # Clip to image size
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
        merged_boxes = np.empty((0, 4))
        merged_scores = np.array([])
        merged_classes = np.array([])

    return merged_boxes, merged_scores, merged_classes

def apply_nms(boxes, scores, classes, iou_threshold=0.5):
    """Apply Non-Maximum Suppression globally."""
    if len(boxes) == 0:
        return boxes, scores, classes

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)

    return boxes[indices], scores[indices], classes[indices]

def save_results(image, boxes, scores, classes, output_path, class_names):
    """Save detection results on the image.
    
    Ensures boxes, scores, and classes are properly formatted as arrays.
    """
    img = image.copy()
    
    # Check if we have any detections
    if len(boxes) == 0:
        cv2.imwrite(str(output_path), img)
        return
    
    # Ensure we're working with arrays
    if not hasattr(boxes, '__iter__') or np.isscalar(boxes):
        boxes = np.array([[boxes]])
    if not hasattr(scores, '__iter__') or np.isscalar(scores):
        scores = np.array([scores])
    if not hasattr(classes, '__iter__') or np.isscalar(classes):
        classes = np.array([classes])
    
    # Make sure boxes is properly shaped as a 2D array
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    
    # Now we can safely iterate
    for i in range(len(scores)):
        if i >= len(boxes):
            break
            
        box = boxes[i]
        score = scores[i]
        cls = classes[i]
        
        # Ensure box is properly formatted
        if len(box) != 4:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls)]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path), img)

def main(image_path, output_dir="output", patch_size=800, stride=600, conf_threshold=0.25):
    model = YOLO("/deac/csc/paucaGrp/golda24/dl_final/models/v10_best.pt")

    print("Class names:", model.names)
    class_names = {0: 'palm'}

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    patches, coordinates = create_patches(image, patch_size, stride)
    predictions = predict_patches(model, patches, conf_threshold)
    boxes, scores, classes = merge_predictions(predictions, coordinates, image.shape)

    # Apply global NMS to reduce duplicate detections
    boxes, scores, classes = apply_nms(boxes, scores, classes)

    # Extract original filename and use it for output
    original_filename = Path(image_path).name
    output_filename = output_dir / f"result_{original_filename}"
    save_results(image, boxes, scores, classes, output_filename, class_names)

    print(f"Results saved to {output_filename}")
    return boxes, scores, classes, class_names

# -- Optional Patch Export for Debugging --
def save_patches(patches, output_dir):
    for i, patch in enumerate(patches):
        patch_filename = output_dir / f"patch_{i + 1}.jpg"
        print(f"Saving patch {i + 1} to {patch_filename}")
        cv2.imwrite(str(patch_filename), patch)

def process_random_images(image_dir, patch_output_dir, patch_size=800, stride=600):
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
        patches, _ = create_patches(image, patch_size, stride)
        image_name = Path(image_path).stem
        output_subdir = Path(patch_output_dir) / image_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        save_patches(patches, output_subdir)
        print(f"Patches saved to {output_subdir}")

if __name__ == "__main__":
    image_dir = "/deac/csc/paucaGrp/golda24/dl_final/Dataset/test/images"
    # image_dir = "/deac/csc/paucaGrp/golda24/dl_final/Dataset/IquitosPalmData-small"
    image_paths = [
        os.path.join(image_dir, file)
        for file in os.listdir(image_dir)
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    for image_path in tqdm(image_paths, desc="Processing images"):
        print(f"Processing {image_path}...", flush=True)
        main(image_path, "output_YOLOv10_try-small", patch_size=512, stride=0, conf_threshold=0.25)

    process_random_images(image_dir, patch_output_dir="patch_output_Yolo10_try-small", patch_size=512, stride=0)
