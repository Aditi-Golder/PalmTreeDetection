import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import cv2
import time
import thop  # For FLOPS calculation
import datetime
import torchvision.ops as ops
import uuid
import sys  # Add this import for sys.argv
from tqdm import tqdm

# Define paths
TEST_IMAGES_DIR = "/deac/csc/paucaGrp/golda24/dl_final/Dataset/test/images"
TEST_LABELS_DIR = "/deac/csc/paucaGrp/golda24/dl_final/Dataset/test/labels"
MODEL_PATHS = {
    "YOLOv8": "/deac/csc/paucaGrp/golda24/dl_final/models/v8_best.pt",
    "YOLOv9": "/deac/csc/paucaGrp/golda24/dl_final/models/v9_best.pt",
    "YOLOv10": "/deac/csc/paucaGrp/golda24/dl_final/models/v10_best.pt",
    "YOLOv11": "/deac/csc/paucaGrp/golda24/dl_final/models/v11_best.pt",
    "YOLOv12": "/deac/csc/paucaGrp/golda24/dl_final/models/yolo12_best.pt",
}
RESULTS_CSV = "/deac/csc/paucaGrp/golda24/dl_final/model_results.csv"

# Create timestamp for output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
TEST_OUTPUT_DIR = f"/deac/csc/paucaGrp/golda24/dl_final/test_output/{timestamp}"


# Predict on patches
def predict_patches(model, patches, conf_threshold=0.25):
    predictions = []
    for patch in tqdm(patches, desc="Predicting patches", leave=False):
        results = model.predict(patch, conf=conf_threshold, verbose=False)
        predictions.append(results[0])
    return predictions

# Merge predictions from patches
def merge_predictions(predictions, coordinates, image_shape, conf_threshold=0.25):
    merged_boxes = []
    merged_scores = []
    merged_classes = []
    
    height, width = image_shape[:2]
    
    for pred, (y_start, x_start, _, _) in tqdm(zip(predictions, coordinates), desc="Merging predictions", leave=False):
        if pred.boxes:
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            classes = pred.boxes.cls.cpu().numpy()
            
            # Filter by confidence threshold
            conf_mask = scores >= conf_threshold
            boxes = boxes[conf_mask]
            scores = scores[conf_mask]
            classes = classes[conf_mask]
            
            # Shift boxes to original image coordinates
            boxes[:, [0, 2]] += x_start
            boxes[:, [1, 3]] += y_start
            
            # Clip to image size
            valid_mask = (boxes[:, 0] < width) & (boxes[:, 1] < height) & \
                        (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            classes = classes[valid_mask]
            
            if len(boxes) > 0:
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

# Apply NMS to remove overlapping boxes
def apply_nms(boxes, scores, classes, iou_threshold=0.5):
    """Apply Non-Maximum Suppression globally."""
    if len(boxes) == 0:
        return boxes, scores, classes

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    
    return boxes[indices.cpu().numpy()], scores[indices.cpu().numpy()], classes[indices.cpu().numpy()]

# Save prediction results
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
        
        if len(box) != 4:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls)]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path), img)

# Function to read YOLO format labels
def read_yolo_labels(label_path):
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # class x y width height
                    class_id = int(parts[0])
                    labels.append(class_id)
    return labels

def convert_yolo_label_to_xyxy(label, image_width, image_height):
    """Convert YOLO format labels (class, x_center, y_center, width, height) to xyxy format."""
    class_id, x_center, y_center, width, height = label
    x1 = (x_center - width/2) * image_width
    y1 = (y_center - height/2) * image_height
    x2 = (x_center + width/2) * image_width
    y2 = (y_center + height/2) * image_height
    return class_id, x1, y1, x2, y2

def get_patch_labels(image_labels, image_shape, patch_coords, patch_size):
    """Extract labels for a specific patch and convert coordinates to patch-relative values."""
    # Unpack coordinates
    y_start, x_start, y_end, x_end = patch_coords
    
    image_height, image_width = image_shape[:2]
    
    patch_labels = []
    
    for label in image_labels:
        # Convert YOLO format to xyxy in original image
        class_id, x_center, y_center, width, height = label
        # Convert to absolute coordinates
        x_center_abs = x_center * image_width
        y_center_abs = y_center * image_height
        width_abs = width * image_width
        height_abs = height * image_height
        
        # Calculate the bounding box in absolute coordinates
        x1_abs = x_center_abs - (width_abs / 2)
        y1_abs = y_center_abs - (height_abs / 2)
        x2_abs = x_center_abs + (width_abs / 2)
        y2_abs = y_center_abs + (height_abs / 2)
        
        # Calculate intersection with patch
        x1_patch = max(x1_abs, x_start)
        y1_patch = max(y1_abs, y_start)
        x2_patch = min(x2_abs, x_end)
        y2_patch = min(y2_abs, y_end)
        
        # Skip if the box doesn't intersect with the patch
        if x1_patch >= x2_patch or y1_patch >= y2_patch:
            continue
        
        # Calculate intersection area
        intersection_area = (x2_patch - x1_patch) * (y2_patch - y1_patch)
        box_area = width_abs * height_abs
        
        # Skip if less than 50% of the box is in the patch
        if intersection_area / box_area < 0.5:
            continue
        
        # Convert back to patch-relative coordinates in YOLO format
        patch_width = x_end - x_start
        patch_height = y_end - y_start
        
        # Convert to patch-relative coordinates
        x1_rel = (x1_patch - x_start)
        y1_rel = (y1_patch - y_start)
        x2_rel = (x2_patch - x_start)
        y2_rel = (y2_patch - y_start)
        
        # Convert back to YOLO format (center_x, center_y, width, height)
        x_center_patch = (x1_rel + x2_rel) / 2
        y_center_patch = (y1_rel + y2_rel) / 2
        width_patch = (x2_rel - x1_rel)
        height_patch = (y2_rel - y1_rel)
        
        patch_labels.append([class_id, x_center_patch, y_center_patch, width_patch, height_patch])
    
    return patch_labels

def read_yolo_labels_with_coords(label_path):
    """Read YOLO format labels with coordinates (class_id, x_center, y_center, width, height)."""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([class_id, x_center, y_center, width, height])
    return labels

# Function to convert predictions to class labels
def get_predicted_classes(boxes, scores, classes, conf_threshold=0.25):
    pred_classes = []
    for cls, score in zip(classes, scores):
        if score >= conf_threshold:
            pred_classes.append(int(cls))
    return pred_classes

# Calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0

def calculate_ap(recalls, precisions):
    """Calculate Average Precision using the 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    return ap

def calculate_map(all_gt_boxes, all_pred_boxes, all_pred_scores, all_gt_classes, all_pred_classes, iou_threshold=0.5):
    """Calculate mAP for object detection."""
    # Get unique classes
    unique_classes = np.unique(np.concatenate((all_gt_classes, all_pred_classes)))
    
    aps = []
    
    for cls in unique_classes:
        # Filter boxes for this class
        gt_indices = np.where(all_gt_classes == cls)[0]
        pred_indices = np.where(all_pred_classes == cls)[0]
        
        gt_boxes_cls = [all_gt_boxes[i] for i in gt_indices]
        pred_boxes_cls = [all_pred_boxes[i] for i in pred_indices]
        pred_scores_cls = [all_pred_scores[i] for i in pred_indices]
        
        if len(gt_boxes_cls) == 0 or len(pred_boxes_cls) == 0:
            continue
        
        # Sort predictions by score (highest first)
        sorted_indices = np.argsort([-score for score in pred_scores_cls])
        pred_boxes_cls = [pred_boxes_cls[i] for i in sorted_indices]
        pred_scores_cls = [pred_scores_cls[i] for i in sorted_indices]
        
        # Go through all predictions
        TP = np.zeros(len(pred_boxes_cls))
        FP = np.zeros(len(pred_boxes_cls))
        gt_detected = [False] * len(gt_boxes_cls)
        
        for i, pred_box in enumerate(pred_boxes_cls):
            # Find the best matching ground truth box
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes_cls):
                if not gt_detected[j]:  # If this gt hasn't been matched yet
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            if best_iou >= iou_threshold:
                # True positive
                TP[i] = 1
                gt_detected[best_gt_idx] = True
            else:
                # False positive
                FP[i] = 1
                
        # Calculate precision and recall
        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        recalls = TP_cumsum / len(gt_boxes_cls)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
        
        # Add a start point (0, 1) for the AP calculation
        precisions = np.concatenate(([1], precisions))
        recalls = np.concatenate(([0], recalls))
        
        # Calculate AP for this class
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
    
    # Calculate mAP
    mAP = np.mean(aps) if len(aps) > 0 else 0
    return mAP

# Add a function to save patches similar to the stride.py implementation
def save_image_patches(patches, coordinates, output_dir, image_name):
    """Save image patches for debugging purposes."""
    patches_dir = os.path.join(output_dir, "patches", image_name)
    os.makedirs(patches_dir, exist_ok=True)
    
    for i, (patch, coord) in enumerate(zip(patches, coordinates)):
        y_start, x_start, y_end, x_end = coord
        patch_filename = os.path.join(patches_dir, f"patch_{i+1}_y{y_start}_x{x_start}.jpg")
        cv2.imwrite(patch_filename, patch)
    
    return patches_dir

# Function to create plots
def create_plots(results_df):
    # Set up the figure with a size
    plt.figure(figsize=(14, 10))
    
    # Bar chart for all metrics
    plt.subplot(2, 3, 1)
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'mAP']
    results_df.set_index('Model')[metrics_to_plot].plot(kind='bar', ax=plt.gca())
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    
    # Individual metrics plots
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'mAP']
    
    # Create individual plots for each metric (except the first one)
    for i, metric in enumerate(metrics[1:], 1):
        plt.subplot(2, 3, i+1)
        plt.bar(results_df['Model'], results_df[metric])
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_OUTPUT_DIR, 'model_comparison_bar.png'))
    
    # Radar chart for comparing models
    categories = metrics
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of categories
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories)
    
    # Draw the chart for each model
    for i, model in enumerate(results_df['Model']):
        values = results_df.loc[i, metrics].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(os.path.join(TEST_OUTPUT_DIR, 'model_comparison_radar.png'))
    
    # Heatmap of the results
    plt.figure(figsize=(10, 8))
    heatmap_data = results_df.set_index('Model')[metrics]
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Score'})
    plt.title('Model Performance Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_OUTPUT_DIR, 'model_comparison_heatmap.png'))
    
    # Instance detection comparison
    plt.figure(figsize=(12, 6))
    x = range(len(results_df))
    width = 0.35
    
    plt.bar(x, results_df['True Instances'], width, label='True Instances', color='green')
    plt.bar([i + width for i in x], results_df['Predicted Instances'], width, label='Predicted Instances', color='blue')
    
    plt.xlabel('Models')
    plt.ylabel('Number of Instances')
    plt.title('Palm Detection Count Comparison')
    plt.xticks([i + width/2 for i in x], results_df['Model'], rotation=45)
    plt.legend()
    
    # Add the Instance Ratio as text on top of each model
    for i, ratio in enumerate(results_df['Instance Ratio']):
        true_instances = results_df['True Instances'].iloc[i]
        pred_instances = results_df['Predicted Instances'].iloc[i]
        
        # Position text above the tallest bar
        y_pos = max(true_instances, pred_instances) + 50
        
        plt.text(i + width/2, y_pos, f'Ratio: {ratio:.2f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_OUTPUT_DIR, 'instance_detection_comparison.png'))

# Function to evaluate model on dataset
def evaluate_model(model, image_dir, label_dir, model_name, conf_threshold=0.25):
    all_true_labels = []
    all_pred_labels = []
    
    # For counting metrics
    total_true_instances = 0
    total_pred_instances = 0
    
    # For mAP calculation
    all_gt_boxes = []
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_classes = []
    all_pred_classes = []
    
    # Create output directory for this model
    model_output_dir = os.path.join(TEST_OUTPUT_DIR, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in tqdm(image_files, desc=f"Evaluating {model_name}", leave=True):
        # Get image and corresponding label path
        image_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}", flush=True)
            continue
            
        # Get ground truth labels (with coordinates)
        image_labels = read_yolo_labels_with_coords(label_path)
        
        # Count true instances for this image
        true_count = len(image_labels)
        total_true_instances += true_count
        
        # Direct prediction on the full image (no patches)
        results = model.predict(image, conf=conf_threshold, verbose=False)[0]
        
        # Extract prediction data
        if results.boxes:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            # Filter by confidence threshold
            conf_mask = scores >= conf_threshold
            boxes = boxes[conf_mask]
            scores = scores[conf_mask]
            classes = classes[conf_mask]
        else:
            boxes = np.empty((0, 4))
            scores = np.array([])
            classes = np.array([])
        
        # Count predicted instances
        pred_count = len(classes)
        total_pred_instances += pred_count
        
        # Save the annotated image with predictions
        output_path = os.path.join(model_output_dir, f"result_{os.path.basename(image_file)}")
        save_results(image, boxes, scores, classes, output_path, model.names)
        
        # Get predicted class list (unique classes)
        pred_classes = list(set([int(cls) for cls in classes]))
        
        # Get true class list (unique classes)
        true_classes = list(set([int(label[0]) for label in image_labels]))
        
        # Store data for mAP calculation
        image_height, image_width = image.shape[:2]
        for label in image_labels:
            class_id, x_center, y_center, width, height = label
            # Convert YOLO format to xyxy
            x1 = (x_center - width/2) * image_width
            y1 = (y_center - height/2) * image_height
            x2 = (x_center + width/2) * image_width
            y2 = (y_center + height/2) * image_height
            all_gt_boxes.append([x1, y1, x2, y2])
            all_gt_classes.append(int(class_id))
        
        for i in range(len(boxes)):
            all_pred_boxes.append(boxes[i])
            all_pred_scores.append(scores[i])
            all_pred_classes.append(int(classes[i]))
        
        # Convert to binary presence/absence for each class
        # Assuming we're working with a fixed set of classes (e.g., 0-9)
        num_classes = 1  # Update this based on your dataset - looks like you only have class 0 (palm)
        
        # Convert to multi-hot encoding
        true_binary = np.zeros(num_classes)
        pred_binary = np.zeros(num_classes)
        
        for cls in true_classes:
            if cls < num_classes:
                true_binary[cls] = 1
                
        for cls in pred_classes:
            if cls < num_classes:
                pred_binary[cls] = 1
        
        all_true_labels.append(true_binary)
        all_pred_labels.append(pred_binary)
    
    # Convert lists to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    
    # Calculate mAP
    mAP = calculate_map(all_gt_boxes, all_pred_boxes, all_pred_scores, 
                        np.array(all_gt_classes), np.array(all_pred_classes))
    
    # Calculate other metrics
    if len(all_true_labels) > 0:
        # Average across all classes
        precision = precision_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
        f1 = f1_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
        
        # For accuracy, we'll use the sample-wise accuracy
        accuracy = 0
        for i in range(len(all_true_labels)):
            if np.array_equal(all_true_labels[i], all_pred_labels[i]):
                accuracy += 1
        accuracy /= len(all_true_labels)
        
        # Calculate instance count metrics
        instance_ratio = total_pred_instances / max(1, total_true_instances)
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'mAP': mAP,
            'Total True Instances': total_true_instances,
            'Total Predicted Instances': total_pred_instances,
            'Instance Ratio': instance_ratio
        }
    else:
        return {
            'Accuracy': 0,
            'Precision': 0,
            'Recall': 0,
            'F1-score': 0,
            'mAP': 0,
            'Total True Instances': 0,
            'Total Predicted Instances': 0,
            'Instance Ratio': 0
        }

# Main function
def main():
    # Create output directory with timestamp
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory: {TEST_OUTPUT_DIR}", flush=True)
    
    results = []
    
    # Evaluation parameters
    conf_threshold = 0.10  # Lower confidence threshold for better detection
    
    # Evaluate each model
    for model_name, model_path in tqdm(MODEL_PATHS.items(), desc="Evaluating models"):
        print(f"Evaluating {model_name}...", flush=True)
        try:
            model = YOLO(model_path)
            metrics = evaluate_model(model, TEST_IMAGES_DIR, TEST_LABELS_DIR, model_name, 
                                    conf_threshold=conf_threshold)
            
            results.append({
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-score': metrics['F1-score'],
                'mAP': metrics['mAP'],
                'True Instances': metrics['Total True Instances'],
                'Predicted Instances': metrics['Total Predicted Instances'],
                'Instance Ratio': metrics['Instance Ratio']
            })
            print(f"{model_name} evaluated successfully.", flush=True)
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}", flush=True)
    
    # Save results to CSV
    results_csv_path = os.path.join(TEST_OUTPUT_DIR, "model_results.csv")
    with open(results_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'mAP',
                     'True Instances', 'Predicted Instances', 'Instance Ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Also save to main results CSV
    with open(RESULTS_CSV, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'mAP',
                     'True Instances', 'Predicted Instances', 'Instance Ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Create comparison plots
    if results:
        results_df = pd.DataFrame(results)
        create_plots(results_df)
        print(f"Comparison plots created in {TEST_OUTPUT_DIR}")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()
