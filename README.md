# PalmTreeDetection

**Palm Tree Detection**  
YOLO-based palm tree detection from UAV imagery, developed for CSC 375/675 at Wake Forest University.

## Project Summary  
We used fine-tuned YOLO models (v8–v12) to detect palms in high-res drone images from the Iquitos rainforest in Peru. Images were split into 800×800 patches to help with annotation and model performance.

## Key Points  
- Used Ultralytics YOLO for inference on patches  
- Models trained on Ecuador data didn’t generalize well to Peru  
- Domain shift and limited training diversity caused poor detection  
- Future: retrain with broader, more diverse datasets
