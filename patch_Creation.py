import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings

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

def save_patches(image_path, output_dir="patch_output", patch_size=800, stride=400):
    """
    Extract patches from an image and save them to the specified output directory.
    
    Args:
        image_path: Path to the input image
        output_dir: Base directory to save patches
        patch_size: Size of the patches (square)
        stride: Stride between patches
    
    Returns:
        Number of patches created
    """
    try:
        # Read the image with error handling
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return 0
            
        # Verify image integrity
        # Try to encode and decode the image to check if it's valid
        is_valid = True
        try:
            _, buffer = cv2.imencode('.jpg', image)
            cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except cv2.error:
            is_valid = False
            
        if not is_valid:
            print(f"Error: Image at {image_path} appears to be corrupted")
            return 0
        
        # Get image name without extension
        image_name = Path(image_path).stem
        
        # Create output directory structure
        image_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Create patches
        patches, coordinates = create_patches(image, patch_size, stride)
        
        # Save patches with sequential numbering
        patch_count = 0
        for i, patch in enumerate(patches):
            try:
                patch_file = os.path.join(image_output_dir, f"{image_name}_{i+1}.jpg")
                success = cv2.imwrite(patch_file, patch)
                if success:
                    patch_count += 1
                else:
                    print(f"Warning: Failed to save patch {i+1} from {image_path}")
            except Exception as e:
                print(f"Error saving patch {i+1} from {image_path}: {str(e)}")
        
        print(f"Created {patch_count} patches from {image_path}")
        return patch_count
        
    except cv2.error as e:
        print(f"OpenCV Error processing {image_path}: {str(e)}")
        return 0
    except Exception as e:
        print(f"Unexpected error processing {image_path}: {str(e)}")
        return 0

def process_directory(input_dir, output_dir="patch_output", patch_size=800, stride=400, extensions=(".jpg", ".jpeg", ".png")):
    """
    Process all images in a directory to create patches.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Base directory to save patches
        patch_size: Size of the patches (square)
        stride: Stride between patches
        extensions: Tuple of valid file extensions to process
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Process each image
    total_patches = 0
    failed_images = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            num_patches = save_patches(image_path, output_dir, patch_size, stride)
            total_patches += num_patches
            if num_patches == 0:
                failed_images.append(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            failed_images.append(image_path)
    
    print(f"Finished processing {len(image_files)} images. Created {total_patches} patches in total.")
    print(f"Patches saved to {os.path.abspath(output_dir)}")
    
    # Report on failed images
    if failed_images:
        print(f"\nWarning: Failed to process {len(failed_images)} images:")
        for failed in failed_images[:10]:  # Show first 10 failures
            print(f" - {failed}")
        if len(failed_images) > 10:
            print(f"   ... and {len(failed_images) - 10} more")
        
        # Save list of failed images to a file
        failed_log = os.path.join(output_dir, "failed_images.txt")
        with open(failed_log, 'w') as f:
            for failed in failed_images:
                f.write(f"{failed}\n")
        print(f"Full list of failed images saved to {failed_log}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create patches from images')
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('--output', '-o', default='patch_output', help='Output directory (default: patch_output)')
    parser.add_argument('--patch-size', '-p', type=int, default=800, help='Patch size (default: 800)')
    parser.add_argument('--stride', '-s', type=int, default=600, help='Stride between patches (default: 400)')
    parser.add_argument('--skip-errors', action='store_true', help='Skip errors and continue processing')
    
    args = parser.parse_args()
    
    # Suppress warnings if skip-errors is enabled
    if args.skip_errors:
        warnings.filterwarnings('ignore')
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single image
        os.makedirs(args.output, exist_ok=True)
        save_patches(args.input, args.output, args.patch_size, args.stride)
    elif os.path.isdir(args.input):
        # Process directory of images
        process_directory(args.input, args.output, args.patch_size, args.stride)
    else:
        print(f"Error: Input {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()
