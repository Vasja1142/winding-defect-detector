# src/prelabel.py
import os
import argparse
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import glob

def run_prelabeling(model_path: str, input_dir: str, output_dir: str, conf_threshold: float, imgsz: int, quiet: bool = False):
    """
    Runs inference on all images in a directory and saves the results as YOLO .txt files.
    """
    # --- 1. Load Model ---
    if not quiet:
        print(f"Loading model from: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}") # Errors should always be printed
        return

    # --- 2. Prepare Directories ---
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = glob.glob(os.path.join(input_path, '*.jpg'))
    if not quiet:
        print(f"Found {len(image_files)} images to process in {input_dir}.")

    # --- 3. Process Images ---
    
    # Use tqdm only if not in quiet mode
    image_iterator = tqdm(image_files, desc="Pre-labeling images") if not quiet else image_files

    for image_path_str in image_iterator:
        image_path = Path(image_path_str)
        
        # Run detection
        results = model(image_path, imgsz=imgsz, conf=conf_threshold, verbose=False)
        
        # Prepare output file path
        output_txt_path = output_path / (image_path.stem + '.txt')
        
        # Write results to .txt file
        with open(output_txt_path, 'w') as f:
            for box in results[0].boxes:
                # Get coordinates in xywhn format (normalized)
                xywhn = box.xywhn[0]
                # Get class id
                class_id = int(box.cls[0])
                
                # Write to file in YOLO format
                f.write(f"{class_id} {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]}\n")

    if not quiet:
        print("\n--- Pre-labeling complete! ---")
        print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch pre-labeling script using a trained YOLO model.")
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained .pt model.')
    parser.add_argument('--input_dir', required=True, type=str, help='Directory with images to label.')
    parser.add_argument('--output_dir', required=True, type=str, help='Directory to save the .txt label files.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference.')
    parser.add_argument('--quiet', action='store_true', help='Suppress all output except for errors.')
    
    args = parser.parse_args()
    
    run_prelabeling(args.model_path, args.input_dir, args.output_dir, args.conf, args.imgsz, args.quiet)
