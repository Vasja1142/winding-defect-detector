# Winding Defect Detector: Project Overview

This document provides a comprehensive overview for developers interacting with the Winding Defect Detector project via the Gemini CLI.

## 1. Core Technologies

-   **Model**: [YOLOv12](https://github.com/ultralytics/ultralytics) is used for object detection.
-   **Experiment Tracking**: [MLflow](https://mlflow.org/) is used to log experiments, parameters, metrics, and artifacts.
-   **Deployment**: The entire environment is containerized using [Docker](https://www.docker.com/) and managed with [Docker Compose](https://docs.docker.com/compose/).
-   **Web Service**: [Flask](https://flask.palletsprojects.com/) is used to create a simple web server for real-time inference.
-   **Core Libraries**: `ultralytics`, `torch`, `opencv-python-headless`, `numpy`.

## 2. Project Workflow

The project follows a standard machine learning pipeline, from data preparation to model deployment.

### Step 1: Data Preparation (`run_data_prep.py`)

This is the main script to start the data preparation pipeline. It orchestrates the following sub-steps:

1.  **Sanitize Filenames (`sanitize_filenames.py`)**: Cleans up video filenames in `data/01_raw`, converting them to a standardized format (lowercase, latin characters, underscores instead of spaces). This prevents issues in subsequent steps.
2.  **Exclude Test Video**: A specific test video (defined in the script) is excluded from the training data.
3.  **Extract Frames (`data_processing.py`)**: The remaining training videos are processed. It extracts frames at a specified interval (`FRAME_SKIP`) and saves them to `data/02_processed/frames`. Each frame is prefixed with its source video name to maintain traceability.

**Command:**
```bash
docker compose exec detector python3 src/run_data_prep.py
```

### Step 2: Annotation (Manual Step)

The extracted frames in `data/02_processed/frames` must be annotated manually using a labeling tool like CVAT or LabelImg. The annotations should be exported in YOLO format.

### Step 3: Create Verification Set (`create_verification_set.py`)

After annotation, this script is used to create a balanced "golden" dataset for verification. It copies all frames that have annotations (positive samples) and a subset of frames that do not (negative samples, sampled at `neg_sample_rate`). For negative samples, it creates an empty `.txt` file, which is the correct format for YOLO.

**Command:**
```bash
docker compose exec detector python3 src/create_verification_set.py --input_dir <path_to_cvat_export> --output_dir <path_to_verification_set>
```

### Step 4: Prepare Final Dataset (`prepare_dataset.py`)

This script takes the verified and cleaned annotations and prepares the final dataset structure required by YOLO. It splits the data into `train` and `valid` sets based on the specified validation split ratio. **Crucially, it now processes ALL files found in the `--label_dir`, assuming they are verified and ready for dataset creation.**

**Command:**
```bash
docker compose exec detector python3 src/prepare_dataset.py --image_dir /app/data/02_processed/frames --label_dir <path_to_verified_labels> --output_dir /app/data/04_datasets
```

### Step 5: Model Training (`train.py`)

This script handles the model training process.

-   It uses the `config.yaml` file to find the training and validation data.
-   It logs all training parameters, metrics, and artifacts (like the best model weights and performance graphs) to the MLflow server.
-   It supports a wide range of data augmentation and optimizer hyperparameters.

**Key Arguments for Training:**
-   `--model`: Path to initial model weights (e.g., `yolo12n.pt` or `/app/data/05_runs/experiment/weights/last.pt`).
-   `--epochs`: Number of training epochs.
-   `--batch`: Batch size.
-   `--imgsz`: Image size for training.
-   `--augment`: Flag to enable data augmentation.
-   `--degrees`, `--translate`, `--scale`, `--fliplr`, `--hsv_h`, `--hsv_s`, `--hsv_v`, `--mosaic`, `--close_mosaic`, `--copy_paste`, `--shear`, `--perspective`: Various augmentation parameters.
-   `--optimizer`: Optimizer to use (e.g., `SGD`, `Adam`).
-   `--cos_lr`: Use cosine learning rate scheduler.
-   `--patience`: Early stopping patience.

**Command:**
```bash
# Start training from a pretrained model
docker compose exec detector python3 src/train.py --model yolo12n.pt --epochs 100 --batch 64 --augment --degrees 10 --hsv_s 0.2

# Resume training from a checkpoint with Adam optimizer and early stopping
docker compose exec detector python3 src/train.py --model /app/data/05_runs/experiment/weights/last.pt --epochs 50 --optimizer Adam --patience 30
```

### Step 6: Inference

There are two ways to perform inference:

#### A. Real-time Streaming Server (`inference_server.py`)

This script launches a Flask web server that streams video from a specified source (file or webcam) with bounding boxes drawn in real-time.

**Command:**
```bash
docker compose exec detector python3 src/inference_server.py --model_path <path_to_best.pt> --source /path/to/video.mp4 --imgsz 640 --confidence 0.7
```
The stream will be available at `http://<YOUR_IP>:5000/video_feed`.

#### B. Create Labeled Video (`create_labeled_video.py`)

This script processes an entire video file at once and saves a new video file with the detection bounding boxes drawn on it. This is useful for creating offline demos or analyzing a full video recording.

**Command:**
```bash
docker compose exec detector python3 src/create_labeled_video.py --model_path <path_to_best.pt> --input_video <input.mp4> --output_video <output.mp4>
```

## 3. Key Configuration Files

-   **`config.yaml`**: Defines the paths to the training/validation data and the class names for the YOLO model. This file is crucial for the `train.py` script.
-   **`docker-compose.yml`**: Defines the `detector` service and the `mlflow-server` service. It maps local data directories into the container for persistent storage.
-   **`Dockerfile`**: Specifies the Docker image for the `detector` service, including all necessary dependencies from `requirements.txt`.
-   **`requirements.txt`**: Lists all Python dependencies required for the project.

## 4. Utility Scripts

This section describes additional utility scripts located in the `src/` and `src/utils/` directories, which provide various functionalities for data preparation, manipulation, and pre-labeling.

### `src/prelabel.py`

This script allows you to automatically generate initial YOLO annotations (`.txt` files) for a batch of images using a trained model. This is extremely useful for speeding up the manual annotation process in tools like CVAT, as annotators only need to correct existing boxes rather than drawing them from scratch.

**Functionality:**
-   Loads a trained YOLO model.
-   Iterates through all `.jpg` images in a specified input directory.
-   Performs inference on each image.
-   Saves the detected bounding boxes (class ID, normalized x_center, y_center, width, height) into corresponding `.txt` files in the YOLO format in an output directory.

**Command:**
```bash
docker compose exec detector python3 src/prelabel.py --model_path /app/data/05_runs/WDD_v1/weights/best.pt --input_dir /app/data/02_processed/frames --output_dir /app/data/06_prelabeled --conf 0.25 --imgsz 640
```
*   `--model_path`: Path to your trained model (e.g., `WDD_v1/weights/best.pt`).
*   `--input_dir`: Directory containing images to be pre-labeled (e.g., `data/02_processed/frames`).
*   `--output_dir`: Directory where the generated `.txt` label files will be saved.
*   `--conf`: Confidence threshold for detections to be included in the labels.
*   `--imgsz`: Image size for inference.

### `src/utils/augment_offline.py`

This script performs offline data augmentation on a dataset of images and their corresponding YOLO labels. It applies transformations like blurring and Gaussian noise, creating augmented copies of the original data. This is useful for expanding your dataset and improving model robustness.

**Functionality:**
-   Reads images and their `.txt` labels from an input directory.
-   Applies a combination of blurring (Gaussian or Median) and controlled Gaussian noise.
-   Saves the augmented images and copies their original labels (as the bounding box coordinates remain the same relative to the image).

**Command:**
```bash
docker compose exec detector python3 src/utils/augment_offline.py --input_dir /app/data/03_annotations_raw/original_frames --output_dir /app/data/03_annotations_raw/augmented_frames --sigma 5.0 --noise_p 0.9
```
*   `--input_dir`: Directory with original images and labels.
*   `--output_dir`: Directory to save augmented images and copied labels.
*   `--sigma`: Standard deviation for Gaussian noise (strength of noise).
*   `--noise_p`: Probability of applying Gaussian noise to an image.

### `src/utils/merge_classes.py`

This script simplifies a multi-class YOLO dataset by merging all existing classes into a single, new class (default ID 0). This can be useful if you initially annotated with multiple defect types but now only care about detecting "any defect."

**Functionality:**
-   Reads all `.txt` label files from an input directory.
-   For every bounding box, it changes the class ID to a specified `new_class_id` (default 0).
-   Copies the corresponding images.
-   Saves the modified labels and images to an output directory.

**Command:**
```bash
docker compose exec detector python3 src/utils/merge_classes.py --input_dir /app/data/04_datasets/train --output_dir /app/data/04_datasets/train_single_class
```
*   `--input_dir`: Directory with the multi-class YOLO dataset.
*   `--output_dir`: Directory to save the single-class dataset.

### `src/utils/remove_duplicate_boxes.py`

This script is designed to clean up CVAT XML annotation files by identifying and removing mass-duplicated bounding boxes. This can happen due to errors during annotation export or copy-paste operations. It specifically looks for the *most common* duplicate box signature (label + coordinates) and removes all instances of it.

**Functionality:**
-   Parses a CVAT XML file.
-   Identifies the bounding box signature (label and coordinates) that appears most frequently.
-   If this "super-clone" appears more than a threshold (currently 10 times), it removes all instances of it.
-   Saves the cleaned XML to a new file.

**Command:**
```bash
docker compose exec detector python3 src/utils/remove_duplicate_boxes.py --input_xml /app/data/03_annotations_raw/cvat_export_with_duplicates.xml --output_xml /app/data/03_annotations_raw/cvat_export_cleaned.xml
```
*   `--input_xml`: Path to the input CVAT XML file.
*   `--output_xml`: Path to save the cleaned CVAT XML file.

### `src/utils/smart_merge.py`

This script provides a more flexible class merging strategy than `merge_classes.py`. It allows you to specify one class to *keep* its original identity (though its ID might change to 0), while all other classes are merged into a single "defect" class (ID 1). This is useful for scenarios where one specific defect type is of primary interest, and all others are considered generic defects.

**Functionality:**
-   Reads images and `.txt` labels from an input directory, along with `obj.names`.
-   Identifies the specified `keep_label_name`.
-   Assigns a new class ID (0 by default) to the `keep_label_name`.
-   Assigns a different new class ID (1 by default) to all other classes.
-   Creates a new `obj.names` file reflecting the new class mapping.
-   Saves the modified labels and images to an output directory.

**Command:**
```bash
docker compose exec detector python3 src/utils/smart_merge.py --input_dir /app/data/04_datasets/train --output_dir /app/data/04_datasets/train_smart_merged --keep_label_name row_gap
```
*   `--input_dir`: Directory with the original multi-class YOLO dataset.
*   `--output_dir`: Directory to save the smart-merged dataset.
*   `--keep_label_name`: The name of the class to keep separate (e.g., `row_gap`).

### `src/utils/yolo_to_cvat_xml.py`

This script converts a dataset in YOLO format (images, `.txt` label files, and `obj.names`) into a single CVAT 1.1 XML annotation file. This is useful if you have YOLO-formatted annotations (e.g., generated by `prelabel.py` or another tool) and want to import them into CVAT for review or further manual correction.

**Functionality:**
-   Reads `obj.names` to get class mappings.
-   Iterates through images and their corresponding `.txt` label files.
-   Converts YOLO normalized bounding box coordinates to CVAT absolute pixel coordinates (xtl, ytl, xbr, ybr).
-   Generates a single CVAT 1.1 XML file containing all image and box annotations.

**Command:**
```bash
docker compose exec detector python3 src/utils/yolo_to_cvat_xml.py --input_dir /app/data/06_prelabeled --output_file /app/data/06_prelabeled/prelabeled_cvat.xml
```
*   `--input_dir`: Directory containing the YOLO dataset (images, `.txt` labels, `obj.names`).
*   `--output_file`: Path to save the generated CVAT XML file.
