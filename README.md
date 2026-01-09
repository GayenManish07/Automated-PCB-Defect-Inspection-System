# Automated Quality Inspection System for PCB

## Overview
This repository contains a Computer Vision prototype for **Automated Optical Inspection (AOI)** in manufacturing. It is designed to identify, classify, and assess the severity of defects on Printed Circuit Boards (PCBs).

Using the **Faster R-CNN** architecture (ResNet50 backbone), the system analyzes "Test" (defective) images against their "Template" (clean) counterparts. It outputs a comprehensive inspection report that includes bounding boxes, confidence scores, defect severity levels, and precise centroid coordinates for robotic actuation.

## Some sample outputs

### Without Augmentation during Training
Some results without using augmentation in training stage:

![Sample 1](Assets/Without%20Augmentation/sample_3.jpg)
*(Figure 1: Sample 1 - Side-by-side analysis without augmentation)*

![Sample 2](Assets/Without%20Augmentation/sample_9.jpg)
*(Figure 2: Sample 2 - Side-by-side analysis without augmentation)*

![Sample 3](Assets/Without%20Augmentation/sample_2.jpg)
*(Figure 3: Sample 3 - Side-by-side analysis without augmentation)*

### With Augmentation during Training
Some results after using augmentation in training stage:

![Sample 1](Assets/With%20Augmentation/sample_3.jpg)
*(Figure 4: Sample 1 - Side-by-side analysis WITH augmentation)*

![Sample 2](Assets/With%20Augmentation/sample_9.jpg)
*(Figure 5: Sample 2 - Side-by-side analysis WITH augmentation)*

![Sample 3](Assets/With%20Augmentation/sample_2.jpg)
*(Figure 6: Sample 3 - Side-by-side analysis WITH augmentation)*

## Key Features

The key features of the framework are as follows:

* **Defect Detection:** Identifies 6 standard PCB defect types: `open`, `short`, `mousebite`, `spur`, `copper`, `pin-hole`.
* **Severity Assessment:** Automatically maps defects to industry-standard action levels:
    * [CRITICAL] Open, Short (Functional failure)
    * [MAJOR] Mousebite, Spur (Structural weakness)
    * [MINOR] Copper, Pin-hole (Cosmetic)
* **Precision Localization:** Calculates and outputs (x, y) centroid coordinates for every defect which can then be used as per requirement.

## Dataset
This project uses the **[DeepPCB Dataset](https://github.com/tangsanli5201/DeepPCB)**.
* **Structure:** The code automatically pairs `_test.jpg` (defective) images with their corresponding `_temp.jpg` (template) images and `_not` (annotation) text files.
* **Preprocessing:** The dataset loader handles coordinate normalization and "sibling folder" traversal automatically.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GayenManish07/Automated-PCB-Defect-Inspection-System.git
    cd Automated-PCB-Defect-Inspection-System
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision opencv-python matplotlib pillow
    ```

3.  **Setup Data:**
    Download the DeepPCB dataset and ensure the following structure:
    ```text
    /path/to/DeepPCB/PCBData/
    ├── group00041/
    │   ├── 00041/ (images)
    │   └── 00041_not/ (annotations)
    ├── group12000/
    ...
    ```

## Usage

Run the main training and inspection script. You can enable or disable data augmentation using the command line arguments.

```bash
python main.py --data_root /path/to/PCBData --epochs 10 --batch_size 4 --use_aug
