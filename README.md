# Intelligent Document AI for Field Extraction

## Convolve Hackathon Submission

An end-to-end Document AI system engineered for automated field extraction, layout analysis, and verification from tractor loan quotations, invoices, and financial receipts. The pipeline combines a modular ensemble of fine-tuned YOLOv8 visual object detectors with a cascaded multilingual PaddleOCR engine and a domain-specific normalization subsystem.

---

## Performance Highlights

- **Fast-Path English Processing**: Standard English quotation processing completes in **2.0 seconds** per document end-to-end.
- **Multilingual Fallback Guarantee**: Complex, low-contrast, or multilingual documents (Devanagari/Hindi, Tamil, Telugu, Kannada) are processed through an exhaustive multi-tier language cascade within a maximum of **28.0 seconds**.
- **Low Compute Overhead**: Operates efficiently on standard CPU or single-GPU environments with an estimated compute cost of approximately **$0.002 per document**.
- **High-Precision Layout Decomposition**: Employs five dedicated YOLOv8 detectors fine-tuned on discrete document regions to eliminate inter-class interference and false positives.

---

## Core Capabilities

1. **Dealer Identification**: Localizes dealer header blocks using YOLOv8, performs multilingual OCR, and standardizes dealer entities.
2. **Tractor Make and Model Extraction**: Recognizes vehicle text and normalizes against a built-in master dictionary covering leading manufacturers (Swaraj, Mahindra, TAFE, Sonalika, Escorts, Farmtrac, Powertrac, Digitrac, John Deere, New Holland, Kubota, Force, Preet, VST, Captain, ACE, Indo Farm, HMT, Eicher, Standard).
3. **Horsepower (HP) Disambiguation**: Extracts horsepower ratings via regex matching (supporting both Latin `HP` and Devanagari `एचपी` markers) and cross-validates against manufacturer specifications in the master catalog.
4. **Asset Cost Parsing**: Detects numerical quotation amounts and handles currency symbols, comma separators, trailing delimiters (`/-`), and scanning artifacts.
5. **Physical Authentication Verification**: Detects the presence of authorized signatures and official dealer stamps/seals, outputting exact bounding box coordinates `[x1, y1, x2, y2]`.

---

## System Architecture Overview

The system executes a multi-stage pipeline decoupling visual spatial localization from text extraction and semantic validation:

```
[ Input Document (JPG/PNG) ]
             │
             ▼
[ YOLOv8 Modular Detection Ensemble ]
  ├── stamp.pt   ──> Presence & Spatial Bounding Box [x1, y1, x2, y2]
  ├── sign.pt    ──> Presence & Spatial Bounding Box [x1, y1, x2, y2]
  ├── dealer.pt  ──> Region of Interest (ROI) Crop
  ├── amount.pt  ──> Region of Interest (ROI) Crop
  └── model.pt   ──> Region of Interest (ROI) Crop
             │
             ▼
[ Cascaded Multilingual OCR Engine ]
  ├── Step 1: Fast English PaddleOCR (Angle Classifier Enabled)
  └── Step 2: Quality Score Evaluation -> Lazy-loaded Indic Cascade (Devanagari, Tamil, Telugu, Kannada)
             │
             ▼
[ Normalization & Disambiguation Subsystem ]
  ├── Asset Cost Sanitizer & Regex Extraction
  ├── HP Pattern Matcher & Catalog Disambiguation
  └── SequenceMatcher Fuzzy String Normalization
             │
             ▼
[ Structured JSON Output & Metric Aggregation ]
```

For comprehensive architectural flowcharts, component diagrams, mathematical scoring formulations, and latency breakdowns, refer to [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Repository Structure

```
Intelligent_Document_AI_for_Field_Extraction/
├── ARCHITECTURE.md                  # Detailed architectural design and flowcharts
├── README.md                        # Master project documentation
├── requirements.txt                 # Core Python dependencies
├── executable.py                    # Main inference script for folder-level batch processing
├── train.json                       # Ground truth annotations for dataset training and validation
├── amount.pt                        # Fine-tuned YOLOv8 model for Amount / Asset Cost detection
├── dealer.pt                        # Fine-tuned YOLOv8 model for Dealer Name block detection
├── model.pt                         # Fine-tuned YOLOv8 model for Tractor Model block detection
├── sign.pt                          # Fine-tuned YOLOv8 model for Authorized Signature detection
├── stamp.pt                         # Fine-tuned YOLOv8 model for Official Stamp/Seal detection
├── assets/
│   └── metrics/                     # Model training performance curves, matrices, and validation batches
│       ├── amount/                  # Evaluation metrics for Amount detector
│       ├── dealer/                  # Evaluation metrics for Dealer detector
│       ├── model/                   # Evaluation metrics for Tractor Model detector
│       ├── sign/                    # Evaluation metrics for Signature detector
│       └── stamp/                   # Evaluation metrics for Stamp detector
└── preparation/                     # Training scripts, annotation tools, and Kaggle notebooks
    ├── labeller.py                  # OpenCV desktop annotation tool for custom YOLO bounding box labeling
    ├── amount.ipynb                 # Training notebook for Amount detection model
    ├── dealer.ipynb                 # Training notebook for Dealer detection model
    ├── model.ipynb                  # Training notebook for Tractor Model detection model
    ├── sign.ipynb                   # Training notebook for Signature detection model
    └── stamp.ipynb                  # Training notebook for Stamp detection model
```

### File and Directory Descriptions

- **`executable.py`**: The primary production entry point. Loads all five YOLOv8 models, initializes the cascaded PaddleOCR engine, accepts an input directory path, runs inference across all images, outputs live JSON payloads to the console, and writes a consolidated `output.json` file.
- **`ARCHITECTURE.md`**: In-depth technical specification containing system flowcharts, OCR routing logic, fuzzy matching formulations, and latency trade-offs.
- **`requirements.txt`**: Pinned Python dependency definitions required to reproduce the environment.
- **`train.json`**: Reference annotations and dataset labels utilized during training and baseline validation.
- **`*.pt` (Model Weights)**: Pre-trained, fine-tuned PyTorch weight files for each of the five layout extraction tasks.
- **`preparation/labeller.py`**: Interactive Python/OpenCV annotation utility built to facilitate fast bounding box labeling on high-resolution quotation scans with automated screen height scaling and YOLO coordinate normalization.
- **`preparation/*.ipynb`**: End-to-end Jupyter notebooks detailing data loading, augmentation, hyperparameter selection, and YOLOv8 training routines executed in a GPU-accelerated cloud environment.
- **`assets/metrics/`**: Visual evidence and evaluation metrics generated during YOLOv8 model training and validation.

---

## Model Evaluation and Training Metrics

Each YOLOv8 detector was trained and validated independently on domain-specific splits. The figures below detail the training loss convergence, precision-recall dynamics, confusion matrices, and validation inferences.

### 1. Dealer Block Detection (`dealer.pt`)

The dealer detection model identifies the header region containing business registration and dealer identity information.

| Metric Curves | Precision-Recall & F1 Curves |
| :---: | :---: |
| ![Dealer Training Results](assets/metrics/dealer/results.png) | ![Dealer PR Curve](assets/metrics/dealer/BoxPR_curve.png) |
| **Training Loss & Metric Progression** | **Precision-Recall Curve** |

| Confusion Matrix | Sample Validation Predictions |
| :---: | :---: |
| ![Dealer Confusion Matrix](assets/metrics/dealer/confusion_matrix_normalized.png) | ![Dealer Validation Batch](assets/metrics/dealer/val_batch0_pred.jpg) |
| **Normalized Confusion Matrix** | **Predicted Bounding Boxes (Validation)** |

---

### 2. Tractor Model Detection (`model.pt`)

The model detection network isolates product specification lines containing brand designations, series numbers, and power metrics.

| Metric Curves | Precision-Recall & F1 Curves |
| :---: | :---: |
| ![Model Training Results](assets/metrics/model/results.png) | ![Model PR Curve](assets/metrics/model/BoxPR_curve.png) |
| **Training Loss & Metric Progression** | **Precision-Recall Curve** |

| Confusion Matrix | Sample Validation Predictions |
| :---: | :---: |
| ![Model Confusion Matrix](assets/metrics/model/confusion_matrix_normalized.png) | ![Model Validation Batch](assets/metrics/model/val_batch0_pred.jpg) |
| **Normalized Confusion Matrix** | **Predicted Bounding Boxes (Validation)** |

---

### 3. Asset Cost / Amount Detection (`amount.pt`)

The amount detector isolates the total quotation value, final invoice sum, and unit pricing fields.

| Metric Curves | Precision-Recall & F1 Curves |
| :---: | :---: |
| ![Amount Training Results](assets/metrics/amount/results.png) | ![Amount PR Curve](assets/metrics/amount/BoxPR_curve.png) |
| **Training Loss & Metric Progression** | **Precision-Recall Curve** |

| Confusion Matrix | Sample Validation Predictions |
| :---: | :---: |
| ![Amount Confusion Matrix](assets/metrics/amount/confusion_matrix_normalized.png) | ![Amount Validation Batch](assets/metrics/amount/val_batch0_pred.jpg) |
| **Normalized Confusion Matrix** | **Predicted Bounding Boxes (Validation)** |

---

### 4. Official Stamp / Seal Detection (`stamp.pt`)

The stamp detector recognizes inked dealer stamps and official organization seals across various colors and rotations.

| Metric Curves | Precision-Recall & F1 Curves |
| :---: | :---: |
| ![Stamp Training Results](assets/metrics/stamp/results.png) | ![Stamp PR Curve](assets/metrics/stamp/BoxPR_curve.png) |
| **Training Loss & Metric Progression** | **Precision-Recall Curve** |

| Confusion Matrix | Sample Validation Predictions |
| :---: | :---: |
| ![Stamp Confusion Matrix](assets/metrics/stamp/confusion_matrix_normalized.png) | ![Stamp Validation Batch](assets/metrics/stamp/val_batch0_pred.jpg) |
| **Normalized Confusion Matrix** | **Predicted Bounding Boxes (Validation)** |

---

### 5. Authorized Signature Detection (`sign.pt`)

The signature detection network identifies handwritten signatures of authorized signatories across diverse backgrounds.

| Precision-Recall Curve | Sample Validation Predictions |
| :---: | :---: |
| ![Signature PR Curve](assets/metrics/sign/BoxPR_curve.png) | ![Signature Validation Batch](assets/metrics/sign/val_batch1_pred.jpg) |
| **Precision-Recall Curve** | **Predicted Signatures (Validation)** |

---

## Installation and Environment Setup

### Prerequisites
- Python 3.8 or higher
- Windows, Linux, or macOS
- (Optional) NVIDIA CUDA-capable GPU for accelerated inference

### 1. Clone the Repository
```bash
git clone https://github.com/kajaredhruv433/Intelligent_Document_AI_for_Field_Extraction.git
cd Intelligent_Document_AI_for_Field_Extraction
```

### 2. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

Alternatively, install the core dependencies directly:
```bash
pip install ultralytics paddlepaddle paddleocr opencv-python
```

*Note: For GPU-accelerated OCR and object detection, install the corresponding CUDA-enabled build of `paddlepaddle-gpu` and `torch` according to your CUDA toolkit version.*

---

## Usage Guide

### Running Batch Document Extraction

Execute the main pipeline:
```bash
python executable.py
```

### Interactive Execution Workflow
1. Upon launch, the script initializes the YOLOv8 detector weights and the primary English OCR engine.
2. The terminal prompts for the target folder containing document images (`.jpg`, `.jpeg`, `.png`):
   ```text
   Enter folder path containing images:
   > C:\path\to\quotations
   ```
3. The system iterates through all images sequentially, rendering formatted JSON outputs to stdout in real time.
4. Upon batch completion, a consolidated JSON file named `output.json` is generated directly inside the target image directory.

---

## JSON Output Schema Specification

Each processed document yields a structured JSON object with the following schema:

```json
{
  "doc_id": "quotation_sample_01",
  "fields": {
    "dealer_name": "SHREE AGRO MOTORS PVT LTD",
    "model_name": "Swaraj 744 FE",
    "horse_power": 48,
    "asset_cost": 845000,
    "signature": {
      "present": true,
      "bbox": [820, 1405, 1050, 1480]
    },
    "stamp": {
      "present": true,
      "bbox": [766, 1387, 1103, 1599]
    }
  },
  "confidence": 0.88,
  "processing_time_sec": 2.14,
  "cost_estimate_usd": 0.002
}
```

### Field Definitions

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `doc_id` | `string` | Base filename of the processed document (without extension). |
| `fields.dealer_name` | `string` | Extracted and sanitized dealer name text. |
| `fields.model_name` | `string` or `null` | Standardized manufacturer brand and tractor model string. |
| `fields.horse_power` | `integer` or `null` | Extracted or catalog-inferred tractor horsepower rating. |
| `fields.asset_cost` | `integer` or `null` | Parsed quotation numerical total / invoice amount. |
| `fields.signature.present` | `boolean` | Indicates whether an authorized signature was detected. |
| `fields.signature.bbox` | `array[4]` or `null` | Pixel coordinates `[x1, y1, x2, y2]` of the signature bounding box. |
| `fields.stamp.present` | `boolean` | Indicates whether an official dealer stamp was detected. |
| `fields.stamp.bbox` | `array[4]` or `null` | Pixel coordinates `[x1, y1, x2, y2]` of the stamp bounding box. |
| `confidence` | `float` | Arithmetic mean of detection confidence scores across identified regions. |
| `processing_time_sec` | `float` | End-to-end wall-clock latency for the individual document (in seconds). |
| `cost_estimate_usd` | `float` | Estimated operational inference cost per document ($0.002). |

---

## Data Annotation Tooling (`preparation/labeller.py`)

To expedite training data preparation, the repository includes a custom OpenCV desktop labeling utility:
- **Automatic Display Scaling**: Calculates desktop viewport height via Windows API (`ctypes.windll.user32`) and scales high-resolution scans down to 90% screen height to avoid multi-monitor clipping.
- **Normalized Coordinate Output**: Converts interactive ROI coordinates back to original native resolution and formats bounding boxes into YOLO standard format (`<class_id> <x_center> <y_center> <width> <height>`).
- **Keyboard Shortcuts**: `ENTER` to confirm and persist label, `ESC` to skip corrupted or ambiguous images.

---

## Author and Project Metadata

- **Author**: Dhruv Kajare
- **Event**: Convolve Hackathon
- **Domain**: Intelligent Document AI / Automated Financial Document Extraction
