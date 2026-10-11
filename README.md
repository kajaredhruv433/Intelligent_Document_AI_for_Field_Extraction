# Intelligent Document AI for Field Extraction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff?logo=yolo&logoColor=white" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/OCR-PaddleOCR-red?logo=baidu&logoColor=white" alt="PaddleOCR" />
  <img src="https://img.shields.io/badge/Computer%20Vision-OpenCV-green?logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Event-Convolve%20Hackathon-orange" alt="Hackathon" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" alt="Status" />
</p>

An enterprise-grade, end-to-end Document AI system engineered for automated field extraction, layout analysis, and authentication verification from tractor loan quotations, dealer invoices, and financial receipts. The system pairs a modular ensemble of five fine-tuned YOLOv8 computer vision detectors with a cascaded multilingual PaddleOCR engine and a domain-specific normalization and catalog disambiguation subsystem.

---

## Executive Summary & Highlights

Processing tractor quotations in Indian rural banking and agritech lending presents unique technical bottlenecks: noisy physical scans, non-standard tabular templates, regional multilingual scripts (Devanagari/Hindi, Tamil, Telugu, Kannada), overlapping ink seals, and handwritten signatures. 

This repository implements a modular, high-throughput pipeline designed to overcome these challenges with optimal latency and minimal compute cost:

- Ultra-Fast English Processing: Clean English quotation processing completes in ~2.0 seconds per document end-to-end.
- Multilingual Fallback Guarantee: Complex, low-contrast, or Indic-script quotations automatically cascade through lazy-loaded regional OCR models within a maximum envelope of 28.0 seconds.
- Ultra-Low Compute Overhead: Runs efficiently on standard CPU or entry-level single GPU environments, operating at approximately $0.002 per document.
- Decoupled Modular Detectors: Employs five independent YOLOv8 detectors fine-tuned on isolated document regions, eliminating cross-class interference and aspect-ratio conflicts.
- Physical Verification: Pinpoints authorized signatures and official dealer stamps with precise pixel-level bounding boxes [x1, y1, x2, y2].
- Domain-Aware Entity Normalization: Resolves tractor makes, models, and horsepower ratings against a comprehensive catalog of 20+ leading Indian manufacturers.

---

## System Architecture

The pipeline decouples spatial layout localization from multilingual optical character recognition and semantic normalization.

### End-to-End System Architecture Diagram

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Ingestion and Preprocessing"]
        A["Input Document Image (JPG / PNG)"] --> B["OpenCV Ingestion & Resolution Validation"]
    end

    subgraph DetectionEnsemble["2. YOLOv8 Modular Layout Decomposition Ensemble"]
        B --> D1["YOLOv8 Stamp Detector (stamp.pt)"]
        B --> D2["YOLOv8 Signature Detector (sign.pt)"]
        B --> D3["YOLOv8 Dealer Block Detector (dealer.pt)"]
        B --> D4["YOLOv8 Amount Block Detector (amount.pt)"]
        B --> D5["YOLOv8 Model Block Detector (model.pt)"]
    end

    subgraph SpatialAnalysis["3. Spatial Verification & ROI Crop Extraction"]
        D1 --> E1["Stamp Presence & Bounding Box [x1, y1, x2, y2]"]
        D2 --> E2["Signature Presence & Bounding Box [x1, y1, x2, y2]"]
        D3 --> E3["Dealer ROI Crop Extraction"]
        D4 --> E4["Amount ROI Crop Extraction"]
        D5 --> E5["Model ROI Crop Extraction"]
    end

    subgraph OCRSubsystem["4. Cascaded Multilingual OCR Engine"]
        E3 --> F1["English PaddleOCR (Angle Classifier Enabled)"]
        E4 --> F2["English PaddleOCR (Angle Classifier Enabled)"]
        E5 --> F3["English PaddleOCR (Angle Classifier Enabled)"]
        
        F1 --> G1{"Quality Score >= 20?"}
        F2 --> G2{"Quality Score >= 20?"}
        F3 --> G3{"Quality Score >= 20?"}
        
        G1 -- "No (Low Alphanumeric Quality)" --> H1["Lazy-Loaded Indic Cascade (Hindi, Tamil, Telugu, Kannada)"]
        G2 -- "No (Low Alphanumeric Quality)" --> H2["Lazy-Loaded Indic Cascade (Hindi, Tamil, Telugu, Kannada)"]
        G3 -- "No (Low Alphanumeric Quality)" --> H3["Lazy-Loaded Indic Cascade (Hindi, Tamil, Telugu, Kannada)"]
        
        G1 -- "Yes (Fast Path: ~2.0s)" --> I1["Raw Dealer Text Candidate"]
        G2 -- "Yes (Fast Path: ~2.0s)" --> I2["Raw Amount Text Candidate"]
        G3 -- "Yes (Fast Path: ~2.0s)" --> I3["Raw Model Text Candidate"]
        
        H1 --> I1
        H2 --> I2
        H3 --> I3
    end

    subgraph Normalization["5. Normalization & Disambiguation Subsystem"]
        I1 --> J1["Dealer Name Cleaning & Case Standardization"]
        I2 --> J2["Currency & Regex Number Sanitizer"]
        I3 --> J3["HP Extraction & Catalog Fuzzy Matching"]
        
        J2 --> K1["Sanitized Integer Asset Cost"]
        J3 --> K2["Canonical Brand & Model Name"]
        J3 --> K3["Validated Horse Power (HP)"]
    end

    subgraph Aggregation["6. Payload Synthesis & Export"]
        E1 --> L["JSON Payload Synthesizer"]
        E2 --> L
        J1 --> L
        K1 --> L
        K2 --> L
        K3 --> L
        L --> M["Console Live Stream & output.json Export"]
    end
```

---

## Detailed Pipeline Breakdown

### 1. Vision and Layout Decomposition Ensemble

Monolithic object detection models struggle on document analysis due to severe class imbalance, extreme aspect-ratio discrepancies (e.g., elongated dealer headers vs. circular stamps vs. irregular signatures), and inconsistent visual features.

To resolve this, the system deploys five dedicated YOLOv8 Nano detectors:

| Model Checkpoint | Target Entity | Ingestion Mode | Output Representation | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `stamp.pt` | Official Stamp / Seal | Full Document | `bbox [x1, y1, x2, y2]`, `present: bool` | Validates dealer authentication seal |
| `sign.pt` | Authorized Signature | Full Document | `bbox [x1, y1, x2, y2]`, `present: bool` | Verifies presence of authorized signatory |
| `dealer.pt` | Dealer Header Block | Full Document | High-resolution cropped image array | Isolates business name and registration |
| `amount.pt` | Asset Cost / Quotation Total | Full Document | High-resolution cropped image array | Isolates total pricing and invoice values |
| `model.pt` | Tractor Model Specification | Full Document | High-resolution cropped image array | Isolates tractor make, variant, and HP specs |

```
Detection Configuration:
- Confidence Threshold: conf = 0.25
- Non-Maximum Suppression (NMS): Native IoU filtering per model
- Coordinate Representation: Standardized 4-point bounding boxes [x1, y1, x2, y2]
```

---

### 2. Cascaded Multilingual OCR Engine

The OCR engine employs a dual-speed cascading topology that prioritizes high throughput while ensuring robust coverage across non-Latin Indian scripts:

```mermaid
flowchart TD
    Crop["Incoming ROI Image Crop"] --> EnOCR["Execute English PaddleOCR (cls=True)"]
    EnOCR --> Eval["Compute Quality Score: Score = len(Text) x (AlphanumericCount / len(Text))"]
    Eval --> Check{"Quality Score >= 20?"}
    
    Check -- "Yes" --> FastReturn["Fast Path Exit (~2.0s): Return English Text Candidate"]
    Check -- "No" --> IndicInit["Trigger Lazy-Loaded Indic Fallback Cascade"]
    
    IndicInit --> L1["Devanagari (Hindi) PaddleOCR"]
    L1 --> L2["Tamil (ta) PaddleOCR"]
    L2 --> L3["Telugu (te) PaddleOCR"]
    L3 --> L4["Kannada (ka) PaddleOCR"]
    
    L4 --> SelectBest["Evaluate Alphanumeric Density & Select Highest Scoring Text"]
    SelectBest --> IndicReturn["Multilingual Exit (<=28.0s): Return Optimal Text"]
```

#### Alphanumeric Heuristic Scoring
The heuristic score determines whether the English OCR extraction was successful or if the crop contains regional script:

$$\text{Quality Score}(T) = \text{len}(T) \times \left( \frac{\sum_{c \in T} \mathbb{I}[c \in \text{Alphanumeric}]}{\max(\text{len}(T), 1)} \right)$$

1. Fast-Path: If the English OCR output produces a Quality Score >= 20, the pipeline returns immediately in ~2.0 seconds.
2. Lazy-Loaded Indic Cascade: If the score is below 20 (indicating low character confidence or non-English text), the engine sequentially evaluates devanagari, ta (Tamil), te (Telugu), and ka (Kannada), returning the highest scoring candidate.

---

### 3. Normalization & Disambiguation Subsystem

Raw OCR strings frequently contain scanning artifacts, currency glyphs, and OCR character substitutions. The normalization subsystem cleans and validates all extracted values:

```mermaid
flowchart LR
    subgraph ModelHP["Tractor Model & HP Disambiguation"]
        RawModel["Raw Model OCR Text"] --> RegexHP["Regex HP Matcher: r'\b(\d{2})\s*(HP|एचपी)\b'"]
        RawModel --> SanitizeModel["String Sanitization & Alphanumeric Tokenization"]
        SanitizeModel --> FuzzyMatch["SequenceMatcher vs Master Catalog (TRACTOR_MODELS)"]
        FuzzyMatch --> TopMatch["Best Canonical Brand + Model"]
        RegexHP --> DecideHP{"Explicit HP Detected?"}
        TopMatch --> DecideHP
        DecideHP -- "Yes" --> FinalHP["Assign Detected HP"]
        DecideHP -- "No" --> CatalogHP["Assign Catalog Default HP Range"]
    end

    subgraph AmountEngine["Asset Cost Parser"]
        RawAmt["Raw Amount OCR Text"] --> CleanAmt["Filter Currency Signs, '/-', Commas & Decimal Dots"]
        CleanAmt --> RegexAmt["Regex Matcher: r'\b(\d{4,7})(?:\.\d{1,2})?\b'"]
        RegexAmt --> IntAmt["Cast to Integer Currency Amount"]
    end
```

#### Master Tractor Catalog Disambiguation
The system includes an extensive built-in knowledge base of over 20 top tractor manufacturers operating in India:
- Swaraj: Code, Target 630, 717, 724 XM, 735 FE, 744 FE, 744 XT, 855 FE, 963 FE, etc.
- Mahindra: OJA Series, Yuvraj, Jivo, XP Plus, SP Plus, Yuvo Tech+, Arjun Novo.
- TAFE: Orchard Plus, Dynatrack, 42 DI, 5900 DI, etc.
- Sonalika: Tiger, Sikander, Worldtrac series.
- Escorts / Farmtrac / Powertrac / Digitrac
- John Deere: 5000 D & E series.
- New Holland: Simba, Excel, 3000 & 5000 series.
- Other OEMs: Kubota, Force, Preet, VST Shakti, Captain, ACE, Indo Farm, HMT, Eicher, Standard.

Fuzzy matching is computed using the Ratcliff-Obershelp similarity metric via Python's difflib.SequenceMatcher:

$$\text{Similarity}(S_1, S_2) = \frac{2 \times M}{|S_1| + |S_2|}$$

where M is the number of matching characters within non-overlapping maximal common substrings.

---

## Runtime Performance & Cost Profile

| Metric | Fast-Path English | Indic Fallback Cascade | Monolithic Layout Model |
| :--- | :--- | :--- | :--- |
| End-to-End Latency | ~2.0 seconds | 6.0 – 28.0 seconds | 12.0 – 35.0 seconds |
| Detection Quality | High precision (90%+) | High precision (90%+) | Prone to class confusion |
| Language Coverage | English | English + Hindi, Tamil, Telugu, Kannada | Single-language bias |
| Inference Cost | $0.002 / doc | $0.002 / doc | $0.015+ / doc |
| Memory Footprint | ~1.2 GB RAM | ~2.4 GB RAM (on-demand loading) | > 4.5 GB RAM |

### Overall Document Confidence Formulation
The aggregate confidence returned for each document represents the arithmetic mean of all YOLOv8 bounding box confidence scores detected across the active models:

$$\text{Document Confidence} = \frac{1}{N} \sum_{i=1}^{N} \text{conf}_i$$

---

## Model Evaluation & Training Metrics

Each YOLOv8 Nano model was trained and validated independently on specialized domain splits. The training curves, precision-recall charts, and validation predictions demonstrate high convergence and spatial precision.

### 1. Dealer Block Detection (`dealer.pt`)
Identifies business headers, company letters, and dealer entity sections.

| Training Progression | Precision-Recall & F1 Curves |
| :---: | :---: |
| ![Dealer Training Results](assets/metrics/dealer/results.png) | ![Dealer PR Curve](assets/metrics/dealer/BoxPR_curve.png) |
| Training Loss & Metric Progression | Precision-Recall Curve |

| Normalized Confusion Matrix | Sample Validation Inferences |
| :---: | :---: |
| ![Dealer Confusion Matrix](assets/metrics/dealer/confusion_matrix_normalized.png) | ![Dealer Validation Batch](assets/metrics/dealer/val_batch0_pred.jpg) |
| Normalized Confusion Matrix | Predicted Bounding Boxes |

---

### 2. Tractor Model Detection (`model.pt`)
Isolates product specification blocks containing vehicle names, series, and horsepower markers.

| Training Progression | Precision-Recall & F1 Curves |
| :---: | :---: |
| ![Model Training Results](assets/metrics/model/results.png) | ![Model PR Curve](assets/metrics/model/BoxPR_curve.png) |
| Training Loss & Metric Progression | Precision-Recall Curve |

| Normalized Confusion Matrix | Sample Validation Inferences |
| :---: | :---: |
| ![Model Confusion Matrix](assets/metrics/model/confusion_matrix_normalized.png) | ![Model Validation Batch](assets/metrics/model/val_batch0_pred.jpg) |
| Normalized Confusion Matrix | Predicted Bounding Boxes |

---

### 3. Asset Cost / Amount Detection (`amount.pt`)
Localizes total quotation figures, base price lines, and final payable values.

| Training Progression | Precision-Recall & F1 Curves |
| :---: | :---: |
| ![Amount Training Results](assets/metrics/amount/results.png) | ![Amount PR Curve](assets/metrics/amount/BoxPR_curve.png) |
| Training Loss & Metric Progression | Precision-Recall Curve |

| Normalized Confusion Matrix | Sample Validation Inferences |
| :---: | :---: |
| ![Amount Confusion Matrix](assets/metrics/amount/confusion_matrix_normalized.png) | ![Amount Validation Batch](assets/metrics/amount/val_batch0_pred.jpg) |
| Normalized Confusion Matrix | Predicted Bounding Boxes |

---

### 4. Official Stamp / Seal Detection (`stamp.pt`)
Recognizes circular, oval, and rectangular inked dealer stamps across varied ink colors (blue, red, black).

| Training Progression | Precision-Recall & F1 Curves |
| :---: | :---: |
| ![Stamp Training Results](assets/metrics/stamp/results.png) | ![Stamp PR Curve](assets/metrics/stamp/BoxPR_curve.png) |
| Training Loss & Metric Progression | Precision-Recall Curve |

| Normalized Confusion Matrix | Sample Validation Inferences |
| :---: | :---: |
| ![Stamp Confusion Matrix](assets/metrics/stamp/confusion_matrix_normalized.png) | ![Stamp Validation Batch](assets/metrics/stamp/val_batch0_pred.jpg) |
| Normalized Confusion Matrix | Predicted Bounding Boxes |

---

### 5. Authorized Signature Detection (`sign.pt`)
Detects authorized agent and customer signatures across clear, stamped, or patterned invoice backgrounds.

| Precision-Recall Dynamics | Sample Validation Inferences |
| :---: | :---: |
| ![Signature PR Curve](assets/metrics/sign/BoxPR_curve.png) | ![Signature Validation Batch](assets/metrics/sign/val_batch1_pred.jpg) |
| Precision-Recall Curve | Predicted Signatures |

---

## Repository Structure

```
Intelligent_Document_AI_for_Field_Extraction/
├── ARCHITECTURE.md                  # Detailed architectural design and engineering specifications
├── README.md                        # Master project documentation
├── requirements.txt                 # Core Python runtime dependencies
├── executable.py                    # Production entry point for batch document inference
├── train.json                       # Ground truth annotations for dataset training and baseline
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
└── preparation/                     # Training notebooks, annotation tools, and dataset utilities
    ├── labeller.py                  # OpenCV desktop annotation tool for custom YOLO bounding box labeling
    ├── amount.ipynb                 # Cloud training notebook for Amount detector
    ├── dealer.ipynb                 # Cloud training notebook for Dealer detector
    ├── model.ipynb                  # Cloud training notebook for Tractor Model detector
    ├── sign.ipynb                   # Cloud training notebook for Signature detector
    └── stamp.ipynb                  # Cloud training notebook for Stamp detector
```

---

## Installation & Setup

### Prerequisites
- Python 3.8, 3.9, 3.10, or 3.11
- Operating System: Windows, Linux, or macOS
- (Optional) NVIDIA CUDA-compatible GPU (Compute Capability 7.0+) for accelerated inference

### 1. Clone the Repository
```bash
git clone https://github.com/kajaredhruv433/Intelligent_Document_AI_for_Field_Extraction.git
cd Intelligent_Document_AI_for_Field_Extraction
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

Direct installation alternative:
```bash
pip install ultralytics paddlepaddle paddleocr opencv-python
```

> [!NOTE]
> For GPU acceleration, install the CUDA-enabled builds of paddlepaddle-gpu and torch matching your local CUDA toolkit version.

---

## Usage Guide

### Running Batch Processing
Execute the production inference pipeline:

```bash
python executable.py
```

### Execution Flow
1. Model Initialization: The script loads all five YOLOv8 checkpoints (stamp.pt, sign.pt, dealer.pt, amount.pt, model.pt) and pre-warms the primary English PaddleOCR engine.
2. Directory Selection: Enter the folder containing target quotation scans (.jpg, .jpeg, .png):
   ```text
   📂 Enter folder path containing images:
   > C:\Users\Juhi\Documents\Quotations
   ```
3. Real-Time Streaming: The pipeline prints live JSON objects to the console as each document completes processing.
4. Export: Upon completion, a unified output.json file is saved directly into the target folder.

---

## JSON Output Schema Specification

Each processed quotation generates a structured output formatted as follows:

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

### Schema Field Definitions

| Field Path | Type | Nullable | Description |
| :--- | :--- | :---: | :--- |
| `doc_id` | `string` | No | Base file identifier of the quotation image (excluding extension). |
| `fields.dealer_name` | `string` | Yes | Extracted and sanitized dealer corporate name. |
| `fields.model_name` | `string` | Yes | Normalized canonical brand and tractor variant. |
| `fields.horse_power` | `integer` | Yes | Extracted or catalog-inferred vehicle horsepower (HP). |
| `fields.asset_cost` | `integer` | Yes | Parsed total invoice amount / quotation sum. |
| `fields.signature.present` | `boolean` | No | `true` if an authorized signature was detected; `false` otherwise. |
| `fields.signature.bbox` | `int[4]` | Yes | Absolute pixel coordinates `[x1, y1, x2, y2]` of the signature box. |
| `fields.stamp.present` | `boolean` | No | `true` if an official stamp/seal was detected; `false` otherwise. |
| `fields.stamp.bbox` | `int[4]` | Yes | Absolute pixel coordinates `[x1, y1, x2, y2]` of the stamp box. |
| `confidence` | `float` | No | Arithmetic mean of detection confidences across identified regions. |
| `processing_time_sec` | `float` | No | Wall-clock latency for the individual document in seconds. |
| `cost_estimate_usd` | `float` | No | Operational compute cost estimate ($0.002 per document). |

---

## Dataset Annotation Tool (`preparation/labeller.py`)

To streamline data labeling on 4K/high-resolution mobile camera quotation scans, the project includes an interactive annotation tool:

- Dynamic Display Rescaling: Automatically queries the Windows API (ctypes.windll.user32) to fit images to 90% of screen height without distortion or window cropping.
- YOLO Coordinate Conversion: Automatically maps mouse ROI drags on the resized viewport back to native high-resolution coordinates and formats them into normalized YOLO annotation format (<class_id> <x_center> <y_center> <width> <height>).
- Rapid Navigation: Press ENTER / SPACE to persist annotation and advance; press ESC to skip ambiguous or corrupted images.

---

## Authors & Acknowledgments

- Lead Developer: Dhruv Kajare
- Hackathon: Convolve Hackathon Submission
- Domain: Intelligent Document Processing (IDP) / Agrifinance Document AI
