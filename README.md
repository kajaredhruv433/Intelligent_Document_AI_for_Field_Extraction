# Intelligent Document AI for Field Extraction

## 🏆 Hackathon Submission – Convolve

This repository hosts an **end-to-end Document AI system** designed to automate information extraction from **tractor loan quotations** and similar invoice-like documents.

The solution leverages a hybrid pipeline of **YOLOv8** for layout analysis (object detection) and **PaddleOCR** for robust multilingual text extraction.

---

## 🚀 Key Features

- **Multilingual Support**: Handling English as well as Indic languages (Hindi/Devanagari, Tamil, Telugu, Kannada).
- **Field Extraction**:
  - **Dealer Name**: Extracted using OCR and validated via fuzzy matching.
  - **Model Name**: Normalized against a comprehensive dictionary of tractor models (Swaraj, Mahindra, Sonalika, etc.).
  - **Horse Power (HP)**: Extracted and cross-referenced with the model database.
  - **Asset Cost**: Parser handles currency formats and noise.
  - **Dealer Signature & Stamp**: Presence detection and bounding box localization using custom YOLO models.
- **Robustness**: Handles noisy scanned images and varies layouts.

---

## 🛠️ Architecture

1.  **Input**: Folder of images (JPG, PNG).
2.  **Detection (YOLOv8)**:
    -   Locates specific regions: `Dealer Block`, `Model Block`, `Amount Block`, `Signature`, `Stamp`.
3.  **OCR (PaddleOCR)**:
    -   English extraction first.
    -   Fallback to specific Indic language models if confidence is low.
4.  **Post-Processing**:
    -   Regex parsing for amounts.
    -   Fuzzy matching (`SequenceMatcher`) for model names.
    -   Dictionary lookups for HP validation.
5.  **Output**: JSON format containing extracted fields.

---

## 📋 Prerequisites & Installation

Ensure you have **Python 3.8+** installed.

### 1. Install Dependencies

Run the following command to install the required libraries:

```bash
pip install ultralytics paddlepaddle paddleocr opencv-python
```

*Note: For GPU support with PaddlePaddle/Ultralytics, please refer to their respective official documentation for CUDA installation.*

---

## 📂 Repository Structure

```
Dhruv/
├── executable.py        # Main entry point script
├── README.md            # Project documentation
├── train.json           # Training dataset annotations
├── amount.pt            # YOLO model for Amount detection
├── dealer.pt            # YOLO model for Dealer Name detection
├── model.pt             # YOLO model for Tractor Model detection
├── sign.pt              # YOLO model for Signature detection
├── stamp.pt             # YOLO model for Stamp detection
├── preparation/        # Dataset preparation & training (Kaggle)
│   ├── labeller.py      # Labeling / annotation code
│   ├── sign.ipynb       # Signature model training
│   ├── model.ipynb     # Tractor model training
│   ├── stamp.ipynb     # Stamp detection training
│   ├── amount.ipynb    # Amount detection training
│   └── dealer.ipynb    # Dealer name detection training
        
```

---

## ▶️ Usage Guide

### 1. Preparing Models
Ensure that the `.pt` model files are accessible.
*Note: The current script configuration expects models to be located in the parent directory (`../`). If running as a standalone submission folder, you may need to adjust the paths in `executable.py` (lines 361-367) to valid paths (e.g., remove `../` if models are in the same folder).*

### 2. Running the Extraction

Execute the main script:

```bash
python executable.py
```

### 3. Execution Steps
1.  The script will initialize and load the OCR models (this may take 10-15 seconds).
2.  You will be prompted to enter the **folder path** containing your target images.
    ```text
    📂 Enter folder path containing images:
    > C:\path\to\images
    ```
3.  The system will process each image and display **Real-time JSON output** in the terminal.
4.  A final consolidated `output.json` will be saved in the input folder.

---

## � Output Format

The system generates a JSON object for each document:

```json
{
  "doc_id": "image_filename",
  "fields": {
    "dealer_name": "Agro Industries Ltd",
    "model_name": "Swaraj 744 FE",
    "horse_power": 48,
    "asset_cost": 801815,
    "signature": {
      "present": true,
      "bbox": [820, 1405, 1050, 1480]
    },
    "stamp": {
      "present": true,
      "bbox": [766, 1387, 1103, 1599]
    }
  },
  "confidence": 0.52,
  "processing_time_sec": 4.8
}
```

---

## 👨‍💻 Author

**Dhruv Kajare**  
Hackathon Participant – Intelligent Document AI

---
