# Architecture Documentation: Intelligent Document AI for Field Extraction

## Overview

The Intelligent Document AI system is an end-to-end, multi-stage document processing pipeline engineered to extract structured fields and verification artifacts from tractor loan quotation documents, invoices, and financial receipts. The system is designed to handle noisy document scans, heterogeneous layouts, multilingual scripts (English, Devanagari/Hindi, Tamil, Telugu, Kannada), and variations in typography.

The architecture decouples layout detection from textual recognition, employing a hybrid topology:
1. A modular ensemble of five fine-tuned YOLOv8 object detection models for visual localization of regions of interest (ROIs).
2. A priority-driven, cascaded multilingual Optical Character Recognition (OCR) engine based on PaddleOCR with lazy-loaded Indic language models.
3. A rule-based post-processing, regex, and fuzzy string normalization subsystem with a master vehicle catalog.

---

## High-Level System Architecture

The following diagram illustrates the complete execution pipeline from raw document image ingestion to final structured JSON output.

```mermaid
flowchart TD
    subgraph Ingestion["1. Document Ingestion"]
        A["Input Document Image (JPG / PNG)"] --> B["OpenCV Image Ingestion & Resolution Validation"]
    end

    subgraph DetectionEnsemble["2. YOLOv8 Layout Decomposition Ensemble"]
        B --> D1["YOLOv8 Stamp Detector (stamp.pt)"]
        B --> D2["YOLOv8 Signature Detector (sign.pt)"]
        B --> D3["YOLOv8 Dealer Block Detector (dealer.pt)"]
        B --> D4["YOLOv8 Amount Block Detector (amount.pt)"]
        B --> D5["YOLOv8 Model Block Detector (model.pt)"]
    end

    subgraph SpatialAnalysis["3. Spatial Processing & Verification"]
        D1 --> E1["Stamp Presence & Bounding Box [x1, y1, x2, y2]"]
        D2 --> E2["Signature Presence & Bounding Box [x1, y1, x2, y2]"]
        D3 --> E3["Dealer Crop Extraction"]
        D4 --> E4["Amount Crop Extraction"]
        D5 --> E5["Model Crop Extraction"]
    end

    subgraph OCRSubsystem["4. Cascaded Multilingual OCR Engine"]
        E3 --> F1["English PaddleOCR (use_angle_cls=True)"]
        E4 --> F2["English PaddleOCR (use_angle_cls=True)"]
        E5 --> F3["English PaddleOCR (use_angle_cls=True)"]
        
        F1 --> G1{"Quality Score >= 20?"}
        F2 --> G2{"Quality Score >= 20?"}
        F3 --> G3{"Quality Score >= 20?"}
        
        G1 -- "No (Low Quality / Indic Script)" --> H1["Lazy-load Indic Fallback Cascade<br/>(Devanagari, Tamil, Telugu, Kannada)"]
        G2 -- "No (Low Quality / Indic Script)" --> H2["Lazy-load Indic Fallback Cascade<br/>(Devanagari, Tamil, Telugu, Kannada)"]
        G3 -- "No (Low Quality / Indic Script)" --> H3["Lazy-load Indic Fallback Cascade<br/>(Devanagari, Tamil, Telugu, Kannada)"]
        
        G1 -- "Yes (Fast Path: ~2s)" --> I1["Raw Dealer Text"]
        G2 -- "Yes (Fast Path: ~2s)" --> I2["Raw Amount Text"]
        G3 -- "Yes (Fast Path: ~2s)" --> I3["Raw Model Text"]
        
        H1 --> I1
        H2 --> I2
        H3 --> I3
    end

    subgraph Normalization["5. Normalization & Disambiguation Engine"]
        I1 --> J1["Dealer Name Clean-up"]
        I2 --> J2["Currency & Regex Number Sanitizer"]
        I3 --> J3["HP Extraction & Fuzzy Matcher vs Master Catalog"]
        
        J2 --> K1["Sanitized Integer Asset Cost"]
        J3 --> K2["Canonical Brand + Model Name"]
        J3 --> K3["Validated Horse Power (HP)"]
    end

    subgraph Assembly["6. Payload Aggregation"]
        E1 --> L["Output Synthesizer"]
        E2 --> L
        J1 --> L
        K1 --> L
        K2 --> L
        K3 --> L
        L --> M["Structured JSON Output + Execution Metrics"]
    end
```

---

## Detailed Subsystem Breakdown

### 1. Vision and Layout Decomposition Subsystem

Rather than relying on a single monolithic detector that suffers from class imbalance across divergent aspect ratios, the layout analysis module employs five specialized YOLOv8 nano detectors. Each model is fine-tuned on targeted annotations to optimize bounding box precision and spatial localization.

| Model Name | Target Object | Input | Task Type | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `stamp.pt` | Official Stamp / Seal | Full Document Image | Object Detection | Verification of physical document authentication. |
| `sign.pt` | Authorized Signature | Full Document Image | Object Detection | Verification of signatory presence and bounding box localization. |
| `dealer.pt` | Dealer Header / Block | Full Document Image | Object Detection | Bounding region isolation for dealer business entity text. |
| `amount.pt` | Quotation Total / Asset Cost | Full Document Image | Object Detection | Bounding region isolation for numerical transaction amount. |
| `model.pt` | Vehicle Model Description | Full Document Image | Object Detection | Bounding region isolation for tractor make, variant, and horsepower. |

#### Inference Configuration
- **Confidence Threshold**: `conf = 0.25`
- **Intersection Over Union (IoU)**: Internal Non-Maximum Suppression (NMS) applied per model.
- **Bounding Box Format**: `[x1, y1, x2, y2]` in absolute pixel coordinates.

---

### 2. Cascaded Multilingual OCR Engine

The OCR engine balances high throughput with multilingual coverage across Indian regional scripts. Processing is organized in a priority cascade.

```mermaid
flowchart TD
    Start["Crop ROI Received"] --> Step1["Run English PaddleOCR with Angle Classifier"]
    Step1 --> ScoreCalc["Compute Quality Score: Score = len(Text) * (AlphanumericCount / len(Text))"]
    ScoreCalc --> Check{"Score >= 20?"}
    
    Check -- "Yes" --> FastExit["Return English Text Immediately (Latency ~ 2.0s)"]
    
    Check -- "No" --> InitFallback["Initialize Fallback Cascade: [Devanagari, Tamil, Telugu, Kannada]"]
    InitFallback --> Loop["Iterate over Languages (Lazy-loaded)"]
    Loop --> RunLang["Execute PaddleOCR(lang=L)"]
    RunLang --> EvalScore["Evaluate Language Score vs Best Score"]
    EvalScore --> UpdateBest{"Score > Best Score?"}
    UpdateBest -- "Yes" --> RecordBest["Update Best Text & Score"]
    UpdateBest -- "No" --> NextLang{"More Languages in Cascade?"}
    RecordBest --> NextLang
    NextLang -- "Yes" --> Loop
    NextLang -- "No" --> ReturnBest["Return Optimal Multi-lingual Text (Latency <= 28.0s)"]
```

#### Mathematical Quality Metric
The heuristic scoring function evaluates character density and alphanumeric integrity:

$$\text{Quality Score}(T) = \text{len}(T) \times \left( \frac{\sum_{c \in T} \mathbb{I}[c \in \text{Alphanumeric}]}{\max(\text{len}(T), 1)} \right)$$

- If the English OCR text satisfies $\text{Quality Score} \ge 20$, execution exits in approximately 2.0 seconds.
- If the score falls below the threshold (typical for documents in Hindi, Tamil, Telugu, or Kannada), the engine systematically evaluates each Indic language model, returning the highest-scoring candidate within a maximum processing window of 28.0 seconds.

---

### 3. Entity Extraction and Normalization Subsystem

```mermaid
flowchart LR
    subgraph ModelMatching["Tractor Model & HP Resolution"]
        RawModel["Raw Extracted OCR Text"] --> RegexHP["Regex HP Matcher: r'\b(\d{2})\s*(HP|एचपी)\b'"]
        RawModel --> Clean["Text Sanitization (Uppercase, Alphanumeric)"]
        Clean --> Fuzzy["SequenceMatcher vs Master Catalog (TRACTOR_MODELS)"]
        Fuzzy --> SelectBest["Select Highest Similarity Record"]
        RegexHP --> ResolveHP{"Detected HP Present?"}
        SelectBest --> ResolveHP
        ResolveHP -- "Yes" --> FinalHP["Preserve Detected HP"]
        ResolveHP -- "No" --> LookupHP["Default to Catalog HP Range"]
    end

    subgraph AmountParsing["Asset Cost Parsing"]
        RawAmt["Raw Amount OCR Text"] --> CleanAmt["Strip Comma, '/-', '=', Decimal Artifacts"]
        CleanAmt --> RegexAmt["Regex Matcher: r'\b(\d{4,7})(?:\.\d{1,2})?\b'"]
        RegexAmt --> IntAmt["Cast to Integer Amount"]
    end
```

#### Master Tractor Catalog Disambiguation
The system maintains a comprehensive dictionary of Indian tractor manufacturers and variants, including:
- Swaraj (Code, Target, 717, 724 XM, 735 FE, 744 FE, 855 FE, 963 FE, etc.)
- Mahindra (OJA series, YUVRAJ, JIVO, XP PLUS, SP PLUS, YUVO TECH+, ARJUN, NOVO)
- TAFE (Orchard Plus, Dynatrack, 42 DI, 5900 DI, etc.)
- Sonalika (Tiger, Sikander, Worldtrac series)
- Escorts / Farmtrac / Powertrac / Digitrac
- John Deere (D series, E series)
- New Holland (Simba, TX series)
- Kubota, Force, Preet, VST, Captain, ACE, Indo Farm, HMT, Eicher, Standard

Fuzzy matching is computed using Ratcliff-Obershelp similarity via Python's `difflib.SequenceMatcher`:

$$\text{Similarity}(S_1, S_2) = \frac{2 \times M}{|S_1| + |S_2|}$$

where $M$ is the number of matching characters within non-overlapping maximal common substrings.

---

### 4. Data Preparation and Annotation Architecture

The training data pipeline includes a custom desktop annotation utility for fast spatial labeling under Windows screen scaling constraints.

```mermaid
flowchart TD
    A["Raw High-Resolution Scans (JPG/PNG)"] --> B["preparation/labeller.py"]
    B --> C["Dynamic Screen-Fit Rescaling: scale = (ScreenHeight * 0.90) / ImageHeight"]
    C --> D["OpenCV Interactive ROI Selection (cv2.selectROI)"]
    D --> E["Inverse Scaling Transformation: (x / scale, y / scale)"]
    E --> F["Normalized YOLO Coordinate Calculation: [x_center, y_center, w, h]"]
    F --> G["YOLO Format Annotation Files (.txt)"]
    G --> H["Kaggle / GPU Training Notebooks (*.ipynb)"]
    H --> I["Trained Weights (*.pt) Export"]
```

---

## Runtime Performance Analysis

| Processing Path | Document Language / Condition | Average Latency | Bottleneck Components |
| :--- | :--- | :--- | :--- |
| Fast Path | Clear English quotations | ~2.0 seconds | YOLOv8 inference + 1x English PaddleOCR pass |
| Indic Fallback | Devanagari / Hindi quotations | ~6.0 - 12.0 seconds | YOLOv8 + English OCR + Devanagari PaddleOCR |
| Full Cascade Fallback | Complex Indic / Multilingual quotations | Max 28.0 seconds | YOLOv8 + Sequential evaluation across 4 Indic models |

### Confidence Metric Formulation
The aggregate confidence score returned per document represents the arithmetic mean of all detected bounding box confidence scores across all five YOLOv8 detector instances:

$$\text{Document Confidence} = \frac{1}{N} \sum_{i=1}^{N} \text{conf}_i$$

where $N$ is the total count of active bounding boxes identified across all sub-models.
