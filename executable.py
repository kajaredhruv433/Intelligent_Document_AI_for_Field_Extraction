import cv2
import json
import time
import re
import os
from difflib import SequenceMatcher
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ==================================================
# SMART MULTILINGUAL OCR (ENGLISH FIRST)
# ==================================================

OCR_EN = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

FALLBACK_LANGS = ["devanagari", "ta", "te", "ka"]
OCR_FALLBACK = {}  # lazy-loaded only when needed


def ocr_quality(text):
    if not text:
        return 0
    alnum = sum(c.isalnum() for c in text)
    return len(text) * (alnum / max(len(text), 1))


def best_multilingual_ocr(image):
    # 1️⃣ Try English first
    try:
        res = OCR_EN.ocr(image, cls=True)
        text_en = " ".join(
            w[1][0] for l in res for w in l
            if w and w[1] and w[1][0]
        ) if res else ""
    except Exception:
        text_en = ""

    score_en = ocr_quality(text_en)

    # ✅ English is good enough → return immediately
    if score_en >= 20:
        return text_en.strip()

    # 2️⃣ Fallback to other languages (lazy-loaded)
    best_text = text_en
    best_score = score_en

    for lang in FALLBACK_LANGS:
        try:
            if lang not in OCR_FALLBACK:
                OCR_FALLBACK[lang] = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    show_log=False
                )

            ocr = OCR_FALLBACK[lang]
            res = ocr.ocr(image, cls=True)
            if not res:
                continue

            text = " ".join(
                w[1][0] for l in res for w in l
                if w and w[1] and w[1][0]
            )

            score = ocr_quality(text)
            if score > best_score:
                best_score = score
                best_text = text

        except Exception:
            continue

    return best_text.strip()

# ==================================================
# TRACTOR DICTIONARY
# ==================================================

TRACTOR_MODELS =[
    # --- SWARAJ ---
    {"brand": "Swaraj", "model": "Code", "hp": (11, 11)},
    {"brand": "Swaraj", "model": "Target 630", "hp": (29, 29)},
    {"brand": "Swaraj", "model": "630 Target", "hp": (29, 29)},
    {"brand": "Swaraj", "model": "717", "hp": (15, 15)},
    {"brand": "Swaraj", "model": "724 XM Orchard NT", "hp": (24, 24)},
    {"brand": "Swaraj", "model": "XM Orchard NT 724", "hp": (24, 24)},
    {"brand": "Swaraj", "model": "825 XM", "hp": (25, 25)},
    {"brand": "Swaraj", "model": "XM 825", "hp": (25, 25)},
    {"brand": "Swaraj", "model": "834 XM", "hp": (35, 35)},
    {"brand": "Swaraj", "model": "XM 834", "hp": (35, 35)},
    {"brand": "Swaraj", "model": "735 FE", "hp": (35, 35)},
    {"brand": "Swaraj", "model": "FE 735", "hp": (35, 35)},
    {"brand": "Swaraj", "model": "735 XT", "hp": (40, 40)},
    {"brand": "Swaraj", "model": "XT 735", "hp": (40, 40)},
    {"brand": "Swaraj", "model": "744 FE", "hp": (48, 48)},
    {"brand": "Swaraj", "model": "FE 744", "hp": (48, 48)},
    {"brand": "Swaraj", "model": "744 XT", "hp": (50, 50)},
    {"brand": "Swaraj", "model": "XT 744", "hp": (50, 50)},
    {"brand": "Swaraj", "model": "744 XM", "hp": (50, 50)},
    {"brand": "Swaraj", "model": "XM 744", "hp": (50, 50)},
    {"brand": "Swaraj", "model": "843 XM", "hp": (42, 42)},
    {"brand": "Swaraj", "model": "XM 843", "hp": (42, 42)},
    {"brand": "Swaraj", "model": "855 FE", "hp": (52, 52)},
    {"brand": "Swaraj", "model": "FE 855", "hp": (52, 52)},
    {"brand": "Swaraj", "model": "855 FE (4WD / Protek / DT Plus)", "hp": (55, 55)},
    {"brand": "Swaraj", "model": "FE (4WD / Protek / DT Plus) 855", "hp": (55, 55)},
    {"brand": "Swaraj", "model": "960 FE", "hp": (55, 55)},
    {"brand": "Swaraj", "model": "FE 960", "hp": (55, 55)},
    {"brand": "Swaraj", "model": "963 FE", "hp": (60, 60)},
    {"brand": "Swaraj", "model": "FE 963", "hp": (60, 60)},
    {"brand": "Swaraj", "model": "969 FE", "hp": (65, 70)},
    {"brand": "Swaraj", "model": "FE 969", "hp": (65, 70)},
    {"brand": "Swaraj", "model": "978 FE", "hp": (75, 75)},
    {"brand": "Swaraj", "model": "FE 978", "hp": (75, 75)},

    # --- MAHINDRA ---
    {"brand": "Mahindra", "model": "OJA 2121", "hp": (21, 21)},
    {"brand": "Mahindra", "model": "2121 OJA", "hp": (21, 21)},
    {"brand": "Mahindra", "model": "OJA 2124", "hp": (24, 24)},
    {"brand": "Mahindra", "model": "2124 OJA", "hp": (24, 24)},
    {"brand": "Mahindra", "model": "OJA 2127", "hp": (27, 27)},
    {"brand": "Mahindra", "model": "2127 OJA", "hp": (27, 27)},
    {"brand": "Mahindra", "model": "OJA 2130", "hp": (30, 30)},
    {"brand": "Mahindra", "model": "2130 OJA", "hp": (30, 30)},
    {"brand": "Mahindra", "model": "OJA 3132", "hp": (32, 32)},
    {"brand": "Mahindra", "model": "3132 OJA", "hp": (32, 32)},
    {"brand": "Mahindra", "model": "OJA 3136", "hp": (36, 36)},
    {"brand": "Mahindra", "model": "3136 OJA", "hp": (36, 36)},
    {"brand": "Mahindra", "model": "OJA 3140", "hp": (40, 40)},
    {"brand": "Mahindra", "model": "3140 OJA", "hp": (40, 40)},
    {"brand": "Mahindra", "model": "YUVRAJ 215 NXT", "hp": (15, 15)},
    {"brand": "Mahindra", "model": "215 NXT YUVRAJ", "hp": (15, 15)},
    {"brand": "Mahindra", "model": "JIVO 225 DI", "hp": (20, 20)},
    {"brand": "Mahindra", "model": "225 DI JIVO", "hp": (20, 20)},
    {"brand": "Mahindra", "model": "JIVO 245 DI", "hp": (24, 24)},
    {"brand": "Mahindra", "model": "245 DI JIVO", "hp": (24, 24)},
    {"brand": "Mahindra", "model": "JIVO 305 DI", "hp": (30, 30)},
    {"brand": "Mahindra", "model": "305 DI JIVO", "hp": (30, 30)},
    {"brand": "Mahindra", "model": "305 Orchard", "hp": (30, 30)},
    {"brand": "Mahindra", "model": "Orchard 305", "hp": (30, 30)},
    {"brand": "Mahindra", "model": "JIVO 365 DI 4WD", "hp": (36, 36)},
    {"brand": "Mahindra", "model": "365 DI 4WD JIVO", "hp": (36, 36)},
    {"brand": "Mahindra", "model": "265 DI XP PLUS", "hp": (33, 33)},
    {"brand": "Mahindra", "model": "XP PLUS 265 DI", "hp": (33, 33)},
    {"brand": "Mahindra", "model": "265 DI SP PLUS", "hp": (31, 31)},
    {"brand": "Mahindra", "model": "SP PLUS 265 DI", "hp": (31, 31)},
    {"brand": "Mahindra", "model": "275 DI XP PLUS", "hp": (37, 37)},
    {"brand": "Mahindra", "model": "XP PLUS 275 DI", "hp": (37, 37)},
    {"brand": "Mahindra", "model": "275 DI SP PLUS", "hp": (37, 37)},
    {"brand": "Mahindra", "model": "SP PLUS 275 DI", "hp": (37, 37)},
    {"brand": "Mahindra", "model": "415 DI XP PLUS", "hp": (42, 42)},
    {"brand": "Mahindra", "model": "XP PLUS 415 DI", "hp": (42, 42)},
    {"brand": "Mahindra", "model": "415 DI SP PLUS", "hp": (42, 42)},
    {"brand": "Mahindra", "model": "SP PLUS 415 DI", "hp": (42, 42)},
    {"brand": "Mahindra", "model": "475 DI XP PLUS", "hp": (44, 44)},
    {"brand": "Mahindra", "model": "XP PLUS 475 DI", "hp": (44, 44)},
    {"brand": "Mahindra", "model": "475 DI SP PLUS", "hp": (44, 44)},
    {"brand": "Mahindra", "model": "SP PLUS 475 DI", "hp": (44, 44)},
    {"brand": "Mahindra", "model": "575 DI XP PLUS", "hp": (47, 47)},
    {"brand": "Mahindra", "model": "XP PLUS 575 DI", "hp": (47, 47)},
    {"brand": "Mahindra", "model": "575 DI SP PLUS", "hp": (47, 47)},
    {"brand": "Mahindra", "model": "SP PLUS 575 DI", "hp": (47, 47)},
    {"brand": "Mahindra", "model": "585 DI XP PLUS", "hp": (49, 49)},
    {"brand": "Mahindra", "model": "XP PLUS 585 DI", "hp": (49, 49)},
    {"brand": "Mahindra", "model": "585 DI SP PLUS", "hp": (50, 50)},
    {"brand": "Mahindra", "model": "SP PLUS 585 DI", "hp": (50, 50)},
    {"brand": "Mahindra", "model": "265 DI YUVO TECH+", "hp": (33, 33)},
    {"brand": "Mahindra", "model": "YUVO TECH+ 265 DI", "hp": (33, 33)},
    {"brand": "Mahindra", "model": "405 YUVO TECH+", "hp": (39, 39)},
    {"brand": "Mahindra", "model": "YUVO TECH+ 405", "hp": (39, 39)},
    {"brand": "Mahindra", "model": "415 YUVO TECH+", "hp": (40, 40)},
    {"brand": "Mahindra", "model": "YUVO TECH+ 415", "hp": (40, 40)},
    {"brand": "Mahindra", "model": "475 YUVO TECH+", "hp": (44, 44)},
    {"brand": "Mahindra", "model": "YUVO TECH+ 475", "hp": (44, 44)},
    {"brand": "Mahindra", "model": "575 YUVO TECH+", "hp": (47, 47)},
    {"brand": "Mahindra", "model": "YUVO TECH+ 575", "hp": (47, 47)},
    {"brand": "Mahindra", "model": "585 YUVO TECH+", "hp": (49, 50)},
    {"brand": "Mahindra", "model": "YUVO TECH+ 585", "hp": (49, 50)},
    {"brand": "Mahindra", "model": "ARJUN 605 DI MS", "hp": (50, 50)},
    {"brand": "Mahindra", "model": "605 DI MS ARJUN", "hp": (50, 50)},
    {"brand": "Mahindra", "model": "ARJUN 605 DI i", "hp": (55, 55)},
    {"brand": "Mahindra", "model": "ARJUN 605 DI PP", "hp": (60, 60)},
    {"brand": "Mahindra", "model": "NOVO 605 DI PS", "hp": (50, 50)},
    {"brand": "Mahindra", "model": "NOVO 605 DI", "hp": (60, 60)},
    {"brand": "Mahindra", "model": "NOVO 655 DI", "hp": (68, 68)},
    {"brand": "Mahindra", "model": "NOVO 755 DI", "hp": (74, 75)},

    # --- TAFE ---
    {"brand": "TAFE", "model": "30 DI Orchard Plus", "hp": (30, 30)},
    {"brand": "TAFE", "model": "30 DI Dynatrack", "hp": (30, 30)},
    {"brand": "TAFE", "model": "42 DI", "hp": (41, 41)},
    {"brand": "TAFE", "model": "45 DI", "hp": (46, 46)},
    {"brand": "TAFE", "model": "5900 DI", "hp": (56, 56)},
    {"brand": "TAFE", "model": "6028 M", "hp": (28, 28)},
    {"brand": "TAFE", "model": "7515", "hp": (71, 71)},
    {"brand": "TAFE", "model": "1002", "hp": (100, 100)},

    # --- SONALIKA ---
    {"brand": "Sonalika", "model": "MM 18", "hp": (18, 18)},
    {"brand": "Sonalika", "model": "DI 20 Tiger", "hp": (20, 20)},
    {"brand": "Sonalika", "model": "DI 22", "hp": (22, 22)},
    {"brand": "Sonalika", "model": "DI 24 RX", "hp": (24, 24)},
    {"brand": "Sonalika", "model": "DI 26", "hp": (26, 26)},
    {"brand": "Sonalika", "model": "DI 30 BAAGBAN", "hp": (30, 30)},
    {"brand": "Sonalika", "model": "DI 32", "hp": (32, 32)},
    {"brand": "Sonalika", "model": "DI 35", "hp": (39, 39)},
    {"brand": "Sonalika", "model": "DI 35 RX", "hp": (39, 39)},
    {"brand": "Sonalika", "model": "DI 42 RX", "hp": (42, 42)},
    {"brand": "Sonalika", "model": "Sikander", "hp": (42, 52)},
    {"brand": "Sonalika", "model": "DI 47 RX", "hp": (50, 50)},
    {"brand": "Sonalika", "model": "DI 50 RX", "hp": (52, 52)},
    {"brand": "Sonalika", "model": "Tiger DI 55", "hp": (55, 55)},
    {"brand": "Sonalika", "model": "Worldtrac 60", "hp": (60, 60)},
    {"brand": "Sonalika", "model": "Worldtrac 75 RX", "hp": (75, 75)},
    {"brand": "Sonalika", "model": "Worldtrac 90 RX", "hp": (90, 90)},

    # --- ESCORTS / FARMTRAC / POWERTRAC / DIGITRAC ---
    {"brand": "Escorts", "model": "335", "hp": (35, 35)},
    {"brand": "Escorts", "model": "375", "hp": (38, 38)},
    {"brand": "Escorts", "model": "450", "hp": (45, 45)},
    {"brand": "Escorts", "model": "550", "hp": (50, 50)},
    {"brand": "Farmtrac", "model": "26", "hp": (26, 26)},
    {"brand": "Farmtrac", "model": "30", "hp": (30, 30)},
    {"brand": "Farmtrac", "model": "35", "hp": (35, 35)},
    {"brand": "Farmtrac", "model": "39", "hp": (39, 39)},
    {"brand": "Farmtrac", "model": "45", "hp": (45, 50)},
    {"brand": "Farmtrac", "model": "47", "hp": (47, 47)},
    {"brand": "Farmtrac", "model": "50", "hp": (50, 52)},
    {"brand": "Farmtrac", "model": "60", "hp": (50, 55)},
    {"brand": "Farmtrac", "model": "6060", "hp": (60, 60)},
    {"brand": "Farmtrac", "model": "6075", "hp": (75, 75)},
    {"brand": "Powertrac", "model": "Euro 28", "hp": (22, 28)},
    {"brand": "Powertrac", "model": "Euro 35", "hp": (35, 35)},
    {"brand": "Powertrac", "model": "Euro 39", "hp": (39, 39)},
    {"brand": "Powertrac", "model": "Euro 42 Plus", "hp": (45, 47)},
    {"brand": "Powertrac", "model": "Euro 45 Plus", "hp": (47, 47)},
    {"brand": "Powertrac", "model": "Euro 50", "hp": (50, 52)},
    {"brand": "Powertrac", "model": "Euro 55", "hp": (55, 55)},
    {"brand": "Powertrac", "model": "Euro 60", "hp": (60, 60)},
    {"brand": "Digitrac", "model": "PP 43i", "hp": (47, 50)},
    {"brand": "Digitrac", "model": "PP 46i", "hp": (50, 55)},
    {"brand": "Digitrac", "model": "PP 51i", "hp": (60, 60)},

    # --- JOHN DEERE ---
    {"brand": "John Deere", "model": "3028 EN", "hp": (28, 28)},
    {"brand": "John Deere", "model": "3036E", "hp": (35, 36)},
    {"brand": "John Deere", "model": "5042D", "hp": (43, 44)},
    {"brand": "John Deere", "model": "5045D", "hp": (45, 46)},
    {"brand": "John Deere", "model": "5050D", "hp": (50, 50)},
    {"brand": "John Deere", "model": "5210", "hp": (50, 50)},
    {"brand": "John Deere", "model": "5055E", "hp": (55, 55)},
    {"brand": "John Deere", "model": "5310", "hp": (55, 55)},
    {"brand": "John Deere", "model": "5060E", "hp": (60, 60)},
    {"brand": "John Deere", "model": "5075E", "hp": (75, 75)},
    {"brand": "John Deere", "model": "5105", "hp": (40, 40)},
    {"brand": "John Deere", "model": "5405", "hp": (63, 65)},

    # --- NEW HOLLAND ---
    {"brand": "New Holland", "model": "Simba 20", "hp": (17, 17)},
    {"brand": "New Holland", "model": "3032 Nx", "hp": (35, 35)},
    {"brand": "New Holland", "model": "3600-2 TX", "hp": (49, 50)},
    {"brand": "New Holland", "model": "3600 TX", "hp": (47, 47)},
    {"brand": "New Holland", "model": "4710", "hp": (47, 47)},
    {"brand": "New Holland", "model": "3630 TX Plus", "hp": (49, 55)},
    {"brand": "New Holland", "model": "5620 TX Plus", "hp": (65, 65)},
    {"brand": "New Holland", "model": "5630 TX Plus", "hp": (75, 75)},

    # --- KUBOTA & FORCE ---
    {"brand": "Kubota", "model": "NeoStar A211N", "hp": (21, 21)},
    {"brand": "Kubota", "model": "NeoStar B2441", "hp": (24, 24)},
    {"brand": "Kubota", "model": "MU4501", "hp": (45, 45)},
    {"brand": "Kubota", "model": "MU5501", "hp": (55, 55)},
    {"brand": "Force", "model": "Orchard DLX", "hp": (27, 27)},
    {"brand": "Force", "model": "Balwan 400", "hp": (39, 40)},
    {"brand": "Force", "model": "Balwan 500", "hp": (50, 50)},
    {"brand": "Force", "model": "Balwan 600", "hp": (60, 60)},

    # --- OTHERS ---
    {"brand": "PREET", "model": "2549", "hp": (25, 25)},
    {"brand": "PREET", "model": "3549", "hp": (35, 35)},
    {"brand": "PREET", "model": "4549", "hp": (45, 45)},
    {"brand": "PREET", "model": "6049", "hp": (60, 60)},
    {"brand": "PREET", "model": "6549", "hp": (65, 65)},
    {"brand": "PREET", "model": "7549", "hp": (75, 75)},
    {"brand": "VST", "model": "MT 171", "hp": (17, 17)},
    {"brand": "VST", "model": "MT 180", "hp": (18, 18)},
    {"brand": "VST", "model": "MT 224-1D", "hp": (22, 22)},
    {"brand": "VST", "model": "927", "hp": (24, 27)},
    {"brand": "CAPTAIN", "model": "200", "hp": (20, 20)},
    {"brand": "CAPTAIN", "model": "250", "hp": (25, 25)},
    {"brand": "CAPTAIN", "model": "280", "hp": (28, 28)},
    {"brand": "ACE", "model": "DI 350", "hp": (35, 35)},
    {"brand": "ACE", "model": "DI 650", "hp": (61, 65)},
    {"brand": "INDO FARM", "model": "2035", "hp": (35, 35)},
    {"brand": "INDO FARM", "model": "4175", "hp": (75, 75)},
    {"brand": "HMT", "member": "2511", "hp": (25, 25)},
    {"brand": "HMT", "member": "5911", "hp": (60, 60)},
    {"brand": "EICHER", "model": "188", "hp": (18, 18)},
    {"brand": "EICHER", "model": "241", "hp": (25, 25)},
    {"brand": "EICHER", "model": "242", "hp": (25, 25)},
    {"brand": "EICHER", "model": "312", "hp": (30, 30)},
    {"brand": "EICHER", "model": "333", "hp": (35, 36)},
    {"brand": "EICHER", "model": "368", "hp": (38, 40)},
    {"brand": "EICHER", "model": "380", "hp": (40, 44)},
    {"brand": "EICHER", "model": "485", "hp": (45, 45)},
    {"brand": "EICHER", "model": "551", "hp": (49, 50)},
    {"brand": "EICHER", "model": "557", "hp": (50, 50)},
    {"brand": "STANDARD", "model": "DI 335", "hp": (35, 35)},
    {"brand": "STANDARD", "model": "DI 460", "hp": (60, 60)},
]

# ==================================================
# UTILS
# ==================================================

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_model_name(item):
    return item.get("model") or item.get("member") or ""


def parse_amount(text):
    if not text:
        return None
    text = text.replace(",", "").replace("/-", "").replace("=", ".")
    m = re.search(r"\b(\d{4,7})(?:\.\d{1,2})?\b", text)
    return int(m.group(1)) if m else None


def force_normalize_model(text, detected_hp=None):
    text = clean_text(text or "")
    best, best_score = None, -1

    for item in TRACTOR_MODELS:
        combined = clean_text(item["brand"] + " " + get_model_name(item))
        score = similarity(text, combined)
        if score > best_score:
            best_score = score
            best = item

    if not best:
        return None, detected_hp

    hp = detected_hp if detected_hp else best["hp"][0]
    return f"{best['brand']} {get_model_name(best)}", hp

# ==================================================
# LOAD YOLO MODELS ONCE
# ==================================================

models = {
    "stamp": YOLO("../stamp.pt"),
    "signature": YOLO("../sign.pt"),
    "dealer": YOLO("../dealer.pt"),
    "amount": YOLO("../amount.pt"),
    "model": YOLO("../model.pt")
}

# ==================================================
# PROCESS ONE IMAGE
# ==================================================

def process_image(image_path):
    start = time.time()
    img = cv2.imread(image_path)
    if img is None:
        return None

    doc_id = os.path.splitext(os.path.basename(image_path))[0]

    output = {
        "doc_id": doc_id,
        "fields": {
            "dealer_name": "",
            "model_name": None,
            "horse_power": None,
            "asset_cost": None,
            "signature": {"present": False, "bbox": None},
            "stamp": {"present": False, "bbox": None}
        },
        "confidence": 0.0,
        "processing_time_sec": 0.0,
        "cost_estimate_usd": 0.002
    }

    confs = []
    raw_model = ""
    detected_hp = None

    for name, model in models.items():
        results = model(img, conf=0.25, verbose=False)

        for r in results:
            if not r.boxes:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confs.append(float(box.conf[0]))
                crop = img[y1:y2, x1:x2]

                if name in ["stamp", "signature"]:
                    output["fields"][name] = {
                        "present": True,
                        "bbox": [x1, y1, x2, y2]
                    }

                elif name == "dealer":
                    output["fields"]["dealer_name"] = best_multilingual_ocr(crop)

                elif name == "amount":
                    amt = parse_amount(best_multilingual_ocr(crop))
                    if amt:
                        output["fields"]["asset_cost"] = amt

                elif name == "model":
                    txt = best_multilingual_ocr(crop)
                    raw_model = txt
                    m = re.search(r"\b(\d{2})\s*(HP|एचपी)\b", txt.upper())
                    if m:
                        detected_hp = int(m.group(1))

    model_name, hp = force_normalize_model(raw_model, detected_hp)
    output["fields"]["model_name"] = model_name
    output["fields"]["horse_power"] = hp

    output["confidence"] = round(sum(confs) / len(confs), 2) if confs else 0.0
    output["processing_time_sec"] = round(time.time() - start, 2)

    return output

# ==================================================
# MAIN (FOLDER ONLY)
# ==================================================

def main():
    print("\n📂 Enter folder path containing images:")
    folder = input("> ").strip('"')

    if not os.path.isdir(folder):
        print("❌ Invalid folder path")
        return

    images = [
        f for f in sorted(os.listdir(folder))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not images:
        print("❌ No images found in folder")
        return

    results = []

    print("\n🚀 Processing started...\n")

    for img in images:
        path = os.path.join(folder, img)
        result = process_image(path)

        if result:
            results.append(result)

            # 🔥 LIVE JSON OUTPUT
            print("====================================")
            print(json.dumps(result, indent=2))
            print("====================================\n")

    output_path = os.path.join(folder, "output.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ DONE")
    print("📄 Final JSON saved at:", output_path)

# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
