import cv2
import os
import ctypes

# -------- PATHS --------
IMAGE_DIR = r"/train1"
LABEL_DIR = r"C:\Users\dhruv\Music\Skill\Python\Compititions\convolve\Try1\model\labels"

os.makedirs(LABEL_DIR, exist_ok=True)

# -------- SCREEN SIZE (Windows) --------
user32 = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

def resize_fit_height(img):
    h, w = img.shape[:2]

    # Force image to fit screen height
    scale = (SCREEN_H * 0.90) / h

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    return resized, scale

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print("\nINSTRUCTIONS:")
print(" - FULL image will be visible (scaled down)")
print(" - Drag box around STAMP")
print(" - ENTER = save | ESC = skip")
print(" - Class ID = 0 (Stamp)\n")

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    original = cv2.imread(img_path)

    if original is None:
        print(f"❌ Cannot open {img_name}")
        continue

    H, W = original.shape[:2]

    display_img, scale = resize_fit_height(original)

    cv2.namedWindow("YOLO Label Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        "YOLO Label Tool",
        display_img.shape[1],
        display_img.shape[0]
    )

    roi = cv2.selectROI(
        "YOLO Label Tool",
        display_img,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w == 0 or h == 0:
        print(f"⏭ Skipped {img_name}")
        continue

    # ---- Map back to original resolution ----
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    # ---- YOLO format ----
    x_center = (x + w / 2) / W
    y_center = (y + h / 2) / H
    w_norm = w / W
    h_norm = h / H

    label_path = os.path.join(
        LABEL_DIR,
        os.path.splitext(img_name)[0] + ".txt"
    )

    with open(label_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"✅ Labeled: {img_name}")

print("\n🎉 All images labeled. Ready for YOLO training.")
