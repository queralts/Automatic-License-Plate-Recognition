
import os
import cv2
import numpy as np
"""
If the file shows many zeros in the coordinates, it means that the function
detectPlatesCropped() did not find any valid plate contour inside the cropped image.
When this happens, the variable quad_or_none is None, so the code falls back to the
full image rectangle: [[0,0], [W-1,0], [W-1,H-1], [0,H-1]].
This usually occurs because the detection filters (aspect ratio, area ratio, etc.)
were designed for full vehicle images, not for already cropped plates.
Therefore, in tightly cropped YOLO outputs, the algorithm often rejects the plate
region as too large and returns None, resulting in coordinates that describe the
whole crop instead of an inner plate area.
"""

def order_box(box):
    """Return points ordered as TL, TR, BR, BL (float32, shape (4,2))."""
    rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]          # TL
    rect[2] = box[np.argmax(s)]          # BR
    diff = np.diff(box, axis=1).ravel()
    rect[1] = box[np.argmin(diff)]       # TR
    rect[3] = box[np.argmax(diff)]       # BL
    return rect

def detectPlatesCropped(crop_bgr):
    """
    Returns a single 4-point quad (np.float32 shape (4,2), in crop coords),
    or None if nothing reliable found.
    """
    H, W = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Black-hat to highlight dark strokes on bright plate
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k3, iterations=3)

    # Horizontal gradient
    gx = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    gx = np.abs(gx)
    mn, mx = gx.min(), gx.max()
    if mx - mn < 1e-6:
        return None
    gx = ((gx - mn) / (mx - mn) * 255).astype("uint8")

    # Smooth + close to bridge gaps between characters
    gx = cv2.GaussianBlur(gx, (7, 7), 0)
    krect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    gx = cv2.morphologyEx(gx, cv2.MORPH_CLOSE, krect, iterations=2)

    # Relative threshold
    T = 0.3 * float(gx.max())
    _, th = cv2.threshold(gx, T, 255, cv2.THRESH_BINARY)

    # Clean up and consolidate
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k3, iterations=4)
    th = cv2.dilate(th, krect, iterations=2)

    # Find candidate contours
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    best = None
    best_score = -1.0
    img_area = float(H * W)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if w == 0 or h == 0:
            continue
        ar = w / float(h)
        area_ratio = area / img_area
        wr = w / float(W)
        hr = h / float(H)

                # 3) Relax geometric gates
        if not (2.0 <= ar <= 8.5):              # was 2.2–7.5
            continue
        if not (0.05 <= area_ratio <= 0.95):    # was 0.10–0.90
            continue
        if not (0.35 <= wr <= 1.00 and 0.20 <= hr <= 0.90):  # was 0.50/0.25
            continue

        # Rectangularity score: area / minAreaRectArea
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype("float32")
        rect_w = np.linalg.norm(box[0] - box[1])
        rect_h = np.linalg.norm(box[1] - box[2])
        rect_area = max(rect_w * rect_h, 1.0)
        rectangularity = area / rect_area
        score = area * rectangularity

        if score > best_score:
            best_score = score
            best = box

    if best is None:
        return None

    # Clip to crop bounds and order
    best[:, 0] = np.clip(best[:, 0], 0, W - 1)
    best[:, 1] = np.clip(best[:, 1], 0, H - 1)
    return order_box(best)

def detection_txt(crop_path, crop_bgr, quad_or_none):
    """
    Writes <crop_path with .txt> containing 'x1 y1 x2 y2 x3 y3 x4 y4' (TL,TR,BR,BL).
    Falls back to the crop corners if no quad found. Always writes ONE line.
    """
    H, W = crop_bgr.shape[:2]
    if quad_or_none is None:
        quad = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
    else:
        quad = quad_or_none.astype(np.float32)

    x1, y1 = quad[0]; x2, y2 = quad[1]; x3, y3 = quad[2]; x4, y4 = quad[3]
    line = f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x3:.2f} {y3:.2f} {x4:.2f} {y4:.2f}"
    out_txt = os.path.splitext(crop_path)[0] + ".txt"
    with open(out_txt, "w") as f:
        f.write(line + "\n")
    return out_txt

if __name__ == "__main__":
    """
    Creates .txt files with plate corner coordinates for YOLO-cropped images.
    Scans Detection_YOLO/cropped_plates/Frontal and Lateral.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "Detection_YOLO", "cropped_plates")

    folders = ["Frontal", "Lateral"]
    for sub in folders:
        folder = os.path.join(base_dir, sub)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue

        print(f"\n[INFO] Processing folder: {folder}")
        for img_name in os.listdir(folder):
            if not img_name.lower().endswith((".jpg", ".png")):
                continue  # skip non-images
            img_path = os.path.join(folder, img_name)
            crop = cv2.imread(img_path)
            if crop is None:
                print(f"[ERR] Cannot read image: {img_path}")
                continue

            quad = detectPlatesCropped(crop)          # may return None
            out = detection_txt(img_path, crop, quad) # always writes one line
            print(f"[OK] {os.path.basename(out)} created")

    print("\n[Done] Coordinate .txt files generated for all YOLO crops.")


