# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:45:11 2025

@author: debora
"""
# import the necessary packages
#from collections import namedtuple
# from skimage.filters import threshold_local
# from skimage import segmentation
# from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import os

SHOW=0
minPlateW=100
minPlateH=30
def order_box_points(box: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    `box` is a (4, 2) float ndarray from cv2.boxPoints (arbitrary order).
    """
    box = np.asarray(box, dtype=np.float32)
    rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]  # TL: smallest x+y
    rect[2] = box[np.argmax(s)]  # BR: largest x+y
    diff = np.diff(box, axis=1).ravel()
    rect[1] = box[np.argmin(diff)]  # TR: smallest (x - y)
    rect[3] = box[np.argmax(diff)]  # BL: largest (x - y)
    return rect
def detection_file(image_path: str, crop_img: np.ndarray, boxes, ml_format: str = "quad"):
    """
    Write a .txt next to `image_path` with four (x, y) corner points per line:
    TLx TLy TRx TRy BRx BRy BLx BLy, in the *crop image* coordinate space.

    - image_path: path to the saved cropped image (.png/.jpg)
    - crop_img: the cropped image array (for size reference)
    - boxes: list of np.ndarray (4,2) quads in crop space
    - ml_format: currently only "quad"
    """
    detection_path = os.path.splitext(image_path)[0] + ".txt"

    h, w = crop_img.shape[:2]
    lines = []
    for box in boxes:
        box = order_box_points(np.asarray(box, dtype=np.float32))
        # Clamp to crop bounds
        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1)

        if ml_format != "quad":
            raise ValueError(f"Unsupported ml_format: {ml_format}")

        flat = box.reshape(-1)  # TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy
        lines.append(" ".join(f"{v:.2f}" for v in flat.tolist()))

    with open(detection_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

def detectPlates(image):
        imHeight, imWidth = image.shape[:2]

        # if the width is greater than 640 pixels, then resize the image
        if image.shape[1] > 640:
            image = imutils.resize(image, width=640)
            
        # initialize the rectangular and square kernels to be applied to the image,
        # then initialize the list of license plate regions

        # Structuring Element first rectangular shape (width 15 hight 5)  second square shape
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  
        # list of potential regions with a license plate
        regions = []
        
        # convert the image to grayscale, and apply the blackhat operation to emphasize narrow regions with dark gray level
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, squareKernel, iterations=3) #rectKernel
        
        if (SHOW):
            plt.figure()
            plt.imshow(blackhat, cmap='gray')
            plt.title("Black Top Hat ")

        # numbers have vertical changes in gradient
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        if (SHOW):
            plt.figure()
            plt.imshow(gradX, cmap='gray')
            plt.title("Gradient X")
            plt.show()
        
        # gaussian blur with a 5 x 5 kernel to smooth detail and noise
        gradX = cv2.GaussianBlur(gradX, (7, 7), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel, iterations=2)
        if (SHOW):
            plt.figure()
            plt.imshow(gradX, cmap='gray')
            plt.title("Gausian Gx")
            plt.show()
            
        # el valor de corte se fija como el 40% del màximo
        ThrValue = (0.40) * np.max(gradX)
        ThrGradX = cv2.threshold(gradX, ThrValue, 255, cv2.THRESH_BINARY)[1]
        if (SHOW):
            plt.figure()
            plt.imshow(ThrGradX, cmap='gray')
            plt.title("Threshold Gx")
            plt.show()
        
        # some morphological operation to join parts first opening to remove small spots second to grow the area of license plate
        thresh = cv2.morphologyEx(ThrGradX, cv2.MORPH_OPEN, squareKernel, iterations=4)
        thresh = cv2.dilate(thresh, rectKernel, iterations=2)
        if(SHOW):
            plt.figure()
            plt.imshow(thresh, cmap='gray')
            plt.title("Possible license plates")
            plt.show()
                  
        # find contours in the thresholded image
        (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # grab the bounding box associated with the contour and compute the area and aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspectRatio = w / float(h)
            if (SHOW):
                print("BLOB ANALYSIS ->", x, y, w, h, aspectRatio, area)
            
            # condition of not touching the border of the image
            NotouchBorder = x != 0 and y != 0 and x+w != imWidth and y+h != imHeight
                
            # dimension conditions of a license plate
            keepArea = area > 3400 and area < 8000
            keepWidth = w > minPlateW and w <= 250
            keepHeight = h > minPlateH and h <= 60
            keepAspectRatio = 2.5 < w/h < 7

            # ensure the aspect ratio, width, and height of the bounding box fall within tolerable limits
            if all((NotouchBorder, keepAspectRatio, keepWidth, keepHeight, keepArea)):
                # compute the rotated bounding box of the region: 
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype("float32")
                regions.append(box)
                if (SHOW):
                    print("REGION BOX ACCEPTED->", box)

        # --------- Fallback for YOLO-cropped plates (shape-first quad) ---------
        if len(regions) == 0:
            H, W = image.shape[:2]
            gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.bilateralFilter(gray2, d=7, sigmaColor=50, sigmaSpace=50)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray2 = clahe.apply(gray2)

            med = np.median(gray2)
            lo = int(max(0, 0.66 * med))
            hi = int(min(255, 1.33 * med))
            edges = cv2.Canny(gray2, lo, hi)

            k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            edges = cv2.dilate(edges, k, iterations=1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)

            cnts2 = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]

            best_quad, best_score = None, -1.0
            img_area = float(H * W)

            for c in cnts2:
                if cv2.contourArea(c) < 50:
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    quad = approx.reshape(-1, 2).astype(np.float32)
                else:
                    rect = cv2.minAreaRect(c)
                    quad = cv2.boxPoints(rect).astype(np.float32)

                quad = order_box_points(quad)
                quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
                quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)

                w1 = np.linalg.norm(quad[0] - quad[1]); w2 = np.linalg.norm(quad[2] - quad[3])
                h1 = np.linalg.norm(quad[1] - quad[2]); h2 = np.linalg.norm(quad[3] - quad[0])
                w_est = max((w1 + w2) * 0.5, 1.0)
                h_est = max((h1 + h2) * 0.5, 1.0)
                ar = w_est / h_est
                if not (2.0 <= ar <= 8.5):
                    continue

                rect_area = w_est * h_est
                if rect_area <= 1.0:
                    continue

                fill_ratio = rect_area / img_area
                if not (0.20 <= fill_ratio <= 1.00):
                    continue

                cx, cy = quad[:, 0].mean(), quad[:, 1].mean()
                cx_n = abs(cx - (W - 1)/2.0) / (W + 1e-6)
                cy_n = abs(cy - (H - 1)/2.0) / (H + 1e-6)
                center_bonus = 1.0 - 0.5 * (cx_n + cy_n)
                score = rect_area * center_bonus

                if score > best_score:
                    best_score, best_quad = score, quad

            if best_quad is not None:
                regions.append(best_quad)
        # ----------------------------------------------------------------------

        return regions



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "cropped_plates")  # <-- match YOLO script

    for sub in ("Frontal", "Lateral"):
        folder = os.path.join(base_dir, sub)
        if not os.path.isdir(folder):
            print(f"Folder not found: {folder}")
            continue

        print(f"\nProcessing: {folder}")
        for name in os.listdir(folder):
            if not name.lower().endswith((".jpg", ".png")):
                continue
            img_path = os.path.join(folder, name)
            crop = cv2.imread(img_path)
            if crop is None:
                print(f"Cannot read: {img_path}")
                continue

            regions = detectPlates(crop)  # list of (4,2) boxes; may be empty
            if len(regions) > 0:
                quad = np.asarray(regions[0], dtype=np.float32)  # use the best/first box
            else:
                # fall back to whole crop rectangle
                h, w = crop.shape[:2]
                quad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)

            detection_file(img_path, crop, [quad], ml_format="quad")

            print(f"Wrote: {os.path.splitext(name)[0]}.txt")
