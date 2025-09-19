"""
This code computes descriptive statistics based on the ROTATION ANGLE (in degrees)
of the detected license plates in our dataset. The goal is to understand the angle
distribution of these images according to the view (frontal or lateral) and obtain
simple "prototypes" (means) for FRONTAL, LATERAL, and ALL images.

Notes:
- Angle convention: degrees relative to the horizontal, positive = counter-clockwise.
- Angles are normalized to the range [-90, +90], which is typical for elongated objects
  (0° and 180° are effectively the same orientation for a plate).
"""

# NECESSARY LIBRARIES
import os
import math
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import seaborn as sns


def norm_plate_angle(rect):
    """
    Normalize cv2.minAreaRect angle to degrees relative to horizontal.
    Positive = counter-clockwise. Output in [-90, 90].
    """
    (_, _), (w, h), a = rect  # OpenCV gives a in (-90, 0]
    if w < h: #taller than wider
        angle = a + 90.0 
    else:
        angle = a
    if angle > 90.0:
        angle -= 180.0
    if angle <= -90.0:
        angle += 180.0
    return float(angle)

# Debora's code, i'm just trying things

def detectPlates(image_bgr):
    """
    Detect license-plate-like regions and return a list of tuples:
      (box_points, angle_deg)
    - box_points: 4x2 float array with the rotated rectangle corners
    - angle_deg: normalized angle in degrees ([-90, +90])
    """
    # Optional resize for stability/speed
    if image_bgr.shape[1] > 640:
        image_bgr = imutils.resize(image_bgr, width=640)

    imHeight, imWidth = image_bgr.shape[:2]

    # Structuring elements
    rectKernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    regions = []

    # Grayscale + blackhat to emphasize dark narrow regions
    #Blackhat emphasizes dark text on ligh background
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, squareKernel, iterations=3)

    # Horizontal gradient (Scharr via ksize=-1)
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    minVal = float(np.min(gradX))
    maxVal = float(np.max(gradX))
    rng = maxVal - minVal
    if rng == 0.0:
        rng = 1.0
    gradX = (255.0 * ((gradX - minVal) / rng)).astype("uint8")

    # Blur + close to connect strokes
    gradX = cv2.GaussianBlur(gradX, (7, 7), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel, iterations=2)

    # Binary threshold (40% of max)
    ThrValue = int(0.40 * float(np.max(gradX)))
    ThrGradX = cv2.threshold(gradX, ThrValue, 255, cv2.THRESH_BINARY)[1]

    # Clean + grow regions
    thresh = cv2.morphologyEx(ThrGradX, cv2.MORPH_OPEN, squareKernel, iterations=4)
    thresh = cv2.dilate(thresh, rectKernel, iterations=2)

    # Contours
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Heuristic filters (tuned for width<=640 images)
    i = 0
    while i < len(cnts):
        c = cnts[i]
        x, y, w, h = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if h == 0:
            aspectRatio = 999.0
        else:
            aspectRatio = w / float(h)

        NotouchBorder = x != 0 and y != 0 and x + w != imWidth and y + h != imHeight
        keepArea = area > 3400.0 and area < 8000.0
        keepWidth = w > 100 and w <= 250
        keepHeight = h > 30 and h <= 60
        keepAspectRatio = aspectRatio > 2.5 and aspectRatio < 7.0

        if NotouchBorder and keepAspectRatio and keepWidth and keepHeight and keepArea:
            rect = cv2.minAreaRect(c)     # ((cx, cy), (W, H), angle)
            box = cv2.boxPoints(rect)     # 4 corner points
            angle_deg = norm_plate_angle(rect)
            regions.append((box, angle_deg))
        i = i + 1

    return regions


def image_angle(img_path):
    """
    Returns one angle (degrees) for a single image, or None if not found or unreadable.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    regions = detectPlates(img)
    if regions is None:
        return None
    if len(regions) == 0:
        return None
    box, ang = regions[0]
    return float(ang)


def compute_overall_angle_stats(path):
    """
    Loop through all images in a directory and compute angle statistics.
    Returns a dict with:
      'angles' (list), 'overall_mean', 'overall_std', 'min', 'max', 'count'
    Non-image files are ignored. Images without detected plates are skipped.
    """
    angles = []
    files = []
    if not os.path.isdir(path):
        print("Directory not found:", path)
        print("Couldn't compute any statistics.")
        return None

    listing = os.listdir(path) #list of all file names
    for fname in listing:
        folder_path = os.path.join(path, fname) #folder + name
        ang = image_angle(folder_path)
        if ang is not None:
            angles.append(ang)
            #files.append(fname) if we want to use it after 


    stats = {}
    stats['angles'] = angles #save the list of angles collected before
    """
    # Case of no angles, needed?¿
    if len(angles) == 0:
        stats['overall_mean'] = np.nan
        stats['overall_std'] = np.nan
        stats['min'] = np.nan
        stats['max'] = np.nan
        stats['count'] = 0
        return stats 
    """
    # Convert to array bc we want numerical work easier
    arr = np.array(angles, dtype=float)

    # Mean
    s = 0.0
    i = 0
    while i < len(arr):
        s = s + float(arr[i])
        i = i + 1
    overall_mean = s / float(len(arr))

    # Std 
    # Formula: std = sqrt( mean( (x - mean)^2 ) )
    s2 = 0.0
    j = 0
    while j < len(arr):
        diff = float(arr[j]) - overall_mean
        s2 = s2 + diff * diff
        j = j + 1
    overall_std = math.sqrt(s2 / float(len(arr)))

    # Min/Max
    current_min = float(arr[0])
    current_max = float(arr[0])
    k = 1
    while k < len(arr):
        val = float(arr[k])
        if val < current_min:
            current_min = val
        if val > current_max:
            current_max = val
        k = k + 1

    stats['overall_mean'] = overall_mean
    stats['overall_std'] = overall_std
    stats['min'] = current_min
    stats['max'] = current_max
    stats['count'] = len(arr)
    return stats


def compute_histogram_angles(angles, frontal_or_lateral):
    """
    Plot histogram (and KDE) of angles in degrees for a given group label.
    """
    if angles is None or len(angles) == 0:
        print("No angles to plot for:", frontal_or_lateral)
        return

    plt.figure(figsize=(7, 5))
    sns.histplot(angles, bins=21, kde=True, color='slateblue')
    plt.title("Plate Angle Distribution: " + frontal_or_lateral + " Images")
    plt.xlabel("Angle (degrees, +CCW, range [-90, +90])")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def compute_all_images_stats(frontal_stats, lateral_stats):

    # Combine angles (Frontal + Lateral) and compute overall stats
    all_angles = []
    if 'angles' in frontal_stats and frontal_stats['angles'] is not None:
        i = 0
        while i < len(frontal_stats['angles']):
            all_angles.append(frontal_stats['angles'][i])
            i = i + 1
    if 'angles' in lateral_stats and lateral_stats['angles'] is not None:
        j = 0
        while j < len(lateral_stats['angles']):
            all_angles.append(lateral_stats['angles'][j])
            j = j + 1

    # Compute overall stats for ALL
    all_stats = {}
    all_stats['angles'] = all_angles 
    if len(all_angles) > 0:
        # mean
        s_all = 0.0
        i_all = 0
        while i_all < len(all_angles):
            s_all = s_all + float(all_angles[i_all])
            i_all = i_all + 1
        mean_all = s_all / float(len(all_angles))

        # std
        s2_all = 0.0
        j_all = 0
        while j_all < len(all_angles):
            diff_all = float(all_angles[j_all]) - mean_all
            s2_all = s2_all + diff_all * diff_all
            j_all = j_all + 1
        std_all = math.sqrt(s2_all / float(len(all_angles)))

        # min/max
        current_min_all = float(all_angles[0])
        current_max_all = float(all_angles[0])
        k_all = 1
        while k_all < len(all_angles):
            val_all = float(all_angles[k_all])
            if val_all < current_min_all:
                current_min_all = val_all
            if val_all > current_max_all:
                current_max_all = val_all
            k_all = k_all + 1

        all_stats['overall_mean'] = mean_all
        all_stats['overall_std'] = std_all
        all_stats['min'] = current_min_all
        all_stats['max'] = current_max_all
        all_stats['count'] = len(all_angles)

        return all_stats
    
    else:
        return None

# --------- MAIN ---------

if __name__ == "__main__":

    # DESCRIPTIVE STATISTICS — ANGLES (DEGREES):

    frontal = "dataset/Frontal"
    lateral = "dataset/Lateral"

    # Compute descriptive statistics for angles in frontal images
    frontal_stats = compute_overall_angle_stats(frontal)
    # Compute descriptive statistics for angles in lateral images
    lateral_stats = compute_overall_angle_stats(lateral)

    if frontal_stats != None and lateral_stats != None:

        print("Frontal Images - Angle Mean:", round(frontal_stats['overall_mean'], 4))
        print("Frontal Images - Angle Std Dev:", round(frontal_stats['overall_std'], 4))
        print("Frontal Images - Angle Min:", round(frontal_stats['min'], 4))
        print("Frontal Images - Angle Max:", round(frontal_stats['max'], 4))
        print("Frontal Images - Count:", frontal_stats['count'])
        print("")

        print("Lateral Images - Angle Mean:", round(lateral_stats['overall_mean'], 4))
        print("Lateral Images - Angle Std Dev:", round(lateral_stats['overall_std'], 4))
        print("Lateral Images - Angle Min:", round(lateral_stats['min'], 4))
        print("Lateral Images - Angle Max:", round(lateral_stats['max'], 4))
        print("Lateral Images - Count:", lateral_stats['count'])
        print("")

        # Compute combined stats for all images
        all_stats = compute_all_images_stats(frontal_stats, lateral_stats)

        if all_stats != None:
            print("All Images - Angle Mean:", round(all_stats['overall_mean'], 4))
            print("All Images - Angle Std Dev:", round(all_stats['overall_std'], 4))
            print("All Images - Angle Min:", round(all_stats['min'], 4))
            print("All Images - Angle Max:", round(all_stats['max'], 4))
            print("All Images - Count:", all_stats['count'])

            # Histograms
            compute_histogram_angles(frontal_stats['angles'], "Frontal")
            compute_histogram_angles(lateral_stats['angles'], "Lateral")
            compute_histogram_angles(all_stats['angles'], "All images")
