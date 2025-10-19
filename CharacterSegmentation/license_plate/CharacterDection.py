# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:45:11 2025
Universitat Autonoma de Barcelona

__author__ = Xavier Roca
__license__ = "GPL"
__email__ = "xavier.roca@cvc.uab.es"
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
import math
# Blob detectors
from skimage.feature import blob_log
from scipy.ndimage import gaussian_laplace


# FUNCTIONS ----------------------------------------

# ----------------------------------------

"""
Function to apply LoG blob detection on a thresholded plate image
Returns an array of blobs: (y, x, radius)
"""

def detect_blobs_log(img, max_sigma=20, num_sigma=10, threshold=0.2, borders=False):

    # convert to float [0,1]
    norm = img.astype(float) / 255.0

    blobs = blob_log(norm, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, exclude_border=borders)
    # convert sigma to radius
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    return blobs

# ----------------------------------------

"""
Function to apply LoG blob detection on a thresholded plate image
Returns an array of blobs: candidate regions, log response
"""

def gaussian_laplace_detector(img, sigma, threshold):

    gray = img.astype(np.float32) / 255.0  # normalize

    # Negative sign is used because characters are white on black
    # blobs become positive peaks
    log_response = -gaussian_laplace(gray, sigma=sigma)
    # Threshold the LoG response to find blobs
    candidates = (log_response > threshold).astype(np.uint8) * 255

    return candidates, log_response

# ----------------------------------------

"""
Post-process the LoG response to obtain character region candidates
Returns a binary mask and a list of bounding boxes
"""

def postprocess_log_candidates(log_response, min_area=50, max_area=2000):

    # Threshold the response
    binary_candidates = (log_response > 0).astype(np.uint8) * 255

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary_candidates = cv2.morphologyEx(binary_candidates, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_candidates = cv2.morphologyEx(binary_candidates, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(binary_candidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    mask = np.zeros_like(binary_candidates)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        if min_area < area < max_area:  # filter out tiny or huge regions
            cv2.drawContours(mask, [c], -1, 255, -1)
            bboxes.append((x, y, w, h))

    return mask, bboxes

# ----------------------------------------

"""
Function to preprocess the plate image before detecting characters
Returns: the enhanced image
"""

def preprocess_plate(thresh, close_size=(2,2), open_size=(2,2), iterations=1):

    img = thresh.copy()
    
    # Closing to fill small gaps in strokes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, close_size)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close, iterations=iterations)
    
    # Opening to remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, open_size)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open, iterations=iterations)
    
    return img

# ----------------------------------------

def detectCharacterCandidates(image, reg, SHOW=0):
    # apply a 4-point transform to extract the license plate
    plate = perspective.four_point_transform(image, reg)
    plate = imutils.resize(plate, width=400)

    if (SHOW):
        cv2.imshow("Perspective Transform", plate)
    
    # extract the Value component from the HSV color space and apply adaptive thresholding
    # to reveal the characters on the license plate
    V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]        
    thresh = cv2.adaptiveThreshold(V, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    if (SHOW):
        cv2.imshow("Adaptative Threshold", thresh)

    # Structuring Element  rectangular shape (width 1 hight 3)  
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    # clossing follow by an opening to fill holes and join regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel,iterations = 2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernel,iterations = 2)

    # resize the license plate region 
    thresh = imutils.resize(thresh, width=400)

    if (SHOW):
        print("START DIMENSIONAL ANALYSIS")
    
    DimY, DimX = thresh.shape[:2]
    (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    MycharCandidates = np.zeros(thresh.shape, dtype="uint8")

    for c in cnts:
        # grab the bounding box associated with the contour and compute the area and
        # aspect ratio
        (x,y,w, h) = cv2.boundingRect(c)

        # condition of not touching the border of the region
        NotouchBorder = x!=0 and y!=0 and x+w!=DimX and y+h!=DimY
        
        if (NotouchBorder):
            

            # hight ratio of the blob numbers are blobs with a hight near the DimY
            hW=(h / float(DimY))
            area = cv2.contourArea(c)
            if (SHOW):
                print("AREA: ",area," ASPECT: ",hW)
            heightRatio = 0.5 < hW <0.9
            if (area>300) and (heightRatio):
                
                hull = cv2.convexHull(c)
                cv2.drawContours(MycharCandidates, [hull], -1, 255, -1)

    if (SHOW):            
        print("END DIMENSIONAL ANALYSIS")

    

    # return the license plate region object containing the license plate, the thresholded
    # license plate, and the character candidates
    return plate, thresh, MycharCandidates

# ----------------------------------------


# MAIN ----------------------------------------

if __name__ == "__main__":

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get file PlateRegions.npz
    file_frontal = os.path.join(script_dir, "../../cropped_real_plates/Frontal/PlateRegions.npz")
    file_lateral = os.path.join(script_dir, "../../cropped_real_plates/Lateral/PlateRegions.npz")

    data_frontal = np.load(file_frontal,allow_pickle=True) 
    data_lateral = np.load(file_lateral, allow_pickle=True)

    #use data.files to get data variables names 

    # Get data from frontal images
    regionsImCropped_frontal = data_frontal['regionsImCropped']
    regionsIm_frontal = data_frontal['regionsIm']
    imageIDs_frontal = data_frontal['imID']

    # Get data from lateral images
    regionsImCropped_lateral = data_lateral['regionsImCropped']
    regionsIm_lateral = data_lateral['regionsIm']
    imageIDs_lateral = data_lateral['imID']

    # Apply pipeline on all lateral cropped images 
    for img, reg in zip(imageIDs_frontal, regionsImCropped_frontal):
        image_path = os.path.join(script_dir, "../../cropped_real_plates/Frontal", img+"_MLPlate0.png")
        image = cv2.imread(image_path)

        # Use character detection pipeline on that image
        plate, thresh, _ = detectCharacterCandidates(image, reg, SHOW=0)

        # Preprocess plate to improve detectors performance
        # Remove comment to see the effect of preprocessing:
        #thresh = preprocess_plate(thresh, close_size=(1,5), open_size=(2,2))

        # Detect blobs with LoG, we won't use list of candidates
        blobs = detect_blobs_log(thresh, max_sigma=10, num_sigma=10, threshold=0.1, borders=True)

        # Detect blobs with gaussian_laplace function from scipy

        # Estimate the appropriate isotropic Gaussian sigma for LoG detection
        # using the radius of blobs detected by blob_log
        median_r = np.median(blobs[:, 2])
        sigma = median_r / 2*np.sqrt(2)
        candidates, log_response = gaussian_laplace_detector(thresh, sigma, 0.03)

        # Postprocess LoG candidates (initial cleaning)
        mask, bboxes = postprocess_log_candidates(log_response)

        # Save cropped license plate characters
        save_dir = os.path.join(script_dir, "../cropped_characters")
        os.makedirs(save_dir, exist_ok=True)

        base_name = img  

        for i, (x, y, w, h) in enumerate(bboxes):
            # Crop the character from the plate image
            char_crop = plate[y:y+h, x:x+w]

            # Skip invalid crops
            if char_crop.size == 0:
                continue

            # Build filename: originalImageName_YOLOPlate_Char#.png
            out_name = f"{base_name}_YOLOPlate_Char{i+1}.png"
            out_path = os.path.join(save_dir, out_name)

            # Save as PNG
            cv2.imwrite(out_path, char_crop)
            print(f"Saved character: {out_path}")

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))  # 5 plots

        axs[0].imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Cropped Plate"); axs[0].axis("off")

        axs[1].imshow(thresh, cmap="gray")
        axs[1].set_title("Thresholded Plate"); axs[1].axis("off")

        axs[2].imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        axs[2].set_title("LoG blobs (skimage)"); axs[2].axis("off")
        for y, x, r in blobs:
            circ = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            axs[2].add_patch(circ)

        axs[3].imshow(candidates, cmap="gray")
        axs[3].set_title("LoG candidates (scipy)"); axs[3].axis("off")

        axs[4].imshow(mask, cmap="gray")
        axs[4].set_title("Postprocessed Regions"); axs[4].axis("off")
        plt.show()
