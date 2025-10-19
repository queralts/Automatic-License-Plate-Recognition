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

def detectPlates(image):
        imHeight, imWidth = image.shape[:2]

        # if the width is greater than 640 pixels, then resize the image
        resized = image
        scale_x = scale_y = 1.0
        if image.shape[1] > 640:
            resized = imutils.resize(image, width=640)
            H1, W1 = resized.shape[:2]
            scale_x = imWidth / float(W1)
            scale_y = imHeight / float(H1)
            
        # initialize the rectangular and square kernels to be applied to the image,
        # then initialize the list of license plate regions

        
        # Structuring Element first rectangular shape (width 15 hight 5)  second square shape
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  
        # list of potential regions with a license plate
        regions = []
        

        # convert the image to grayscale, and apply the blackhat operation to emphasize narrow regions with dark gray level
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, squareKernel,iterations =3) #rectKernel
        
        if (SHOW):
            plt.figure()
            plt.imshow(blackhat,cmap='gray')
            plt.title("Black Top Hat ")

        # numbers have vertical changes in gradient
        gradX = cv2.Sobel(blackhat,ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        if (SHOW):
            plt.figure()
            plt.imshow(gradX,cmap='gray')
            plt.title("Gradient X")
            plt.show()
        
        # gaussian blur with a 5 x 5 kernel to smooth detail and noise
        gradX = cv2.GaussianBlur(gradX, (7, 7), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel,iterations = 2)
        if (SHOW):
            plt.figure()
            plt.imshow(gradX,cmap='gray')
            plt.title("Gausian Gx")
            plt.show()
            
        
        # el valor de corte se fija como el 40% del màximo
        ThrValue= (0.40)*np.max(gradX)
        ThrGradX = cv2.threshold(gradX, ThrValue, 255, cv2.THRESH_BINARY)[1]
        if (SHOW):
            plt.figure()
            plt.imshow(ThrGradX,cmap='gray')
            plt.title("Threshold Gx")
            plt.show()
        

        # some morphological operation to join parts first opening to remove small spots second to grow the area of license plate
        thresh = cv2.morphologyEx(ThrGradX, cv2.MORPH_OPEN, squareKernel,iterations = 4 )
        thresh = cv2.dilate(thresh, rectKernel, iterations=2)
        if(SHOW):
            plt.figure()
            plt.imshow(thresh,cmap='gray')
            plt.title("Possible license plates")
            plt.show()
                  
        # find contours in the thresholded image
        (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Hr, Wr = resized.shape[:2]

        # loop over the contours
        for c in cnts:
            # grab the bounding box associated with the contour and compute the area and
            # aspect ratio
            (x,y,w, h) = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspectRatio = w / float(h)
            if (SHOW):
                print("BLOB ANALYSIS ->",x,y,w,h,aspectRatio,area)
            
            # condition of not touching the border of the image
            NotouchBorder = (x != 0) and (y != 0) and (x + w != Wr) and (y + h != Hr)

            
                
            # dimension conditions of a license plate
            keepArea = area > 3400 and area < 8000
            keepWidth = w > minPlateW and w <= 250
            keepHeight = h > minPlateH and h <= 60
            keepAspectRatio = 2.5<w/h<7

            # ensure the aspect ratio, width, and height of the bounding box fall within
            # tolerable limits, then update the list of license plate regions
            if all((NotouchBorder,keepAspectRatio,keepWidth,keepHeight,keepArea)):
                # compute the rotated bounding box of the region: 
                
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                # Rescale to coordinates of the original image
                box[:, 0] = box[:, 0] * scale_x
                box[:, 1] = box[:, 1] * scale_y
                box[:, 0] = np.clip(box[:, 0], 0, imWidth - 1)
                box[:, 1] = np.clip(box[:, 1], 0, imHeight - 1)
                regions.append(box)
                if (SHOW):
                    print("REGION BOX ACCEPTED->",box)
        return regions

def order_box(box):
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    `box` is a (4, 2) float ndarray from cv2.boxPoints (arbitrary order).
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]  # top-left has the smallest x+y
    rect[2] = box[np.argmax(s)]  # bottom-right has the largest x+y
    diff = np.diff(box, axis=1).ravel()
    rect[1] = box[np.argmin(diff)]  # top-right has the smallest (x - y)
    rect[3] = box[np.argmax(diff)]  # bottom-left has the largest (x - y)
    return rect

def detection_file(image_path, crop_img, boxes, ml_format="quad"):
    """
    Save a text file next to each cropped image containing the plate coordinates
    as four (x, y) corner points: TL, TR, BR, BL.
    
    - image_path: path to the saved cropped image (.png)
    - crop_img: the cropped license plate image (used for size reference)
    - boxes: list of 4-point np.ndarray (4,2) with coordinates in crop space
    """
    detection_path = os.path.splitext(image_path)[0] + ".txt"

    lines = []
    for box in boxes or []:  # handles None
        rect = order_box(box)  # TL, TR, BR, BL
        x1, y1 = rect[0]; x2, y2 = rect[1]; x3, y3 = rect[2]; x4, y4 = rect[3]
        lines.append(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x3:.2f} {y3:.2f} {x4:.2f} {y4:.2f}")

    if not lines: 
        h, w = crop_img.shape[:2]
        lines.append(f"0 0 {w-1} 0 {w-1} {h-1} 0 {h-1}")

    # Write all detections to the text file
    with open(detection_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[ML] Detection file saved: {detection_path} ({len(lines)} detection(s))")
if __name__ == "__main__":
    #image = cv2.imread("./dataset/Frontal/0216KZP.jpg")
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    image_path = os.path.join(script_dir, "../../new_images/with_Protocol/Frontal/0084HNC_2.jpg")
    image = cv2.imread(image_path)
    detectPlates(image)
    
