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

SHOW=1
minPlateW=100
minPlateH=30

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
            NotouchBorder = x!=0 and y!=0 and x+w!=imWidth and y+h!=imHeight

            
                
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

                regions.append(box)
                if (SHOW):
                    print("REGION BOX ACCEPTED->",box)
        return regions


if __name__ == "__main__":
    #image = cv2.imread("./dataset/Frontal/0216KZP.jpg")
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    image_path = os.path.join(script_dir, "../../new_images/with_Protocol/Frontal/0084HNC_2.jpg")
    image = cv2.imread(image_path)
    detectPlates(image)
    
"""
We want to explore the parameters of kernel size, threshold, area, aspect ratio and iterations.
In order to do that I''ve made a small trasnformation of the code to easily acces to thiese variables and change it as we want to see how they affect.
# --- Plate detection with tunable parameters ---
from imutils import perspective
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt

def detectPlates(
    image,
    # Preprocess / resize
    resize_max_width=640,
    # Kernels
    rect_kernel_size=(15, 5), 
    square_kernel_size=(3, 3),  
    # Morphology iteration counts
    blackhat_iters=3,
    close_iters=2,
    open_iters=4,
    dilate_iters=2,
    # Gradient smoothing
    grad_blur_ksize=7,         
    # Threshold (relative to max response)
    thresh_factor=0.40,         # raise -> fewer candidates; lower -> more noise
    # Geometric filters
    min_area=3400, max_area=8000,
    min_w=100,  max_w=250,
    min_h=30,   max_h=60,
    min_ar=2.5, max_ar=7.0,     
    show=True
):
    # Resize (keeping aspect ratio) to stabilize scales
    if image.shape[1] > resize_max_width:
        image = imutils.resize(image, width=resize_max_width)

    imH, imW = image.shape[:2]

    # Structuring elements
    rectKernel   = cv2.getStructuringElement(cv2.MORPH_RECT, rect_kernel_size)
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, square_kernel_size)

    # Grayscale + blackhat to highlight dark strokes on bright plate
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, squareKernel, iterations=blackhat_iters)
    if show:
        plt.figure(); plt.imshow(blackhat, cmap='gray'); plt.title("Blackhat")

    # Horizontal gradient (characters produce vertical transitions)
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    gmin, gmax = np.min(gradX), np.max(gradX)
    if gmax > gmin:
        gradX = (255 * ((gradX - gmin) / (gmax - gmin))).astype("uint8")
    else:
        gradX = np.zeros_like(gradX, dtype="uint8")

    if show:
        plt.figure(); plt.imshow(gradX, cmap='gray'); plt.title("Grad X (norm)")

    # Smooth and close to connect strokes into bars
    gradX = cv2.GaussianBlur(gradX, (grad_blur_ksize, grad_blur_ksize), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel, iterations=close_iters)
    if show:
        plt.figure(); plt.imshow(gradX, cmap='gray'); plt.title("Blurred + Close")

    # Relative threshold
    T = thresh_factor * np.max(gradX) if np.max(gradX) > 0 else 0
    ThrGradX = cv2.threshold(gradX, T, 255, cv2.THRESH_BINARY)[1]
    if show:
        plt.figure(); plt.imshow(ThrGradX, cmap='gray'); plt.title(f"Threshold @ {thresh_factor:.2f} * max")

    # Open to remove specks, then dilate to grow bars into plate-like blobs
    thresh = cv2.morphologyEx(ThrGradX, cv2.MORPH_OPEN, squareKernel, iterations=open_iters)
    thresh = cv2.dilate(thresh, rectKernel, iterations=dilate_iters)
    if show:
        plt.figure(); plt.imshow(thresh, cmap='gray'); plt.title("Candidates (open + dilate)")

    # Find external contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    regions = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = w / float(h) if h > 0 else 0

        # Avoid touching borders (often partial/false)
        no_touch_border = (x > 0) and (y > 0) and (x + w < imW) and (y + h < imH)

        # Geometry gates
        keep_area  = (min_area <= area <= max_area)
        keep_w     = (min_w <= w <= max_w)
        keep_h     = (min_h <= h <= max_h)
        keep_ar    = (min_ar <= ar <= max_ar)

        if all((no_touch_border, keep_area, keep_w, keep_h, keep_ar)):
            rect = cv2.minAreaRect(c)
            box  = cv2.boxPoints(rect)
            regions.append(box)

            if show:
                print(f"ACCEPTED: x={x} y={y} w={w} h={h} ar={ar:.2f} area={area:.0f}")

    if show:
        plt.show()

    return regions

if __name__ == "__main__":
    img = cv2.imread("./dataset/Frontal/0216KZP.jpg")
    boxes = detectPlates(
        img,
        # Try nudging these while inspecting results:
        rect_kernel_size=(17, 5),
        square_kernel_size=(3, 3),
        blackhat_iters=3,
        close_iters=2,
        open_iters=4,
        dilate_iters=2,
        thresh_factor=0.38,       # lower if plates are faint; raise if many false blobs
        min_area=3000, max_area=12000,
        min_w=90,  max_w=280,
        min_h=28,  max_h=80,
        min_ar=2.2, max_ar=7.5,
        show=True
    )
    # boxes contains the quadrilaterals of candidate plate regions
"""