# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:52:26 2025

Example of WaterShed Segmentation

Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres"
__license__ = "GPL"
__email__ = "debora,gtorres@cvc.uab.es"

"""

import cv2
import numpy as np
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get file PlateRegions.npz
img = os.path.join(script_dir, "../cropped_real_plates/Frontal/0216KZP_MLPlate0.png")

plate = cv2.imread(img)
#img = cv2.imread("../new_images/with_Protocol/Frontal/0084HNC_1.jpg")

gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
img = gray_plate  

# Supongamos que 'img' es la imagen original en gris
# 1. Umbral binario
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("Step 1 - Binary (OTSU_INV)", binary)
cv2.waitKey(0)


# 2. Ruido y separación de letras cercanas 
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("Step 2 - Morphological Opening", opening)
cv2.waitKey(0)


# 3. Distancia transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
cv2.imshow("Step 3 - Distance Transform (normalized)", cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype("uint8"))
cv2.waitKey(0)

_, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
cv2.imshow("Step 3 - Sure Foreground", sure_fg)
cv2.waitKey(0)

unknown = cv2.subtract(opening, sure_fg)
cv2.imshow("Step 3 - Unknown Region", unknown)
cv2.waitKey(0)

# 4. Marcadores
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown==255] = 0

cv2.imshow("Step 4 - Markers (visualized as uint8)", (markers * (255 / markers.max())).astype("uint8"))
cv2.waitKey(0)

# 5. Watershed
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
markers = cv2.watershed(img_color, markers)
img_color[markers == -1] = [0,0,255]  # Bordes en rojo

# 6. Visualización
cv2.imshow("Watershed", img_color)
cv2.waitKey(0)
#used to see each image and manually move to the next step
cv2.destroyAllWindows()

