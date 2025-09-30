# import the necessary packages
import numpy as np
import cv2
import glob
import os
from imutils import perspective
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from ultralytics import YOLO


#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
#DataDir=r'D:\Teaching\Grau\GrauIA\V&L\Challenges\Matricules\Dades\real_plates'

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

#DataDir = os.path.join(script_dir, "../dataset")
DataDir = os.path.join(script_dir, "../new_images")
Views=['Frontal','Lateral']

# Load YOLO model
model = YOLO("Detection_YOLO/LP-detection.pt")
# Trained on COCO DataSet
modelclasses=np.array(list(model.names.values()))
model.device # By default model is in GPU device: model=model.to('cpu') for execution in CPU


#### COMPUTE PROPERTIES FOR EACH VIEW
yoloConf={}
yoloObj={}
for View in Views:
    
    ImageFiles=sorted(glob.glob(os.path.join(DataDir,View,'*.jpg')))
    yoloConf[View]=[] # stores confidence scores
    yoloObj[View]=[] # stores detected objects

    # loop over the images
    for imagePath in ImageFiles:
        # load the image
        image = cv2.imread(imagePath)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(img_rgb) 

        obj=results[0].boxes.cls.cpu().numpy().astype(int) # Class IDs
        conf = results[0].boxes.conf.cpu().numpy() # Confidence scores

        # Store detected objects and confidence scores
        yoloObj[View].append(obj)
        yoloConf[View].append(conf)

        # Visualize detection with bounding boxes
        #results[0].show()

####  EXPLORE OBJECT DISTRIBUTION FOR EACH VIEW USING HISTOGRAMS AND BOXPLOTS

## EXPLORE OBJECTS DETECTED PER VIEW:

# Flatten lists into a single array
frontal_objs = np.concatenate(yoloObj['Frontal'])
lateral_objs = np.concatenate(yoloObj['Lateral'])

print("Detected plates in frontal images:", len(frontal_objs))
print("Total frontal images (total plates to detect): 33")

print("Detected plates in lateral images:", len(lateral_objs))
print("Total lateral images (total plates to detect): 66")

# Map class IDs to class names
frontal_labels = np.array([modelclasses[i] for i in frontal_objs])
lateral_labels = np.array([modelclasses[i] for i in lateral_objs])

# Create a DataFrame
df = pd.DataFrame({
    'View': ['Frontal']*len(frontal_labels) + ['Lateral']*len(lateral_labels),
    'Class': np.concatenate([frontal_labels, lateral_labels])
})

# Plot histogram

plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Class', hue='View', palette={'Frontal': 'paleturquoise', 'Lateral': 'plum'})
plt.ylabel("Number of Detections")
plt.title("Class Distribution Across Views")
plt.show()

## EXPLORE CONFIDENCE SCORE DISTRIBUTION PER VIEW:

# Frontal view confidence scores distribution------

conf_list_frontal = []
label_list_frontal = []

for objs, confs in zip(yoloObj['Frontal'], yoloConf['Frontal']):
    if len(objs) > 0:  # make sure there are detections
        conf_list_frontal.extend(confs)
        label_list_frontal.extend([modelclasses[i] for i in objs])

df_conf_frontal = pd.DataFrame({
    'Confidence': conf_list_frontal,
    'Object': label_list_frontal
})

# Create and show Boxplot

plt.figure(figsize=(12,6))
sns.boxplot(data=df_conf_frontal, x='Object', y='Confidence', palette='pastel')
plt.title("Frontal View: Confidence Distribution per class")
plt.ylabel("Confidence Score")
plt.xlabel("Detected Class")
plt.show()

# Lateral view confidence scores distribution------

conf_list_lateral = []
label_list_lateral = []

for objs, confs in zip(yoloObj['Lateral'], yoloConf['Lateral']):
    if len(objs) > 0:  # make sure there are detections
        conf_list_lateral.extend(confs)
        label_list_lateral.extend([modelclasses[i] for i in objs])


df_conf_lateral = pd.DataFrame({
    'Confidence': conf_list_lateral,
    'Object': label_list_lateral
})

# Create and show Boxplot

plt.figure(figsize=(12,6))
sns.boxplot(data=df_conf_lateral, x='Object', y='Confidence', palette='pastel')
plt.title("Lateral View: Confidence Distribution per class")
plt.ylabel("Confidence Score")
plt.xlabel("Detected Class")
plt.show()