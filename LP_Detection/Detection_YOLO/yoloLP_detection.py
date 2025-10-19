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
from LicensePlateDetector import detectPlates
from LicensePlateDetector import detection_file

#### EXP-SET UP
# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
#DataDir=r'D:\Teaching\Grau\GrauIA\V&L\Challenges\Matricules\Dades\real_plates'

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to the model
model_path = os.path.join(script_dir, "LP-detection.pt")

#DataDir = os.path.join(script_dir, "../dataset")
DataDir = os.path.join(script_dir, "../../datasets/real_plates")
Views=['Frontal','Lateral']

# Load YOLO model
model = YOLO(model_path)
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
        """

        print(f"\nImage: {os.path.basename(imagePath)}")
        for box, cls_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            class_name = modelclasses[int(cls_id)]
            print(f"Detected {class_name} | Confidence: {conf:.2f} | BBox: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
        """
        obj=results[0].boxes.cls.cpu().numpy().astype(int) # Class IDs
        conf = results[0].boxes.conf.cpu().numpy() # Confidence scores

        # Store detected objects and confidence scores
        yoloObj[View].append(obj)
        yoloConf[View].append(conf)

        # Visualize detection with bounding boxes
        results[0].show()

        # Save cropped license plate image
        save_dir = os.path.join(script_dir, "cropped_plates", View)
        os.makedirs(save_dir, exist_ok=True)

        # Loop through all detections in the image
        for i, (box, cls_id, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf)):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = modelclasses[int(cls_id)]

            # Crop the detected region
            crop = image[y1:y2, x1:x2]

            # Skip invalid crops
            if crop.size == 0:
                continue

            # Build filename: originalImageName_YOLOPlate.png
            base_name = os.path.splitext(os.path.basename(imagePath))[0]
            out_name = f"{base_name}_YOLOPlate.png"
            out_path = os.path.join(save_dir, out_name)

            # If multiple plates in one image, append an index
            if i > 0:
                out_name = f"{base_name}_YOLOPlate_{i+1}.png"
                out_path = os.path.join(save_dir, out_name)

            # Save cropped plate
            cv2.imwrite(out_path, crop)
            print(f"Saved: {out_path}")

            # Now with the saved cropped plate we'll get its coordinates
            boxes_ml = detectPlates(crop)                 # list of 4-point boxes (in crop coords)
            detection_file(out_path, crop, boxes_ml)   


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