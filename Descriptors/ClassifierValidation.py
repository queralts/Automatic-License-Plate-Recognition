import os
import glob
import cv2
import numpy as np
from collections import defaultdict
from hog import FeatureHOG  # same descriptor used during training
from CharacterDescriptors_Example import SVM_digits_clf, SVM_alpha_clf  # trained models

# --- CONFIG ---
target_size = (30, 15)  # (height, width)
descHOG = FeatureHOG()  # same feature extractor used in training

# --- PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
cropped_characters_dir = os.path.join(script_dir, "../CharacterSegmentation/cropped_characters")

# --- UTILS ---
def is_digit(char_name):

    basename = os.path.basename(char_name)
    # Extract the main plate text before '_Char'
    plate_id = basename.split('_')[0]
    # The filename has structure: e.g. 0216KZP_Char3.png
    # Detect which char position this image corresponds to
    char_pos = int(basename.split('Char')[-1].split('.')[0]) - 1
    if char_pos < len(plate_id):
        ch = plate_id[char_pos]
        return ch.isdigit()
    return False

def extract_hog_features(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))
    features = descHOG.extract_image_features(img_resized)
    return np.array(features).reshape(1, -1)

# --- MAIN PROCESSING ---
def recognize_license_plates(cropped_dir):

    results = {}

    # Collect all character images
    all_images = sorted(glob.glob(os.path.join(cropped_dir, "*.png")))

    # Group by license plate prefix (everything before '_Char')
    grouped = defaultdict(list)
    for img_path in all_images:
        # Get file name
        filename = os.path.basename(img_path)

        # Skip invalid files that do not follow proper format
        if filename.startswith(("PXL")):
            continue

        plate_prefix = os.path.basename(img_path).split('_Char')[0]
        grouped[plate_prefix].append(img_path)

    # Process each license plate
    for plate_id, img_list in grouped.items():
        # Sort by Char number
        img_list = sorted(img_list, key=lambda x: int(os.path.basename(x).split('Char')[-1].split('.')[0]))

        recognized = []
        for img_path in img_list:
            try:
                features = extract_hog_features(img_path)
                if is_digit(img_path):
                    pred = SVM_digits_clf.predict(features)[0]
                else:
                    pred = SVM_alpha_clf.predict(features)[0]
                recognized.append(pred)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                recognized.append("?")  

        results[plate_id] = "".join(recognized)

    return results


if __name__ == "__main__":
    results = recognize_license_plates(cropped_characters_dir)
    print("\n=== Final Results ===")
    for plate, text in results.items():
        print(f"{plate} -> {text}")
