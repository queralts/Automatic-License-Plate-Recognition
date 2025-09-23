"""
This code computes some descriptive statistics, based on the values 
of the V channel (brightness values) of the images in our dataset.
With the goal of understanding the brightness distribution of 
these images according to the view (frontal or lateral).
"""

# NECESSARY LIBRARIES
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import seaborn as sns

# Loop through all images in a directory and compute descriptive statistics
def compute_overall_V_stats(path):

    all_pixels = []

    for img_file in os.listdir(path):
        image = cv2.imread(os.path.join(path, img_file))
        height, width = image.shape[0], image.shape[1]

        # Ensure all images are of the same size (4032x2268)
        # That is, they contain the same number of pixels
        if height != 2268 or width != 4032:
            image = cv2.resize(image, (4032, 2268))

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        V = hsv_image[:,:,2].flatten()

        # store all pixels for global stats
        all_pixels.extend(V)

    all_pixels = np.array(all_pixels)

    stats = {
        'global_mean': np.mean(all_pixels),
        'global_std': np.std(all_pixels),
        'global_min': np.min(all_pixels),
        'global_max': np.max(all_pixels),
        'all_pixels': all_pixels  
    }

    return stats

# Function to compute and show histograms
def compute_histogram(frontal_pixels, lateral_pixels):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Frontal histogram
    sns.histplot(frontal_pixels, bins=50, color="paleturquoise", kde=True, stat="density", ax=axes[0])
    axes[0].set_title("Frontal Images - Brightness Distribution")
    axes[0].set_xlabel("V channel value")
    axes[0].set_ylabel("Density")

    # Lateral histogram
    sns.histplot(lateral_pixels, bins=50, color="slateblue", kde=True, stat="density", ax=axes[1])
    axes[1].set_title("Lateral Images - Brightness Distribution")
    axes[1].set_xlabel("V channel value")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    plt.show()

#Function to sample pixels from all_pixels
def sample_pixels(arr, sample_size):
    if len(arr) > sample_size:
        idx = np.random.choice(len(arr), sample_size, replace=False)
        return arr[idx]
    return arr

# MAIN
if __name__ == "__main__":

    # DESCRIPTIVE STATISTIC - V CHANNEL (BRIGTHNESS VALUES):

    # Compute descriptive statistics for V channel in frontal images
    frontal_stats = compute_overall_V_stats("./dataset/Frontal")
    print("Frontal Images V Channel Statistics:")
    print("Frontal Images - Global Mean:", frontal_stats['global_mean'])
    print("Frontal Images - Global Std Dev:", frontal_stats['global_std'])
    print("\n")

    # Compute descriptive statistics for V channel in lateral images
    lateral_stats = compute_overall_V_stats("./dataset/Lateral")
    print("Lateral Images V Channel Statistics:")
    print("Lateral Images - V Channel Mean:", lateral_stats['global_mean'])
    print("Lateral Images - V Channel Std Dev:", lateral_stats['global_std'])

    # Sample from all the pixels and compute and show histograms for frontal and lateral images
    frontal_pixels = sample_pixels(frontal_stats['all_pixels'], 500000)
    lateral_pixels = sample_pixels(lateral_stats['all_pixels'], 500000)

    compute_histogram(frontal_pixels, lateral_pixels)