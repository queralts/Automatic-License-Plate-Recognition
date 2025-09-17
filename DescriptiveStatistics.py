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


# Function to find mean and standard deviation of V channel of an image
def find_v_channel_stats(img):

    # convert RGB image to HSV and extract V channel
    image = cv2.imread(img)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    V = hsv_image[:,:,2]

    """
    plt.figure()
    plt.imshow(hsv,cmap='gray')
    plt.title("HSV image")
    plt.show()
    """

    # Compute mean and standard deviation of V channel for that particular image
    mean_val = np.mean(V)
    std_val = np.std(V)

    return mean_val, std_val


# Loop through all images in a directory and compute descriptive statistics
def compute_overall_V_stats(path):
    means_list = []
    stds_list = []
    stats = {}

    for img_file in os.listdir(path):
        mean, std = find_v_channel_stats(os.path.join(path, img_file))
        means_list.append(mean)
        stds_list.append(std)
    
    overall_mean = np.mean(means_list)
    overall_std = np.mean(stds_list)
    min_val = np.min(means_list)
    max_val = np.max(means_list)

    stats['means'] = means_list
    stats['stds'] = stds_list
    stats['overall_mean'] = overall_mean
    stats['overall_std'] = overall_std
    stats['min'] = min_val
    stats['max'] = max_val
    
    return stats

# Function to compute and show histograms
def compute_histogram(means, stds, frontal_or_lateral):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  

    # Histogram of means
    sns.histplot(means, bins=20, kde=True, color='paleturquoise', ax=axes[0])
    axes[0].set_title("V Channel Means: " + frontal_or_lateral + " Images")

    # Histogram of std deviations
    sns.histplot(stds, bins=20, kde=True, color='slateblue', ax=axes[1])
    axes[1].set_title("V Channel Std Devs: " + frontal_or_lateral + " Images")

    plt.tight_layout()  
    plt.show()


# MAIN
if __name__ == "__main__":

    # DESCRIPTIVE STATISTIC - V CHANNEL (BRIGTHNESS VALUES):

    # Compute descriptive statistics for V channel in frontal images
    frontal_stats = compute_overall_V_stats("dataset/Frontal")
    print("Frontal Images - V Channel Mean:", frontal_stats['overall_mean'])
    print("Frontal Images - V Channel Std Dev:", frontal_stats['overall_std'])
    print("Frontal Images - V Channel Min:", frontal_stats['min'])
    print("Frontal Images - V Channel Max:", frontal_stats['max'])
    print("\n")

    # Compute descriptive statistics for V channel in lateral images
    lateral_stats = compute_overall_V_stats("dataset/Lateral")
    print("Lateral Images - V Channel Mean:", lateral_stats['overall_mean'])
    print("Lateral Images - V Channel Std Dev:", lateral_stats['overall_std'])
    print("Lateral Images - V Channel Min:", lateral_stats['min'])
    print("Lateral Images - V Channel Max:", lateral_stats['max'])

    # Compute and show histograms for frontal images
    compute_histogram(frontal_stats['means'], frontal_stats['stds'], "Frontal")
    compute_histogram(lateral_stats['means'], lateral_stats['stds'], "Lateral")