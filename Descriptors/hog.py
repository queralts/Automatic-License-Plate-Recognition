#!/usr/bin/env python

"""
Histograms of Oriented Gradients feature extraction 
wrapper.


Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres"
__license__ = "GPL"
__email__ = "debora,gtorres@cvc.uab.es"


"""

import os
import glob
from skimage import feature
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import pickle

class FeatureHOG:
    """
    Class for the computation of HOG descriptor (using feature.hog): 
        orientations: Number of orientations for gradient computation
        Default: orientations=8
        pixels_per_cell: Size of the cell for orientation computation (default, (4,4))
        cells_per_block: Cells for each block for cell histograms normalization
    """
    def __init__(self,orientations=8,pixels_per_cell = (4,4),cells_per_block=(2,2),
                 visualize=False,feature_vector=True):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell 
        self.cells_per_block = cells_per_block
        self.block_norm = 'L2-Hys'
        self.visualize = visualize
        self.feature_vector = feature_vector

    def extract_pixel_features(self, image):
        feats,feats_im = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                            cells_per_block=self.cells_per_block, block_norm=self.block_norm, 
                            visualize=True,feature_vector=self.feature_vector)
        return feats_im

    def extract_image_features(self, image):
        feats = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                            cells_per_block=self.cells_per_block, block_norm=self.block_norm, 
                            visualize=False,feature_vector=self.feature_vector)
        return feats
    

if __name__ == "__main__":

    # Path to digits dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    digits_path = os.path.join(script_dir, "../example_fonts/digitsIms.pkl")

    # Load the pickle
    with open(digits_path, 'rb') as f:
        data = pickle.load(f)

    # Extract images and labels
    images = data['digitsIms']
    labels = data['digitsLabels']

    cell_sizes = [(4,4), (8,8), (16,16)]

    # Prepare a figure: rows = cell sizes, cols = digits
    fig, axes = plt.subplots(len(cell_sizes), 10, figsize=(15, 5))
    fig.suptitle("HOG visualizations for digits (padded + resized)\nDifferent cell sizes", fontsize=14)

    for i in range(10):  
        image = images[i]

        # Pad and resize
        padded = np.pad(image, pad_width=5, mode='constant', constant_values=0)
        resized = resize(padded, (padded.shape[0]*2, padded.shape[1]*2), anti_aliasing=True)

        # Compute and show HOG for each cell size
        for j, cell_size in enumerate(cell_sizes):
            extractor = FeatureHOG(pixels_per_cell=cell_size)
            hog_vis = extractor.extract_pixel_features(resized)

            ax = axes[j, i]
            ax.imshow(hog_vis, cmap='gray')
            ax.axis('off')

            if i == 0:
                ax.set_ylabel(f"cell={cell_size}", fontsize=10)
            if j == 0:
                ax.set_title(str(labels[i]))

    plt.tight_layout()
    plt.show()

    # Compute descriptors for all images without resizing
    # And analyze its dimensionality
    FeatureExtractor = FeatureHOG(pixels_per_cell=(4,4))

    for img in images:
        hog_feat = FeatureExtractor.extract_image_features(img)
        print("Dimensionality of the feature descriptor: ", len(hog_feat))