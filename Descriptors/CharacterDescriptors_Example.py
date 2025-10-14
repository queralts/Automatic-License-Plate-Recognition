# USAGE
# python train_simple.py --fonts input/example_fonts --char-classifier output/simple_char.cpickle \
#	--digit-classifier output/simple_digit.cpickle

##### PYTHON PACKAGES
# Generic
from imutils import paths
import argparse
import pickle
import cv2
import imutils
import numpy as np
import pandas
import os
from matplotlib import pyplot as plt

# Classifiers
# include different classifiers
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
from blockbinarypixelsum import FeatureBlockBinaryPixelSum
from intensity import FeatureIntensity
from lbp import FeatureLBP
from hog import FeatureHOG

#### EXP-SET UP
# DB Main Folder
script_dir = os.path.dirname(os.path.abspath(__file__))

DataDir = os.path.join(script_dir, "../example_fonts")
ResultsDir = os.path.join(script_dir, "Results")

# Load Font DataSets
fileout=os.path.join(DataDir,'alphabetIms')+'.pkl'    
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()  
alphabetIms=data['alphabetIms']
alphabetLabels=np.array(data['alphabetLabels'])
   

fileout=os.path.join(DataDir,'digitsIms')+'.pkl'   
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()   
digitsIms=data['digitsIms']
digitsLabels=np.array(data['digitsLabels'])

digitsFeat={}
alphabetFeat={}

# initialize descriptors
blockSizes =((5, 5),)#((5, 5), (5, 10), (10, 5), (10, 10))
target_size = (30, 15)  
descBlckAvg = FeatureBlockBinaryPixelSum()
descHOG = FeatureHOG()
descLBP_block = FeatureLBP(lbp_type='block_lbp')
descLBP_global = FeatureLBP(lbp_type='simple')

### EXTRACT FEATURES

for digit_roi in digitsIms:
    # Resize images
    digit_resized = cv2.resize(digit_roi, (target_size[1], target_size[0]))
    
    # Block Binary Pixel Sum
    block_features_digit = descBlckAvg.extract_image_features(digit_resized)
    digitsFeat.setdefault('BLCK_AVG_BLOCK', []).append(block_features_digit)
    
    # HOG
    hog_features_digit = descHOG.extract_image_features(digit_resized)
    digitsFeat.setdefault('HOG_BLOCK', []).append(hog_features_digit)
    
    # LBP
    # Block-wise aggregation
    lbp_block = descLBP_block.extract_image_features(digit_resized)
    digitsFeat.setdefault('LBP_BLOCK', []).append(lbp_block)
    # Global aggregation
    lbp_global = descLBP_global.extract_image_features(digit_resized)
    digitsFeat.setdefault('LBP_GLOBAL', []).append(lbp_global)

for alphabet_roi in alphabetIms:
    # Resize images
    alphabet_resized = cv2.resize(alphabet_roi, (target_size[1], target_size[0]))
    
    # Block Binary Pixel Sum
    block_features_alpha = descBlckAvg.extract_image_features(alphabet_resized)
    alphabetFeat.setdefault('BLCK_AVG_BLOCK', []).append(block_features_alpha)
    
    # HOG
    hog_features_alpha = descHOG.extract_image_features(alphabet_resized)
    alphabetFeat.setdefault('HOG_BLOCK', []).append(hog_features_alpha)
    
    # LBP
    # Block-wise aggregation
    lbp_block_alpha = descLBP_block.extract_image_features(alphabet_resized)
    alphabetFeat.setdefault('LBP_BLOCK', []).append(lbp_block_alpha)
    # Global aggregation
    lbp_global_alpha = descLBP_global.extract_image_features(alphabet_resized)
    alphabetFeat.setdefault('LBP_GLOBAL', []).append(lbp_global_alpha)

### VISUALIZE FEATURE SPACES
color=['r','m','g','cyan','y','k','orange','lime','b']
from sklearn.manifold import TSNE,trustworthiness
tsne = TSNE(n_components=2, random_state=42)

# ----------------- DIGIT FEATURE SPACES -----------------

for targetFeat in digitsFeat.keys():
    embeddings_2d = tsne.fit_transform(np.stack(digitsFeat[targetFeat]))
    
    plt.figure()
    plt.scatter(embeddings_2d[digitsLabels=='0', 0], embeddings_2d[digitsLabels=='0', 1], 
                marker='s')
    k=0
    for num in np.unique(digitsLabels)[1::]:
        plt.scatter(embeddings_2d[digitsLabels==num, 0], embeddings_2d[digitsLabels==num, 1], 
                     marker='o',color=color[k])
        k=k+1
    plt.legend(np.unique(digitsLabels), loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.title(targetFeat)

    # Build full output path
    save_path = os.path.join(ResultsDir, f"{targetFeat}DigitsFeatSpace.png")

    # Save and close the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ----------------- ALPHABET CHARACTERS FEATURE SPACES -----------------

for targetFeat in alphabetFeat.keys():
    embeddings_2d = tsne.fit_transform(np.stack(alphabetFeat[targetFeat]))
    
    plt.figure()
    
    unique_labels = np.unique(alphabetLabels)
    cmap = plt.colormaps['tab20'].resampled(len(unique_labels))  
    
    for k, char in enumerate(unique_labels):
        plt.scatter(
            embeddings_2d[alphabetLabels == char, 0],
            embeddings_2d[alphabetLabels == char, 1],
            marker='o',
            color=cmap(k),
            label=str(char)
        )
    
    plt.legend(np.unique(alphabetLabels), loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=1)
    plt.title(targetFeat)

    # Save plot
    save_path = os.path.join(ResultsDir, f"{targetFeat}AlphabetFeatSpace.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    
### VISUALIZE FEATURES IMAGES

## LBP Images for Digits


# HOG Images for Digits


