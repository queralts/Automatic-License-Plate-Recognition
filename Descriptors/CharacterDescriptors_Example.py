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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)
from blockbinarypixelsum import FeatureBlockBinaryPixelSum
from intensity import FeatureIntensity
from lbp import FeatureLBP
from hog import FeatureHOG

#### EXP-SET UP
# DB Main Folder
script_dir = os.path.dirname(os.path.abspath(__file__))

DataDir = os.path.join(script_dir, "../datasets/example_fonts")
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
    digitsFeat.setdefault('BLCK_AVG', []).append(block_features_digit)
    
    # HOG
    hog_features_digit = descHOG.extract_image_features(digit_resized)
    digitsFeat.setdefault('HOG', []).append(hog_features_digit)
    
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
    alphabetFeat.setdefault('BLCK_AVG', []).append(block_features_alpha)
    
    # HOG
    hog_features_alpha = descHOG.extract_image_features(alphabet_resized)
    alphabetFeat.setdefault('HOG', []).append(hog_features_alpha)
    
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
# ---- QUICK PARAM SWEEP+ ----
# Tests a few HOG and LBP settings on digits only

def resize(im, target_hw=(30, 15)):
    h, w = target_hw
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

def safe_perplexity(n):
    # t-SNE constraint: n > 3*perplexity + 1
    return min(30, max(2, (n - 2) // 3))

def tsne_plot(X, y, title, out_png):
    from sklearn.manifold import TSNE
    perp = safe_perplexity(len(X))
    emb = TSNE(
        n_components=2, random_state=42, init="pca",
        learning_rate="auto", perplexity=perp
    ).fit_transform(X)
    plt.figure()
    classes = np.unique(y)
    cmap = plt.colormaps["tab20"].resampled(max(2, len(classes)))
    for i, c in enumerate(classes):
        m = (y == c)
        plt.scatter(emb[m, 0], emb[m, 1], label=str(c), color=cmap(i), s=12)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    plt.title(f"{title} (perp={perp})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def quick_score(X, y, tag):
    # ultra-simple split + LinearSVC 
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = LinearSVC(dual="auto", random_state=42, max_iter=5000)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    print(f"{tag}: accuracy={acc:.3f}")

def quick_param_sweep_digits():
    os.makedirs(ResultsDir, exist_ok=True)

    hog_presets = [
        {"orientations": 6},
        {"orientations": 9},
        {"orientations": 12},
    ]
    lbp_presets = [
        {"radius": 1},
        {"radius": 2},
        {"radius": 3},
    ]

    # HOG sweeps
    for i, kwargs in enumerate(hog_presets, 1):
        try:
            desc = FeatureHOG(**kwargs)
            X = [desc.extract_image_features(resize(im)) for im in digitsIms]
            X = np.asarray(X, dtype=np.float32)
            tag = f"HOG{kwargs}"
            tsne_plot(X, digitsLabels, f"Digits – {tag}", os.path.join(ResultsDir, f"Digits_{tag}.png"))
            quick_score(X, digitsLabels, tag)
        except TypeError as e:
            print(f"Skipping HOG preset {kwargs}: {e}")

    # LBP (block) sweeps
    for i, kwargs in enumerate(lbp_presets, 1):
        try:
            desc = FeatureLBP(lbp_type="block_lbp", **kwargs)
            X = [desc.extract_image_features(resize(im)) for im in digitsIms]
            X = np.asarray(X, dtype=np.float32)
            tag = f"LBP_BLOCK{kwargs}"
            tsne_plot(X, digitsLabels, f"Digits – {tag}", os.path.join(ResultsDir, f"Digits_{tag}.png"))
            quick_score(X, digitsLabels, tag)
        except TypeError as e:
            print(f"Skipping LBP preset {kwargs}: {e}")

quick_param_sweep_digits()


## LBP Images for Digits


# HOG Images for Digits


