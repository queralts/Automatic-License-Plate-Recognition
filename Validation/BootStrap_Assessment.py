# USAGE
# python train_simple.py --fonts input/example_fonts --char-classifier output/simple_char.cpickle \
#	--digit-classifier output/simple_digit.cpickle

##### PYTHON PACKAGES
# Generic
import pickle
import cv2
import imutils
import numpy as np
import pandas
import os
from matplotlib import pyplot as plt
import scipy

# Classifiers
# include differnet classifiers
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support,precision_score, recall_score, f1_score, accuracy_score, classification_report

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)


#### STEP0. EXP-SET UP

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
ResultsDir = os.path.join(script_dir, "../datasets/validation_dataset")

# Load Font DataSets
fileout=os.path.join(ResultsDir,'AlphabetDescriptors')+'.pkl'    
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()  
alphabetFeat=data['alphabetFeat']
alphabetLabels=data['alphabetLabels']
   

fileout=os.path.join(ResultsDir,'DigitsDescriptors')+'.pkl'   
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()   
digitsFeat=data['digitsFeat']
digitsLabels=data['digitsLabels']




#### DEFINE BINARY DATASET
DescriptorsTags=list(digitsFeat.keys())
targetFeat=DescriptorsTags[0]

digits=np.stack(digitsFeat[targetFeat])
digitsLab=np.zeros(digits.shape[0])
chars=np.stack(alphabetFeat[targetFeat])
charsLab=np.ones(chars.shape[0])

X=np.concatenate((digits,chars))
y=np.concatenate((digitsLab,charsLab))

### STEP1. TRAIN BINARY CLASSIFIERS [CHARACTER VS DIGITS]

# Initialize variables
#NTrial=1
NTrial = 30

aucMLP=[]
aucSVC=[]
aucKNN=[]

accMLP=[]
accSVC=[]
accKNN=[]

averages = ['micro', 'macro', 'weighted']
recMLP={a: [] for a in averages}   
recSVC={a: [] for a in averages}    
recKNN={a: [] for a in averages}

precMLP={a: [] for a in averages}   
precSVC={a: [] for a in averages}
precKNN={a: [] for a in averages}

for kTrial in np.arange(NTrial):
    # Random Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)
    
    ##### SVM
    ## Train Model
    ModelSVC = SVC(C=1.0,class_weight='balanced') #compute loss with weights accounting for class unbalancing
    # Use CalibratedClassifierCV to calibrate probabilites
    ModelSVC = CalibratedClassifierCV(ModelSVC,n_jobs=-1)
    ModelSVC.fit(X_train, y_train)
    ## Evaluate Model
    pSVC = ModelSVC.predict_proba(X_test)
    
    ## Metrics
    auc=roc_auc_score(y_test, pSVC[:,1])
    aucSVC.append(auc)
    y_pred=(pSVC[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                       zero_division=0)

    # Compute and Store metric information for SVC
    acc = accuracy_score(y_test, y_pred)
    accSVC.append(acc)

    for avg_method in averages:
        prec = precision_score(y_test, y_pred, average=avg_method)
        rec = recall_score(y_test, y_pred, average=avg_method)

        recSVC[avg_method] = rec
        precSVC[avg_method] = prec


    ##### KNN
    ## Train Model
    ModelKNN = KNeighborsClassifier(n_neighbors=10)
    ModelKNN = CalibratedClassifierCV(ModelKNN,n_jobs=-1)
    ModelKNN.fit(X_train, y_train)
    ## Evaluate Model
    pKNN = ModelKNN.predict_proba(X_test)
    # Metrics
    auc=roc_auc_score(y_test, pKNN[:,1])
    aucKNN.append(auc)
    y_pred=(pKNN[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                       zero_division=0)
    
    # Compute and Store metric information for KNN
    acc = accuracy_score(y_test, y_pred)
    accKNN.append(acc)
    
    for avg_method in averages:
        prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        recKNN[avg_method].append(rec)
        precKNN[avg_method].append(prec)

    #### MLP
    ## Train Model
    ModelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, 20), random_state=1,max_iter=100000)
    ModelMLP = CalibratedClassifierCV(ModelMLP,n_jobs=-1)
    ModelMLP.fit(X_train, y_train)
    ## Evaluate Model
    pMLP = ModelMLP.predict_proba(X_test)
    # Metrics
    auc=roc_auc_score(y_test, pMLP[:,1])
    aucMLP.append(auc)
    y_pred=(pMLP[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                       zero_division=0)

    # Compute and Store metric information for MLP
    acc = accuracy_score(y_test, y_pred)
    accMLP.append(acc)
    
    for avg_method in averages:
        prec = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        recMLP[avg_method].append(rec)
        precMLP[avg_method].append(prec)

#### STEP2. ANALYZE RESULTS
## Visual Exploration
#recSVC=np.stack(recSVC)
#recKNN=np.stack(recKNN)
#recMLP=np.stack(recMLP)
# #### Plots accoss trials (random splits)
plt.figure()
plt.plot(np.arange(NTrial),aucSVC,marker='o',c='b',markersize=10)
plt.plot(np.arange(NTrial),aucKNN,marker='o',c='r',markersize=10)
plt.plot(np.arange(NTrial),aucMLP,marker='o',c='g',markersize=10)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(NTrial), fontsize=10)
plt.xlabel("Trial", fontsize=15)
plt.ylabel("AUC", fontsize=15)
plt.show()

# Store all metrics for comparison
results = []

# Combine metrics from each model
models = {
    'SVC': (precSVC, recSVC, accSVC),
    'KNN': (precKNN, recKNN, accKNN),
    'MLP': (precMLP, recMLP, accMLP)
}

for model_name, (prec_dict, rec_dict, acc_list) in models.items():
    for avg_method in averages:
        results.append({
            'Model': model_name,
            'Average': avg_method,
            'Precision': prec_dict[avg_method],
            'Recall': rec_dict[avg_method],
            'Accuracy': np.mean(acc_list)  
        })

# Convert to DataFrame for display
results_df = pandas.DataFrame(results)

print("\n COMPARISON OF PRECISION, RECALL, AND ACCURACY: \n")
print(results_df)
"""
# Boxplots for AUC and Accuracy
plt.figure(figsize=(8,4))
plt.boxplot([aucSVC, aucKNN, aucMLP], labels=['SVM','KNN','MLP'])
plt.ylabel('AUC'); plt.title('AUC across trials (boxplot)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.boxplot([accSVC, accKNN, accMLP], labels=['SVM','KNN','MLP'])
plt.ylabel('Accuracy'); plt.title('Accuracy across trials (boxplot)')
plt.tight_layout(); plt.show()

# Barplots of mean ± std for AUC and Accuracy
def bar_mean_std(values_list, labels, title, ylabel):
    means = [np.mean(v) for v in values_list]
    stds  = [np.std(v)  for v in values_list]
    x = np.arange(len(values_list))
    plt.figure(figsize=(8,4))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, labels)
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.show()

bar_mean_std([aucSVC, aucKNN, aucMLP], ['SVM','KNN','MLP'], 'AUC: mean ± std', 'AUC')
bar_mean_std([accSVC, accKNN, accMLP], ['SVM','KNN','MLP'], 'Accuracy: mean ± std', 'Accuracy')

# Histograms of AUC per model
for data, name in zip([aucSVC, aucKNN, aucMLP], ['SVM','KNN','MLP']):
    plt.figure(figsize=(8,4))
    plt.hist(data, bins=10, alpha=0.85)
    plt.xlabel('AUC'); plt.ylabel('Count'); plt.title(f'AUC distribution: {name}')
    plt.tight_layout(); plt.show()
"""