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
from scipy.stats import f_oneway

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

from scipy.stats import ttest_ind

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
NTrial = 20

aucMLP=[]
aucSVC=[]
aucKNN=[]

accMLP=[]
accSVC=[]
accKNN=[]

f1MLP=[]
f1SVC=[]
f1KNN=[]

averages = ['micro', 'macro', 'weighted', 'class0', 'class1', 'mean']
recMLP={a: [] for a in averages}   
recSVC={a: [] for a in averages}    
recKNN={a: [] for a in averages}

precMLP={a: [] for a in averages}   
precSVC={a: [] for a in averages}
precKNN={a: [] for a in averages}

avgs = ['micro', 'macro', 'weighted']

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
    recSVC['class0'].append(rec[0])
    recSVC['class1'].append(rec[1])
    recSVC['mean'].append(np.mean(rec))

    precSVC['class0'].append(prec[0])
    precSVC['class1'].append(prec[1])
    precSVC['mean'].append(np.mean(prec))

    acc = accuracy_score(y_test, y_pred)
    accSVC.append(acc)

    for avg_method in avgs:
        prec = precision_score(y_test, y_pred, average=avg_method)
        rec = recall_score(y_test, y_pred, average=avg_method)

        recSVC[avg_method].append(rec)
        precSVC[avg_method].append(prec)

    # Weighted F1 for SVC
    f1SVC.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    # --- inside your trial loop, at the end of the KNN block, ensure lists are appended (already correct), then add: ---
    f1KNN.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    # --- inside your trial loop, at the end of the MLP block, ensure lists are appended (already correct), then add: ---
    f1MLP.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

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
    recKNN['class0'].append(rec[0])
    recKNN['class1'].append(rec[1])
    recKNN['mean'].append(np.mean(rec))

    precKNN['class0'].append(prec[0])
    precKNN['class1'].append(prec[1])
    precKNN['mean'].append(np.mean(prec))

    acc = accuracy_score(y_test, y_pred)
    accKNN.append(acc)
    
    for avg_method in avgs:
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
    recMLP['class0'].append(rec[0])
    recMLP['class1'].append(rec[1])
    recMLP['mean'].append(np.mean(rec))

    precMLP['class0'].append(prec[0])
    precMLP['class1'].append(prec[1])
    precMLP['mean'].append(np.mean(prec))

    acc = accuracy_score(y_test, y_pred)
    accMLP.append(acc)
    
    for avg_method in avgs:
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

# Create an array for trial numbers
trials = np.arange(1, NTrial+1)

# Plot Accuracy across trials
plt.figure(figsize=(8,4))
plt.plot(trials, accSVC, marker='o', c='b', markersize=8, label='SVC')
plt.plot(trials, accKNN, marker='o', c='r', markersize=8, label='KNN')
plt.plot(trials, accMLP, marker='o', c='g', markersize=8, label='MLP')
plt.xlabel('Trial', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy across trials')
plt.xticks(trials)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Precision (mean across classes) across trials
plt.figure(figsize=(8,4))
plt.plot(trials, precSVC['mean'], marker='o', c='b', markersize=8, label='SVC')
plt.plot(trials, precKNN['mean'], marker='o', c='r', markersize=8, label='KNN')
plt.plot(trials, precMLP['mean'], marker='o', c='g', markersize=8, label='MLP')
plt.xlabel('Trial', fontsize=12)
plt.ylabel('Precision (mean)', fontsize=12)
plt.title('Precision across trials')
plt.xticks(trials)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Recall (mean across classes) across trials
plt.figure(figsize=(8,4))
plt.plot(trials, recSVC['mean'], marker='o', c='b', markersize=8, label='SVC')
plt.plot(trials, recKNN['mean'], marker='o', c='r', markersize=8, label='KNN')
plt.plot(trials, recMLP['mean'], marker='o', c='g', markersize=8, label='MLP')
plt.xlabel('Trial', fontsize=12)
plt.ylabel('Recall (mean)', fontsize=12)
plt.title('Recall across trials')
plt.xticks(trials)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# METRIC ANALYSIS:

# Store all metrics for comparison
results = []

# Combine metrics from each model
models = {
    'SVC': (precSVC, recSVC, accSVC),
    'KNN': (precKNN, recKNN, accKNN),
    'MLP': (precMLP, recMLP, accMLP)
}

for model_name, (prec_dict, rec_dict, acc_list) in models.items():
    for avg_method in avgs:
        results.append({
            'Model': model_name,
            'Average': avg_method,
            'Precision': prec_dict[avg_method][0],
            'Recall': rec_dict[avg_method][0],
            'Accuracy': acc_list[0]
        })

# Convert to DataFrame for display
results_df = pandas.DataFrame(results)
# Compare results for the first run
print("\n COMPARISON OF PRECISION, RECALL, AND ACCURACY: \n")
print(results_df)

# Compare average scores to precision and recall scores per class:
comparison = []
score = ['class0', 'class1', 'mean']

for model_name, (prec_dict, rec_dict, acc_list) in models.items():
    for trial_idx in range(NTrial):
        for metric in score:
            comparison.append({
                'Model': model_name,
                'Trial': trial_idx + 1,
                'Method': metric,
                'Precision': prec_dict[metric][trial_idx],
                'Recall': rec_dict[metric][trial_idx],
            })

# Convert to DataFrame for display
comparison_df = pandas.DataFrame(comparison)
comparison_df = comparison_df.sort_values(by=['Trial', 'Model', 'Method']).reset_index(drop=True)
# Show results
print("\n COMPARISON OF PRECISION AND RECALL ACROSS CLASSES \n")
print(comparison_df)

#------------- COMPUTE STUDENT'S T-TEST FOR ALL PAIRS ---------------
# Compute differences between classifiers
diff_SVC_KNN = np.array(aucSVC) - np.array(aucKNN)
diff_SVC_MLP = np.array(aucSVC) - np.array(aucMLP)
diff_KNN_MLP = np.array(aucKNN) - np.array(aucMLP)

# Perform one-sample t-tests for each pairing
TStudent_SVC_KNN, p_SVC_KNN = scipy.stats.ttest_1samp(diff_SVC_KNN, 0.0)
TStudent_SVC_MLP, p_SVC_MLP = scipy.stats.ttest_1samp(diff_SVC_MLP, 0.0)
TStudent_KNN_MLP, p_KNN_MLP = scipy.stats.ttest_1samp(diff_KNN_MLP, 0.0)

# Display results
print("\nT-TEST RESULTS (TStudent, pval) for mean difference = 0:")
print(f"SVC - KNN:  TStudent = {TStudent_SVC_KNN:.4f},  pval = {p_SVC_KNN:.4e}")
print(f"SVC - MLP:  TStudent = {TStudent_SVC_MLP:.4f},  pval = {p_SVC_MLP:.4e}")
print(f"KNN - MLP:  TStudent = {TStudent_KNN_MLP:.4f},  pval = {p_KNN_MLP:.4e}")

# Then we want to determine significance (p < 0.01)
pairs = {
    "SVC vs KNN": p_SVC_KNN,
    "SVC vs MLP": p_SVC_MLP,
    "KNN vs MLP": p_KNN_MLP
}

print("\nSIGNIFICANTLY DIFFERENT CLASSIFIERS (p < 0.01):")
for pair, pval in pairs.items():
    if pval < 0.01:
        print(f"  {pair} are significant difference (p={pval:.4e})")
    else:
        print(f"  {pair} are NOT significantly different (p={pval:.4e})")


#F_stat, p_ANOVA = f_oneway(aucSVC, aucKNN, aucMLP)
#print("\nANOVA TEST (AUC across classifiers):")
#print(f"F-statistic = {F_stat:.4f}, p-value = {p_ANOVA:.4e}")
#--------------- VISUALIZE CLASSIFIERS PERFORMANCE ------------------
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

# ---------- COMPUTE METRICS FOR ALL CLASSIFIERS -----------------------

def summarize_metric(metric_lists, metric_name):
    rows = []
    for model, vals in metric_lists.items():
        s = pandas.Series(vals)
        desc = {
            'Model': model,
            'Metric': metric_name,
            'Mean': s.mean(),
            'Median': s.median(),
            'Std': s.std(ddof=1),
            'Q25': s.quantile(0.25),
            'Q50': s.quantile(0.50),
            'Q75': s.quantile(0.75),
            'Min': s.min(),
            'Max': s.max()
        }
        rows.append(desc)
    return pandas.DataFrame(rows)

auc_stats = summarize_metric({'SVC': aucSVC, 'KNN': aucKNN, 'MLP': aucMLP}, 'AUC')
acc_stats = summarize_metric({'SVC': accSVC, 'KNN': accKNN, 'MLP': accMLP}, 'Accuracy')
f1_stats  = summarize_metric({'SVC': f1SVC,  'KNN': f1KNN,  'MLP': f1MLP},  'F1_weighted')

all_stats = pandas.concat([auc_stats, acc_stats, f1_stats], ignore_index=True)

print("\nDESCRIPTIVE STATISTICS ACROSS NTrials (AUC, Accuracy, F1_weighted):\n")
print(all_stats.sort_values(['Metric','Model']).to_string(index=False))

overall = (
    all_stats.pivot_table(index='Model', columns='Metric', values='Mean')
    .assign(Overall_Mean=lambda df: df.mean(axis=1))
    .reset_index()
    .sort_values('Overall_Mean', ascending=False)
)
print("\nOVERALL ranking (AUC, Accuracy, F1_weighted):\n")
print(overall.to_string(index=False))


# ------------ COMPUTE METRIC RANGES  ------------

def print_metric_ranges(all_stats):

    df = all_stats.loc[:, ['Model', 'Metric', 'Mean', 'Std']].copy()

    metric_label = {'AUC': 'AUC', 'Accuracy': 'Accuracy', 'F1_weighted': 'F1'}

    model_order = ['KNN', 'MLP', 'SVC']
    metric_order = ['AUC', 'Accuracy', 'F1_weighted']

    df['Mean'] = df['Mean'].round(3)
    df['Std']  = df['Std'].round(3)

    for model in model_order:
        row = df[df['Model'] == model].set_index('Metric').reindex(metric_order)
        parts = []
        for met in metric_order:
            if met in row.index and not row.loc[met, ['Mean','Std']].isna().any():
                parts.append(f"{metric_label[met]} = {row.loc[met, 'Mean']:.3f} ± {row.loc[met, 'Std']:.3f}")
        print(f"\n{model}: " + ", ".join(parts))

print("\nMETRIC RANGES (mean ± std):")
print_metric_ranges(all_stats)
"""