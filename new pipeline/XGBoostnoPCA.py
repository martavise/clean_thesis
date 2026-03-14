# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 18:37:57 2025

XGBoost for ENIGMA, with AutoML

This automated machine learning pipeline is as follows:
1. Repeated stratified cross-validation [5-folds, 3 times (20 times idealy)];
2. Subject level standardization;
3. PCA dimensionality reduction for any feature sets that has more than 100 features;
4. Hyperparameter tuning via FLAML AutoML/Randomized Search;
5. Output accuracy and ROC_AUC.

@author: Ruan

adapted pipeline without dimensional reduction
"""

import numpy as np
import pandas as pd
import pickle    #to store the results
import os
from statistics import mean, stdev
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from flaml import AutoML
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score

class SubZScoreTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Compute mean and standard deviation for each subject (row)
        means = np.mean(X, axis=1, keepdims=True)
        stds = np.std(X, axis=1, keepdims=True)
        stds[stds == 0] = 1
        return (X - means) / stds

def RepeatedStratifiedAutoML(X, y, grouplabels, n_repeats=3, k=5, estimator='xgboost'):
    Acc = []
    AUC = []
    
    # Create the repeated CV
    for rep in range(n_repeats):
        print(f"Iteration: {rep+1}/{n_repeats}")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42 + rep)
        
        for fold, (train_index, test_index) in enumerate(skf.split(X, grouplabels)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            zscorer = SubZScoreTransformer()
            X_train_z = zscorer.fit_transform(X_train)
            X_test_z = zscorer.transform(X_test)

                
            automl = AutoML()
            automl_settings = {
                "time_budget": 60,  # seconds; adjust as needed
                "metric": "accuracy",
                "task": "classification",
                "estimator_list": [estimator],
                "log_file_name": "flaml.log",
                "verbose": 1,
                "n_jobs": -1,  # use all available cores
            }
            
            # Fit FLAML’s AutoML on the training data
            automl.fit(X_train_z, y_train, **automl_settings)
            print("Best Hyperparmeter:", automl.best_config)
            # best_estimator = automl.model.estimator
            
            # Get predictions from the best model
            y_pred = automl.predict(X_test_z)
            y_prob = automl.predict_proba(X_test_z)[:, 1]
            
            # Compute metrics
            acc_fold = accuracy_score(y_test, y_pred)
            auc_fold = roc_auc_score(y_test, y_prob)
            
            print(f"  Fold {fold+1}/{k} | ACC: {acc_fold:.4f}, AUC: {auc_fold:.4f}")
            Acc.append(acc_fold)
            AUC.append(auc_fold)
    return Acc, AUC

def RepeatedStratifiedXGB(X, y, grouplabels, n_repeats=3, k=5, search=False):
    Acc = []
    AUC = []
    
    for rep in range(n_repeats):
        print(f"Iteration: {rep+1}/{n_repeats}")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42 + rep)
        
        for fold, (train_index, test_index) in enumerate(skf.split(X, grouplabels)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            grouplabels_train = grouplabels[train_index]

            # Subject-wise z-score (you may later remove this too for TDA)
            zscorer = SubZScoreTransformer()
            X_train_z = zscorer.fit_transform(X_train)
            X_test_z = zscorer.transform(X_test)

            xgb = XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                n_jobs=-1
            )

            if search:
                params = {
                    'min_child_weight': [1, 5, 10],
                    'gamma': [0.5, 1, 2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'max_depth': [3, 4, 5]
                }

                model = RandomizedSearchCV(
                    xgb,
                    param_distributions=params,
                    n_iter=5,
                    scoring='roc_auc',
                    n_jobs=4,
                    cv=skf.split(X_train_z, grouplabels_train),
                    verbose=1
                )
            else:
                model = xgb

            model.fit(X_train_z, y_train)

            y_pred = model.predict(X_test_z)
            y_prob = model.predict_proba(X_test_z)[:, 1]

            Acc.append(accuracy_score(y_test, y_pred))
            AUC.append(roc_auc_score(y_test, y_prob))

    return Acc, AUC

# %%

# Basic settings

MainDir = '/home/marta/Downloads'
TDAFeatures = np.load(os.path.join(MainDir, 'TDADes_400_precomputed_Sparse.npz'))
TDAFeatureNames = {'PL', 'PI', 'PS', 'BC'}
FCFeatures = np.load(os.path.join(MainDir, 'FCMetrics.npz'))
print(FCFeatures.files)  #to check the actual feature names
FCFeatureNames = {'FCVectors', 'fALFF', 'FCVectors_Net', 'fALFF_Net'} 
DemTable = pd.read_excel('/home/marta/Documents/Bachelor-Thesis/DemTable.xlsx')

# %%

# Use FlaML for automatic machine learning hyperparameter tuning
y = np.array(DemTable.Diag)
# Change the labeling into 0 and 1 for XGBoost
y = y - 1 

# Set Grouplables for stratified CV, ensuring equally split data from each center 
AllGroupLabels = np.array([f"{center}_{label}" for center, label in zip(np.array(DemTable.Center), y)])

SumBoard_Flaml = {}

########################### proviamo a togliere anche da qua i 7 soggetti sospetti ###############################################
# Excluded subjects (IDs)
Excluded = [1556, 1557, 1582, 1583, 1584, 1587, 1589]

GroupLabels = np.delete(AllGroupLabels, Excluded)
print(len(AllGroupLabels))
print(len(GroupLabels))
##################################################################################################################################



# Classification using FC features
for Name in FCFeatureNames:
    print(f"Classify using: {Name}")
    Feature_In = FCFeatures[Name]
    Acc, AUC = RepeatedStratifiedAutoML(Feature_In, y, GroupLabels)
    Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC),
    f'Accuracy_{Name}_std': stdev(Acc),
    f'ROC_AUC_{Name}_std': stdev(AUC)}

    SumBoard_Flaml.update(Summary)

# Classification using TDA descriptors
for Name in TDAFeatureNames:
    print(f"Classify using: {Name}")
    Feature_In = TDAFeatures[Name]
    for i in range(Feature_In.shape[1]):
        Feature_In_Dim = Feature_In[:, i, :].reshape(Feature_In.shape[0], -1)
        Acc, AUC = RepeatedStratifiedAutoML(Feature_In_Dim, y, GroupLabels)
        Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC),
        f'Accuracy_{Name}_std': stdev(Acc),
        f'ROC_AUC_{Name}_std': stdev(AUC)}

        SumBoard_Flaml.update(Summary)
        
for Name in TDAFeatureNames:
    print(f"Classify using: Concatenated {Name}")
    Feature_In = TDAFeatures[Name]
    Feature_In_C = Feature_In.reshape(Feature_In.shape[0], -1)
    print(Feature_In_C.shape)
    Acc, AUC = RepeatedStratifiedAutoML(Feature_In_C, y, GroupLabels)
    Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC),
    f'Accuracy_{Name}_std': stdev(Acc),
    f'ROC_AUC_{Name}_std': stdev(AUC)}

    SumBoard_Flaml.update(Summary)
    
FCVec = FCFeatures['FCVectors']    
for Name in TDAFeatureNames:
    print(f"Classify using: FC concatenated {Name}")
    Feature_In = TDAFeatures[Name]
    Feature_In_C = Feature_In.reshape(Feature_In.shape[0], -1)
    Feature_In_FC = np.concatenate((FCVec, Feature_In_C), axis=1)
    print(Feature_In_FC.shape)
    Acc, AUC = RepeatedStratifiedAutoML(Feature_In_FC, y, GroupLabels)
    Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC)}
    Summary = {    f'Accuracy_{Name}_std': stdev(Acc),    f'ROC_AUC_{Name}_std': stdev(AUC)}

    SumBoard_Flaml.update(Summary)


# saves the results locally
with open(os.path.join(MainDir, 'XGboostResultsFLAML.pkl'), 'wb') as f:
    pickle.dump(SumBoard_Flaml, f) 
    print("Saved FLAML results to 'XGboostResultsFLAML.pkl'")


# %%
    
# Use predefined parameter set for Randomized Search
SumBoard_Randomized = {}

# Classification using FC features
for Name in FCFeatureNames:
    print(f"Classify using: {Name}")
    Feature_In = FCFeatures[Name]
    Acc, AUC = RepeatedStratifiedXGB(Feature_In, y, GroupLabels, search=True)
    Summary = {
    f'Accuracy_{Name}_std': stdev(Acc),
    f'ROC_AUC_{Name}_std': stdev(AUC)}

    SumBoard_Randomized.update(Summary)

# Classification using TDA descriptors
for Name in TDAFeatureNames:
    print(f"Classify using: {Name}")
    Feature_In = TDAFeatures[Name]
    for i in range(Feature_In.shape[1]):
        Feature_In_Dim = Feature_In[:, i, :]
        Acc, AUC = RepeatedStratifiedXGB(Feature_In_Dim, y, GroupLabels, search=True)
        Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC),
        f'Accuracy_{Name}_std': stdev(Acc),
        f'ROC_AUC_{Name}_std': stdev(AUC)}
        SumBoard_Randomized.update(Summary)
        
for Name in TDAFeatureNames:
    print(f"Classify using: Concatenated {Name}")
    Feature_In = TDAFeatures[Name]
    Feature_In_C = Feature_In.reshape(Feature_In.shape[0], -1)
    print(Feature_In_C.shape)
    Acc, AUC = RepeatedStratifiedXGB(Feature_In_C, y, GroupLabels, search=True)
    Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC),
    f'Accuracy_{Name}_std': stdev(Acc),
    f'ROC_AUC_{Name}_std': stdev(AUC)}
    SumBoard_Randomized.update(Summary)
    
FCVec = FCFeatures['FCVectors']    
for Name in TDAFeatureNames:
    print(f"Classify using: FC concatenated {Name}")
    Feature_In = TDAFeatures[Name]
    Feature_In_C = Feature_In.reshape(Feature_In.shape[0], -1)
    Feature_In_FC = np.concatenate((FCVec, Feature_In_C), axis=1)
    print(Feature_In_FC.shape)
    Acc, AUC = RepeatedStratifiedXGB(Feature_In_FC, y, GroupLabels, search=True)
    Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC),
    f'Accuracy_{Name}_std': stdev(Acc),
    f'ROC_AUC_{Name}_std': stdev(AUC)}
    SumBoard_Randomized.update(Summary)


with open(os.path.join(MainDir, 'XGboostResults.pkl'), 'rb') as f:
    SumBoard_Randomized = pickle.load(f)  
  

with open(os.path.join(MainDir, 'XGboostResults.pkl'), 'wb') as f:
    pickle.dump(SumBoard_Randomized, f)  # Save Randomized Search results
    print("Saved Randomized Search results to 'XGboostResults.pkl'")

#%%
# ===============================
# voglio piangere non aveva funzionato, metto ua toppa
# ===============================

import numpy as np
import pandas as pd
import pickle    #to store the results
import os
from statistics import mean 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from flaml import AutoML
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score

# Paths
MainDir = '/home/marta/Downloads'

# Load features
FCFeatures = np.load(os.path.join(MainDir, 'FCMetrics.npz'))
TDAFeatures = np.load(os.path.join(MainDir, 'TDADes_400_precomputed_Sparse.npz'))

# Feature name sets
FCFeatureNames = {'FCVectors', 'fALFF', 'FCVectors_Net', 'fALFF_Net'}
TDAFeatureNames = {'PL', 'PI', 'PS', 'BC'}

# Load demographics
DemTable = pd.read_excel('/home/marta/Documents/Bachelor-Thesis/DemTable.xlsx')

# Labels
y = np.array(DemTable.Diag) - 1

# Group labels
AllGroupLabels = np.array(
    [f"{center}_{label}" for center, label in zip(DemTable.Center, y)]
)

# Exclude subjects
Excluded = [1556, 1557, 1582, 1583, 1584, 1587, 1589]
GroupLabels = np.delete(AllGroupLabels, Excluded)

print("Setup complete")
print("FCFeatures:", FCFeatures.files)
print("y shape:", y.shape)
print("GroupLabels shape:", GroupLabels.shape)
    
# %%

def run_randomized_xgb(
    FCFeatures,
    FCFeatureNames,
    TDAFeatures,
    TDAFeatureNames,
    y,
    GroupLabels,
    MainDir
):
    from statistics import mean
    import numpy as np
    import pickle
    import os

    SumBoard_Randomized = {}

    # Classification using FC features
    for Name in FCFeatureNames:
        print(f"Classify using: {Name}")
        Feature_In = FCFeatures[Name]
        Acc, AUC = RepeatedStratifiedXGB(
            Feature_In, y, GroupLabels, search=True
        )
        Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC),
        f'Accuracy_{Name}_std': stdev(Acc),
        f'ROC_AUC_{Name}_std': stdev(AUC)}

        SumBoard_Randomized.update(Summary)

    # Classification using TDA descriptors
    for Name in TDAFeatureNames:
        print(f"Classify using: {Name}")
        Feature_In = TDAFeatures[Name]
        for i in range(Feature_In.shape[1]):
            Feature_In_Dim = Feature_In[:, i, :].reshape(Feature_In.shape[0], -1)
            Acc, AUC = RepeatedStratifiedXGB(
                Feature_In_Dim, y, GroupLabels, search=True
            )
            Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC),
            f'Accuracy_{Name}_std': stdev(Acc),
            f'ROC_AUC_{Name}_std': stdev(AUC)}

            SumBoard_Randomized.update(Summary)

    # Concatenated TDA
    for Name in TDAFeatureNames:
        print(f"Classify using: Concatenated {Name}")
        Feature_In = TDAFeatures[Name]
        Feature_In_C = Feature_In.reshape(Feature_In.shape[0], -1)
        Acc, AUC = RepeatedStratifiedXGB(
            Feature_In_C, y, GroupLabels, search=True
        )
        Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC),
        f'Accuracy_{Name}_std': stdev(Acc),
        f'ROC_AUC_{Name}_std': stdev(AUC)}

        
        SumBoard_Randomized.update(Summary)

    # FC + TDA
    FCVec = FCFeatures['FCVectors']
    for Name in TDAFeatureNames:
        print(f"Classify using: FC concatenated {Name}")
        Feature_In = TDAFeatures[Name]
        Feature_In_C = Feature_In.reshape(Feature_In.shape[0], -1)
        Feature_In_FC = np.concatenate((FCVec, Feature_In_C), axis=1)
        Acc, AUC = RepeatedStratifiedXGB(
            Feature_In_FC, y, GroupLabels, search=True
        )
        Summary = {f'Accuracy_{Name}': mean(Acc), f'ROC_AUC_{Name}': mean(AUC),
        f'Accuracy_{Name}_std': stdev(Acc),
        f'ROC_AUC_{Name}_std': stdev(AUC)}
        SumBoard_Randomized.update(Summary)

    # Save results
    out_file = os.path.join(MainDir, 'XGboostResultsRandom.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(SumBoard_Randomized, f)

    print("Saved Randomized Search results to 'XGboostResultsRandom.pkl'")
    return SumBoard_Randomized

def RepeatedStratifiedXGB(X, y, grouplabels, n_repeats=3, k=5, pca_threshold=30, search=True):
    Acc = []
    AUC = []
    
    # Create the repeated CV
    for rep in range(n_repeats):
        print(f"Iteration: {rep+1}/{n_repeats}")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42 + rep)
        
        for fold, (train_index, test_index) in enumerate(skf.split(X, grouplabels)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            grouplabels_train = grouplabels[train_index]
            zscorer = SubZScoreTransformer()
            X_train_z = zscorer.fit_transform(X_train)
            X_test_z = zscorer.transform(X_test)
            if X_train_z.shape[1] > pca_threshold:
                pca = PCA(n_components=pca_threshold)
                X_train_z = pca.fit_transform(X_train_z)
                X_test_z = pca.transform(X_test_z)
            
            xgb = XGBClassifier(objective='binary:logistic')
            if search is True:
                params = {
                    'min_child_weight': [1, 5, 10],
                    'gamma': [0.5, 1, 1.5, 2, 5],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'max_depth': [3, 4, 5]
                    }
                model = RandomizedSearchCV(xgb, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train_z,grouplabels_train), verbose=2)
            else:
                model = xgb
            
            model.fit(X_train_z, y_train)
            y_pred = model.predict(X_test_z)
            y_prob = model.predict_proba(X_test_z)[:, 1]
            
            # Compute metrics
            acc_fold = accuracy_score(y_test, y_pred)
            auc_fold = roc_auc_score(y_test, y_prob)
            
            Acc.append(acc_fold)
            AUC.append(auc_fold)
    return Acc, AUC

class SubZScoreTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Compute mean and standard deviation for each subject (row)
        means = np.mean(X, axis=1, keepdims=True)
        stds = np.std(X, axis=1, keepdims=True)
        stds[stds == 0] = 1
        return (X - means) / stds



# %%
SumBoard_Randomized = run_randomized_xgb(
    FCFeatures,
    FCFeatureNames,
    TDAFeatures,
    TDAFeatureNames,
    y,
    GroupLabels,
    MainDir
)

