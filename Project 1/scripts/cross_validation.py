# École polytechnique fédérale de Lausanne, Switzerland
# CS-433 Machine Learning, project 1
# 
# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
# David Alonso del Barrio            ~ david.alonsodelbarrio@epfl.ch
# Andrés Montero Ranc                ~ andres.monteroranc@epfl.ch
#
# Functions for training and validating least squares, ridge regression,
# logistic regression and regularized logistic regression models.

import numpy as np

from proj1_helpers   import *
from preprocessing   import *
from implementations import *

# Hyper parameters for GD, SGD and least squares
MISSING_VALUES_THRESHOLDS = [0.99]
CORRELATION_THRESHOLDS    = [0.95]
OUTLIERS_THRESHOLDS       = [10]     
AUGMENTATION_DEGREES      = [2]

# Hyper parameters for LR and RLR
LAMBDAS = [1]              
GAMMAS  = [1e-7]   

# Least squares grid search for hyper parameters with cross validation
def train_least_squares(y, tx):
    p_max = -1 

    for mvt in MISSING_VALUES_THRESHOLDS:
        for ct in CORRELATION_THRESHOLDS:
            for ot in OUTLIERS_THRESHOLDS:
                for ad in AUGMENTATION_DEGREES:
                    p = cross_validation_LS(y, tx, 10, mvt, ct, ot, ad)
                    print('Precission '+str(p)+' obtained for MVT='+str(mvt)+' CT='+str(ct)+' OT='+str(ot)+' AD='+str(ad)) 
 
                    if p > p_max:
                        p_max,best_mvt,best_ct,best_ot,best_ad = p,mvt,ct,ot,ad

    return p_max, best_mvt, best_ct, best_ot, best_ad  

# Ridge regression grid search for hyper parameters with cross validation
def train_ridge_regression(y, tx):
    p_max = -1 

    for mvt in MISSING_VALUES_THRESHOLDS:
        for ct in CORRELATION_THRESHOLDS:
            for ot in OUTLIERS_THRESHOLDS:
                for ad in AUGMENTATION_DEGREES:
                    for lamb in LAMBDAS: 
                        p = cross_validation_RR(y, tx, 10, mvt, ct, ot, ad,lamb)
                        print('Precission '+str(p)+' obtained for MVT='+str(mvt)+' CT='+str(ct)+' OT='+str(ot)+' AD='+str(ad)+ ' LAMB='+str(lamb)) 
 
                        if p > p_max:
                            p_max,best_mvt,best_ct,best_ot,best_ad,best_lamb = p,mvt,ct,ot,ad,lamb 

    return p_max, best_mvt, best_ct, best_ot, best_ad, best_lamb  

# Logistic regression grid search for hyper parameters with cross validation
def train_logistic_regression(y, tx):
    p_max = -1 

    for mvt in MISSING_VALUES_THRESHOLDS:
        for ct in CORRELATION_THRESHOLDS:
            for ot in OUTLIERS_THRESHOLDS:
                for ad in AUGMENTATION_DEGREES:
                    for gamma in GAMMAS:
                        p = cross_validation_LR(y, tx, 5, mvt, ct, ot, ad, gamma)
                        print('Precission '+str(p)+' obtained for MVT='+str(mvt)+' CT='+str(ct)+' OT='+str(ot)+' AD='+str(ad)+ ' GAMMA='+str(gamma)) 

                        if p > p_max:
                            p_max,best_mvt,best_ct,best_ot,best_ad,best_gamma = p,mvt,ct,ot,ad,gamma  

    return p_max, best_mvt, best_ct, best_ot, best_ad, best_gamma

# Regularized logistic regression grid search for hyper parameters with cross validation
def train_reg_logistic_regression(y, tx):
    p_max = -1 

    for mvt in MISSING_VALUES_THRESHOLDS:
        for ct in CORRELATION_THRESHOLDS:
            for ot in OUTLIERS_THRESHOLDS:
                for ad in AUGMENTATION_DEGREES:
                    for lamb in LAMBDAS:
                        for gamma in GAMMAS:
                            p = cross_validation_RLR(y, tx, 5, mvt, ct, ot, ad, lamb, gamma)
                            print('Precission '+str(p)+' obtained for MVT='+str(mvt)+' CT='+str(ct)+' OT='+str(ot)+' AD='+str(ad)+' LAMBDA='+str(lamb)+' GAMMA='+str(gamma)) 

                            if p > p_max:
                                p_max,best_mvt,best_ct,best_ot,best_ad,best_lambda,best_gamma = p,mvt,ct,ot,ad,lamb,gamma 

    return p_max, best_mvt, best_ct, best_ot, best_ad, best_lambda, best_gamma


# Leas squares k-fold corss validation of a model
def cross_validation_LS(y, tx, k, mvt=0.7, ct=0.85, ot=3, ad=2):
    # Data random distribution in k sets
    idxs = np.arange(tx.shape[0])
    np.random.shuffle(idxs)
    idxs_folds = np.array_split(idxs, k)
    accuracies = []

    for i in range(k):
        # Data splitting in test and train set, it's important to copy them
        tx_test  = tx[idxs_folds[i]].copy()
        y_test   = y[ idxs_folds[i]].copy()
        tx_train = tx[[j for j in idxs if j not in idxs_folds[i]]].copy()
        y_train  = y[[ j for j in idxs if j not in idxs_folds[i]]].copy()

        # Data preprocessing
        y_train, tx_train, tx_test = preprocess_data(y_train, tx_train, tx_test, mvt, ct, ot, ad)
        
        # Model train
        w, loss     = least_squares(y_train, tx_train)

        # Model test
        y_test_pred = predict_labels(w, tx_test)
        accuracy    = len([a for (a,b) in zip(y_test,y_test_pred) if a == b])/tx_test.shape[0]

        accuracies.append(accuracy)

    return sum(accuracies)/len(accuracies)


# Ridge regression k-fold cross validation of a model
def cross_validation_RR(y, tx, k, mvt=0.7, ct=0.85, ot=3, ad=2, lamb=1):
    # Data random distribution in k sets
    idxs = np.arange(tx.shape[0])
    np.random.shuffle(idxs)
    idxs_folds = np.split(idxs, k)
    accuracies = []

    for i in range(k):
        # Data splitting in test and train set, it's important to copy them
        tx_test  = tx[idxs_folds[i]].copy()
        y_test   = y[ idxs_folds[i]].copy()
        tx_train = tx[[j for j in idxs if j not in idxs_folds[i]]].copy()
        y_train  = y[[ j for j in idxs if j not in idxs_folds[i]]].copy()

        # Data preprocessing
        y_train, tx_train, tx_test = preprocess_data(y_train, tx_train, tx_test, mvt, ct, ot, ad)

        # Model train
        w, loss     = ridge_regression(y_train, tx_train, lamb)

        # Model test
        y_test_pred = predict_labels(w, tx_test)
        accuracy    = len([a for (a,b) in zip(y_test,y_test_pred) if a == b])/tx_test.shape[0] 

        accuracies.append(accuracy)

    return sum(accuracies)/len(accuracies)

# Logistic regression k-fold corss validation of a model
def cross_validation_LR(y, tx, k, mvt=0.7, ct=0.85, ot=3, ad=2, gamma=0.00001):
    # Data random distribution in k sets
    idxs = np.arange(tx.shape[0])
    np.random.shuffle(idxs)
    idxs_folds = np.split(idxs, k)
    accuracies = []

    for i in range(k):
        # Data splitting in test and train set, it's important to copy them
        tx_test  = tx[idxs_folds[i]].copy()
        y_test   = y[ idxs_folds[i]].copy()
        tx_train = tx[[j for j in idxs if j not in idxs_folds[i]]].copy()
        y_train  = y[[ j for j in idxs if j not in idxs_folds[i]]].copy()

        # Data preprocessing
        y_train, tx_train, tx_test = preprocess_data(y_train, tx_train, tx_test, mvt, ct, ot, ad)
        
        # Model train
        w, loss     = logistic_regression(y_train, tx_train, np.ones(tx_train.shape[1]), 100, gamma)

        # Model test
        y_test_pred = predict_labels(w, tx_test)
        accuracy    = len([a for (a,b) in zip(y_test,y_test_pred) if a == b])/tx_test.shape[0] 

        accuracies.append(accuracy)

    return sum(accuracies)/len(accuracies)


# Regularized logistic regression k-fold corss validation of a model
def cross_validation_RLR(y, tx, k, mvt=0.7, ct=0.85, ot=3, ad=2, lamb=10, gamma=0.00001):
    # Data random distribution in k sets
    idxs = np.arange(tx.shape[0])
    np.random.shuffle(idxs)
    idxs_folds = np.array_split(idxs, k)
    accuracies = []

    for i in range(k):
        # Data splitting in test and train set, it's important to copy them
        tx_test  = tx[idxs_folds[i]].copy()
        y_test   = y[ idxs_folds[i]].copy()
        tx_train = tx[[j for j in idxs if j not in idxs_folds[i]]].copy()
        y_train  = y[[ j for j in idxs if j not in idxs_folds[i]]].copy()

        # Data preprocessing
        y_train, tx_train, tx_test = preprocess_data(y_train, tx_train, tx_test, mvt, ct, ot, ad)
        
        # Model train
        w, loss     = reg_logistic_regression(y_train, tx_train, lamb, np.zeros(tx_train.shape[1]), 200, gamma)

        # Model test
        y_test_pred = predict_labels(w, tx_test)
        accuracy    = len([a for (a,b) in zip(y_test,y_test_pred) if a == b])/tx_test.shape[0] 

        accuracies.append(accuracy)

    return sum(accuracies)/len(accuracies)
             
