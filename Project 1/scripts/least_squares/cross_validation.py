# Cross validation utilities for Machine Learning course, project 1
#
# École polytechnique fédérale de Lausanne, Switzerland

import numpy as np

from proj1_helpers   import *
from preprocessing   import *
from implementations import *

# Hyper parameters for GD, SGD and least squares
MISSING_VALUES_THRESHOLDS = [0.99]
CORRELATION_THRESHOLDS    = [0.9]
OUTLIERS_THRESHOLDS       = [4,6]     
AUGMENTATION_DEGREES      = [1]

# Hyper parameters for LR and RLR
REG_LOG_REG_LAMBDA = [1]              
REG_LOG_REG_GAMMA    = [ 1e-7]   

# Function to chose by cross validation the best hyper parameters for our model
def train_least_squares(y, tx):
    p_max = 0 

    for mvt in MISSING_VALUES_THRESHOLDS:
        for ct in CORRELATION_THRESHOLDS:
            for ot in OUTLIERS_THRESHOLDS:
                for ad in AUGMENTATION_DEGREES:
                    # Here we have our data with some features dropped
                    p = cross_validation(y, tx, 5, mvt, ct, ot, ad)
                    print('Precission '+str(p)+' obtained for MVT='+str(mvt)+' CT='+str(ct)+' OT='+str(ot)+' AD='+str(ad)) 
 
                    # Are we improving our best precission?
                    if p > p_max:
                        best_mvt = mvt
                        best_ct  = ct
                        best_ad  = ad
                        best_ot  = ot
                        p_max    = p  

    return p_max, best_mvt, best_ct, best_ot, best_ad   


# Cross validation for concrete hyper parameters for our model
def cross_validation(y, tx, k, mvt=0.7, ct=0.85, ot=3, ad=2):
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
        high_MV_freq_features = get_high_MV_freq_features(y_train, tx_train, mvt)
        tx_train = np.delete(tx_train, high_MV_freq_features, axis=1)
        tx_test  = np.delete(tx_test,  high_MV_freq_features, axis=1)
        
        corr_features = get_correlated_features(y_train, tx_train, ct)
        tx_train = np.delete(tx_train, corr_features, axis=1)  
        tx_test  = np.delete(tx_test,  corr_features, axis=1)    
  
        replace_MV_by_average(y_train, tx_train)# Shouldn't we remove y argument from those functions not needing it?
        replace_MV_by_average(y_test,  tx_test) # Shouldn't we replace MV in test set with average in train set?

        y_train, tx_train = remove_outliers(y_train, tx_train,ot)

        tx_train = feature_augmentation(y_train, tx_train, ad)
        tx_test  = feature_augmentation(y_test,  tx_test,  ad)  
      
        tx_train = standarize(y_train, tx_train)
        tx_test  = standarize(y_test,  tx_test)
        
        tx_train = np.c_[np.ones(tx_train.shape[0]), tx_train]
        tx_test  = np.c_[np.ones(tx_test.shape[ 0]), tx_test ]
        
        # Model train
        w, loss     = least_squares(y_train, tx_train)

        # Model test
        y_test_pred = predict_labels(w, tx_test)
        accuracy    = len([a for (a,b) in zip(y_test,y_test_pred) if a == b])/tx_test.shape[0] # Sure there is a fancier way
        #print(accuracy)

        accuracies.append(accuracy)

    return sum(accuracies)/len(accuracies)





# Function to chose by cross validation the best hyper parameters for our model
def train_reg_logistic_regression(y, tx):
    p_max = 0 

    for mvt in MISSING_VALUES_THRESHOLDS:
        for ct in CORRELATION_THRESHOLDS:
            for ot in OUTLIERS_THRESHOLDS:
                for ad in AUGMENTATION_DEGREES:
                    for lamb in REG_LOG_REG_LAMBDA:
                        for gamma in REG_LOG_REG_GAMMA:
                            # Here we have our data with some features dropped
                            p = cross_validation_rlr(y, tx, 5, mvt, ct, ot, ad, lamb, gamma)
                            print('Precission '+str(p)+' obtained for MVT='+str(mvt)+' CT='+str(ct)+' OT='+str(ot)+' AD='+str(ad)+ ' LAMBDA='+str(lamb)+ ' GAMMA='+str(gamma)) 

                            # Are we improving our best precission?
                            if p > p_max:
                                best_mvt = mvt
                                best_ct  = ct
                                best_ad  = ad
                                p_max    = p  

    return p_max, best_mvt, best_ct, best_ad

# Cross validation for concrete hyper parameters for our model
def cross_validation_rlr(y, tx, k, mvt=0.7, ct=0.85, ot=3, ad=2, lamb=10, gamma=0.00001):
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
        high_MV_freq_features = get_high_MV_freq_features(y_train, tx_train, mvt)
        tx_train = np.delete(tx_train, high_MV_freq_features, axis=1)
        tx_test  = np.delete(tx_test,  high_MV_freq_features, axis=1)
        
        corr_features = get_correlated_features(y_train, tx_train, ct)
        tx_train = np.delete(tx_train, corr_features, axis=1)  
        tx_test  = np.delete(tx_test,  corr_features, axis=1)    
  
        replace_MV_by_average(y_train, tx_train)# Shouldn't we remove y argument from those functions not needing it?
        replace_MV_by_average(y_test,  tx_test) # Shouldn't we replace MV in test set with average in train set?

        y_train, tx_train = remove_outliers(y_train, tx_train,ot)

        tx_train = feature_augmentation(y_train, tx_train, ad)
        tx_test  = feature_augmentation(y_test,  tx_test,  ad)  
      
        tx_train = standarize(y_train, tx_train)
        tx_test  = standarize(y_test,  tx_test)
        
        tx_train = np.c_[np.ones(tx_train.shape[0]), tx_train]
        tx_test = np.c_[ np.ones(tx_test.shape[ 0]), tx_test ]

        
        # Model train
        w, loss     = reg_logistic_regression(y_train, tx_train, lamb, np.zeros(tx_train.shape[1]), 200, gamma)

        # Model test
        y_test_pred = predict_labels(w, tx_test)
        accuracy    = len([a for (a,b) in zip(y_test,y_test_pred) if a == b])/tx_test.shape[0] # Sure there is a fancier way
        #print(accuracy)

        accuracies.append(accuracy)

    return sum(accuracies)/len(accuracies)
             
