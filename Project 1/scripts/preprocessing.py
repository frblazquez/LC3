# Data preprocessing utilities for Machine Learning course, project 1
#
# École polytechnique fédérale de Lausanne, Switzerland

# TODO: Handle PRI_jet_num categorical variable!
# TODO: Check there is no NaNs
# TODO: Implement all the functions
# TODO: Remove low variance features ? 

import numpy as np

# Removes features with high Pearson correlation coefficient
def remove_highly_correlated_features(y, tx, threshold):
    # First we compute the correlation coefficient of every pair
    corr_coef = np.corrcoef(tx, y, False)
    # Then we remove one feature of every pair of those with corr. coef. 
    # greater than the threshold provided
    highly_correlated_features = []
    for i in range(corr_coef.shape[1]-1):
        if i in highly_correlated_features:
            continue
        for j in range(i+1, corr_coef.shape[1]):
            if j in highly_correlated_features:
                continue
            if abs(corr_coef[i][j]) > threshold:
                # print('Removing '+str(j)+' for its correalation with '+str(i))
                highly_correlated_features.append(j)
    return np.delete(tx, highly_correlated_features, 1)

# Removes features with missing values frequency greater than threshold
def remove_features_with_MV(y, tx, MV, threshold):
    return np.delete(tx,np.where(np.count_nonzero(tx == MV, axis=0)/tx.shape[0]>threshold),axis=1)

def remove_points_with_MV(y, tx, MV):
    points_with_MV = np.where(np.count_nonzero(tx == MV, axis=1)>0)
    tx = np.delete(tx, points_with_MV, axis=0)
    #y  = np.delete(y,  points_with_MV, axis=0)
    return tx #, y

def replace_MV_by_average(y, tx, MV):
    avgs = np.avg(remove_points_with_MV(y, tx, MV))
    # Work in progress
    raise NotImplementedError

def replace_MV_by_zero(y, tx, MV):
    raise NotImplementedError

def remove_outliers(y, tx, threshold):
    raise NotImplementedError
    
def standarize(y, tx):
    raise NotImplementedError

def feature_augmentation(y, tx):
    raise NotImplementedError

