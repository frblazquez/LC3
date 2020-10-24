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

# Removes points in tx with missing values, it doesn't modify y!
def remove_points_with_MV(y, tx, MV):
    #points_with_MV = np.where(np.count_nonzero(tx == MV, axis=1)>0)
    #tx = np.delete(tx, points_with_MV, axis=0)
    #y  = np.delete(y,  points_with_MV, axis=0)
    return np.delete(tx, np.where(np.count_nonzero(tx == MV, axis=1)>0, axis=0))

# Replaces every missing value entry with the average of valid values in that feature (modifies tx)
def replace_MV_by_average(y, tx, MV):
    # NOT CORRECT!    
    tx[tx == MV] = np.mean(remove_points_with_MV(y, tx, MV))

# Replaces every missing value entry with zero (modifies tx)
def replace_MV_by_zero(y, tx, MV):
    tx[tx == MV] = 0

# Removes points deviated from the mean more than a std factor given
def remove_outliers(y, tx, threshold):
    # NOT CORRECT!
    return tx[np.linalg.norm(tx - np.mean(tx,axis=0)) < threshold*np.std(tx)]
    
# Z standarization of the data
def standarize(y, tx):
    return (tx - np.mean(tx, axis=0)) / np.std(tx, axis=0)

# 
def feature_augmentation(y, tx):
    raise NotImplementedError

