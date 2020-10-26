# Data preprocessing utilities for Machine Learning course, project 1
#
# École polytechnique fédérale de Lausanne, Switzerland

import numpy as np

# Removes features with high Pearson correlation coefficient
def get_correlated_features(y, tx, threshold):
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
    return highly_correlated_features

# Removes features with missing values frequency greater than threshold
def get_high_MV_freq_features(y, tx, threshold, MV=-999):
    return np.where(np.count_nonzero(tx == MV, axis=0)/tx.shape[0]>threshold)

# Removes points in tx with missing values, it doesn't modify y!
def remove_points_with_MV(y, tx, MV=-999):
    return np.delete(tx, np.where(np.count_nonzero(tx == MV, axis=1)>0, axis=0))

# Replaces every missing value entry with the average of valid values in that feature (modifies tx)
def replace_MV_by_average(y, tx, MV=-999):
    tx[tx == MV] = np.nan
    col_mean = np.nanmean(tx, axis=0)
    inds = np.where(np.isnan(tx))
    tx[inds] = np.take(col_mean, inds[1])
    return tx

# Replaces every missing value entry with zero (modifies tx)
def replace_MV_by_zero(y, tx, MV=-999):
    tx[tx == MV] = 0

# Removes points deviated from the mean more than a std factor given
def remove_outliers(y, tx, threshold=4):    
    z_train = np.abs((tx - tx.mean(axis=0, keepdims=True)) / tx.std(axis=0, ddof=0, keepdims=True))
    return y[((z_train < threshold).all(axis=1))], tx[((z_train < threshold).all(axis=1))]

# Z standarization of the data
def standarize(y, tx):
    return (tx - np.mean(tx, axis=0)) / np.std(tx, axis=0)

# Add new features to increase (partially) regression order
def feature_augmentation(y, tx, k):
    tx_aux = tx
    for i in range(2,k+1):
        tx_aux = np.append(tx_aux, np.power(tx,i), axis=1)
    if i > 1:
        cross_terms_order_2 = np.array([tx[:, i] * tx[:, j] for i in range(tx.shape[1]) for j in range(i+1, tx.shape[1])]).T
        tx_aux = np.append(tx_aux, cross_terms_order_2, axis=1)
    return tx_aux
        

# Create this function from the preprocessing section in cross_validation.py
#
# def preprocess_data(y_train, tx_train, tx_test, mvt, ct, ot, ad):
#        corr_features = get_correlated_features(y_train, tx_train, ct)
#        tx_train = np.delete(tx_train, corr_features, axis=1)  
#        tx_test  = np.delete(tx_test,  corr_features, axis=1)    
#
#        high_MV_freq_features = get_high_MV_freq_features(y_train, tx_train, mvt)
#        tx_train = np.delete(tx_train, high_MV_freq_features, axis=1)
#        tx_test  = np.delete(tx_test,  high_MV_freq_features, axis=1)
#        .....

    

