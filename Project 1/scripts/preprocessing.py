# École polytechnique fédérale de Lausanne, Switzerland
# CS-433 Machine Learning, project 1
# 
# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
# David Alonso del Barrio            ~ david.alonsodelbarrio@epfl.ch
# Andrés Montero Ranc                ~ andres.monteroranc@epfl.ch
#
# Data preprocessing utilities.

import numpy as np

# Function for data preprocessing:
#
# MVT = Missing values threshold
# CT  = Correlation threshold
# OT  = Outliers threshold
# AD  = Augmentation degree
def preprocess_data(y_train, tx_train, tx_test, MVT=1, CT=1, OT=100, AD=1, std_data=True, ones_column=True):

    # Remove features with missing values frequency greater than MVT
    high_MV_freq_features = get_high_MV_freq_features(tx_train, MVT)
    tx_train = np.delete(tx_train, high_MV_freq_features, axis=1)
    tx_test  = np.delete(tx_test,  high_MV_freq_features, axis=1)

    # Remove features with correlation coefficient greater than CT
    corr_features = get_correlated_features(y_train, tx_train, CT)
    tx_train = np.delete(tx_train, corr_features, axis=1)  
    tx_test  = np.delete(tx_test,  corr_features, axis=1)    

    # Replace remaining missing values with average of valid measurements
    replace_MV_by_average(tx_train)
    replace_MV_by_average(tx_test) 

    # Remove outliers (any measurement deviated more than OT*std from the mean)
    y_train, tx_train = remove_outliers(y_train, tx_train, OT)

    # Feature augmentation until degree AD, for order two we add pair columns products
    tx_train = feature_augmentation(tx_train, AD)
    tx_test  = feature_augmentation(tx_test,  AD)  

    # Standarize data
    if std_data:
        tx_train = standarize(tx_train)
        tx_test  = standarize(tx_test)

    # Add ones column (for having w0 variable)
    if ones_column:
        tx_train = np.c_[np.ones(tx_train.shape[0]), tx_train]
        tx_test  = np.c_[np.ones(tx_test.shape[ 0]), tx_test ]

    return y_train, tx_train, tx_test


# Removes features with Pearson correlation coefficient greater than threshold
def get_correlated_features(y, tx, threshold):
    # First we compute the correlation coefficient of every pair
    corr_coef = np.corrcoef(tx, y, False)
    # Then we get one of every pair of correlated features
    highly_correlated_features = []
    for i in range(corr_coef.shape[1]-1):
        if i in highly_correlated_features:
            continue
        for j in range(i+1, corr_coef.shape[1]):
            if j in highly_correlated_features:
                continue
            if abs(corr_coef[i][j]) > threshold:
                highly_correlated_features.append(j)
    return highly_correlated_features

# Removes features with missing values frequency greater than threshold
def get_high_MV_freq_features(tx, threshold, MV=-999):
    return np.where(np.count_nonzero(tx == MV, axis=0)/tx.shape[0]>threshold)

# Removes points in tx with missing values
def remove_points_with_MV(tx, MV=-999):
    return np.delete(tx, np.where(np.count_nonzero(tx == MV, axis=1)>0, axis=0))

# Replaces every missing value entry with the average of valid values in that feature
def replace_MV_by_average(tx, MV=-999):
    # Be aware we are modifying tx!    
    tx[tx == MV] = np.nan
    col_mean = np.nanmean(tx, axis=0)
    inds = np.where(np.isnan(tx))
    tx[inds] = np.take(col_mean, inds[1])
    return tx

# Replaces every missing value entry with zero (modifies tx)
def replace_MV_by_zero(tx, MV=-999):
    # Be aware we are modifying tx!  
    tx[tx == MV] = 0

# Removes points deviated from the mean more than given factor of standard deviations
def remove_outliers(y, tx, threshold=4):    
    z_train = np.abs((tx - tx.mean(axis=0, keepdims=True)) / tx.std(axis=0, ddof=0, keepdims=True))
    return y[((z_train < threshold).all(axis=1))], tx[((z_train < threshold).all(axis=1))]

# Z standarization of the data
def standarize(tx):
    return (tx - np.mean(tx, axis=0)) / np.std(tx, axis=0)

# Add new features to increase (partially) regression order
def feature_augmentation(tx, k):
    if k < 2:
        return tx
    # We add columns raised to exponents until k
    tx_aux = tx
    for i in range(2,k+1):
        tx_aux = np.append(tx_aux, np.power(tx,i), axis=1)
    # We add columns cross products
    cross_terms_order_2 = np.array([tx[:, i] * tx[:, j] for i in range(tx.shape[1]) for j in range(i+1, tx.shape[1])]).T
    tx_aux = np.append(tx_aux, cross_terms_order_2, axis=1)
    return tx_aux


