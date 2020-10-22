# Data preprocessing utilities for Machine Learning course, project 1
#
# École polytechnique fédérale de Lausanne, Switzerland

# TODO: Handle PRI_jet_num categorical variable!
# TODO: Check there is no NaNs
# TODO: Implement all the functions
# TODO: Remove low variance features ? 

import numpy as np

def remove_highly_correlated_features(y, tx, threshold):
    raise NotImplementedError

def remove_features_with_MV(y, tx, MV, threshold):
    raise NotImplementedError

def remove_points_with_MV(y, tx, MV):
    raise NotImplementedError
    
def replace_MV_by_average(y, tx, MV):
    raise NotImplementedError

def replace_MV_by_zero(y, tx, MV):
    raise NotImplementedError

def remove_outliers(y, tx, threshold):
    raise NotImplementedError
    
def standarize(y, tx):
    raise NotImplementedError

def feature_augmentation(y, tx):
    raise NotImplementedError

