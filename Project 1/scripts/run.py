# École polytechnique fédérale de Lausanne, Switzerland
# CS-433 Machine Learning, project 1
# 
# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
# David Alonso del Barrio            ~ david.alonsodelbarrio@epfl.ch
# Andrés Montero Ranc                ~ andres.monteroranc@epfl.ch
#
# Function to cross validate our models with their best parameters found.

from cross_validation import *

DATA_TRAIN_PATH = '../data/train.csv'
# DATA_TEST_PATH  = '../data/test.csv'
# OUTPUT_PATH_LS  = '../data/output_LS.csv'
# OUTPUT_PATH_RR  = '../data/output_RR.csv'
# OUTPUT_PATH_LR  = '../data/output_LR.csv'
# OUTPUT_PATH_RLR = '../data/output_RLR.csv'

print('École polytechnique fédérale de Lausanne, Switzerland')
print('CS-433 Machine Learning, project 1')
print()
print('Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch')
print('David Alonso del Barrio            ~ david.alonsodelbarrio@epfl.ch')
print('Andrés Montero Ranc                ~ andres.monteroranc@epfl.ch')
print()
print('Script for validation of the best models obtained')
print()
print()
    
# Get train and test data from specified path
y_train, tx_train, ids = load_csv_data(DATA_TRAIN_PATH)

# Least squares: MVT=1 CT=0.925 OT=11 AD=4
print('Validating least squares model...')
p_LS = cross_validation_LS(y_train.copy(), tx_train.copy(), 10, 1,0.925, 11, 4)
    
# Ridge regression: MVT=1, CT=1, OT=10, AD=4, LAMBDA=11
print('Validating ridge regression model...')
p_RR = cross_validation_RR(y_train.copy(), tx_train.copy(), 10, 1, 1, 10, 4, 11)

# Logistic regression: MVT=1 CT=1 OT=6 AD=2, GAMMA=1e-06
print('Validating logistic regression model...')
p_LR = cross_validation_LR(y_train.copy(), tx_train.copy(), 10, 1, 1, 6, 2, 1e-6)

# Regularized logistic regression: MVT=1, CT=0.85, OT=8, AD=8, LAMBDA=1, GAMMA=1e-7
print('Validating regularized logistic regression model...')
p_RLR= cross_validation_RLR(y_train.copy(), tx_train.copy(), 10, 1,0.85, 8, 8, 1, 1e-7)

# Print best precissions achieved
print()
print('Precission obtained with least squares            = '+str(p_LS))
print('Precission obtained with ridge regression         = '+str(p_RR))
print('Precission obtained with logistic regression      = '+str(p_LR))
print('Precission obtained with reg. logistic regression = '+str(p_RLR))
   
