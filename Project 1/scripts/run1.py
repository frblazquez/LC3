# École polytechnique fédérale de Lausanne, Switzerland
# CS-433 Machine Learning, project 1
# 
# Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch
# David Alonso del Barrio            ~ david.alonsodelbarrio@epfl.ch
# Andrés Montero Ranc                ~ andres.monteroranc@epfl.ch
#
# Function to generate output file with predictions over test set with
# the best model we have found.

from proj1_helpers import *
from implementations import *
from cross_validation import *
from preprocessing import *

DATA_TRAIN_PATH = './data/train.csv'
DATA_TEST_PATH  = './data/test.csv'
OUTPUT_PATH     = './data/output.csv'

print('École polytechnique fédérale de Lausanne, Switzerland')
print('CS-433 Machine Learning, project 1')
print()
print('Francisco Javier Blázquez Martínez ~ francisco.blazquezmartinez@epfl.ch')
print('David Alonso del Barrio            ~ david.alonsodelbarrio@epfl.ch')
print('Andrés Montero Ranc                ~ andres.monteroranc@epfl.ch')
print()
print('Script for generating output.csv with our best output file')
print()
print()
   
# Generating output.csv with best precision
print('Generating output.csv with best precision...')

y_train, tx_train, ids      = load_csv_data(DATA_TRAIN_PATH)
y_test,  tx_test,  ids_test = load_csv_data(DATA_TEST_PATH)

# Best ridge regression parameters found
MVT = 1
CT = 1
OT = 10
AD = 4
LAMBDA = 11

y_train, tx_train, tx_test = preprocess_data(y_train, tx_train, tx_test, MVT, CT, OT, AD)

# computing loss
w, loss = ridge_regression(y_train, tx_train,11)

#generating output.csv
y_pred = predict_labels(w, tx_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


