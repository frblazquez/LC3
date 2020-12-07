# -*- coding: utf-8 -*-
""" Module that contains various helper functions for the project """

# Libraries for general data management
import csv
import numpy as np
import pandas as pd

# Libraries for creating and validating models
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score

# Libraries for data visualization
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

# Libraries to model with r-style formulas
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Create, estimate, validate and show the model
def leave_one_out_validation(X, y, model=LinearRegression(), degree=1):
    assert(degree==1 or degree==2 or degree=='1' or degree=='2')
    
    # Train the model
    model.fit(X, y) 
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))
    if(degree==1 or degree=='1'):
        print(f"f(x) = {model.intercept_} + {model.coef_[0]}*x")
        ax.plot(X, np.dot(X,model.coef_) + model.intercept_,'r-')
        ax.scatter(X, y, edgecolors=(0, 0, 0))
    else:
        # This could be generalized but degree n >= 3 leads to overfitting!
        print(f"f(x) = {model.intercept_} + {model.coef_[0]}*x + {model.coef_[1]}*x^2")
        ax.plot(X[:,0], np.dot(X,model.coef_) + model.intercept_,'r-')
        ax.scatter(X[:,0], y, edgecolors=(0, 0, 0))
    ax.set_xlabel('% Kaolinite content')
    ax.set_ylabel('Compressive strength')
    plt.show()
    
    # Get the list of predictions obtained while validating
    predicted = cross_val_predict(model, X, y, cv=LeaveOneOut())
    
    # Model and metrics
    print(f"MSE: {mean_squared_error(y, predicted)}")
    print(f"R^2: {model.score(X,y)}")


# Helpers to create sklearn linear models

# Create a r-style formula
def create_r_formula(day, variables):
    formula = f"day_{day} ~ "
    equals = "+".join(variables)
    return formula+equals

# Get the adjusted R squared using the sm library
def get_model_r2_adj(name, formula, df):
    mods = smf.ols(formula=formula, data=df)
    res = mods.fit()
    return res.rsquared_adj

# Get the information about the model (sm library)
def get_model_summary(name, formula, df):
    mods = smf.ols(formula=formula, data=df)
    res = mods.fit()
    return res.summary()