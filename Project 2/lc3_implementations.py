# Francisco Javier Blázquez Martínez
# David Alonso Del Barrio 
# Andrés Montero Ranc
#
# École polytechnique fédérale de Lausanne, Switzerland
#
# Description:
# Implementations for lc3 compressive strength data analysis and confidence intervals


# Libraries for general data management
import pandas as pd
import numpy  as np

# Libraries for creating and validating models
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score

# Libraries for data visualization
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

# Libraries for statistical analisys
import statsmodels.api as sm
import statsmodels.formula.api as smf
from   statsmodels.sandbox.regression.predstd import wls_prediction_std

# Plots linear regression model and print model function and metrics
def leave_one_out_validation(X, y, model=LinearRegression()):
    # Train the model
    model.fit(X, y) 

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))

    if X.shape[1]==1:
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



# Function to return the R2 and validation score for a model (linear regression by default)
def get_model_validation(X, y, model=LinearRegression()):
    # Train the model
    model.fit(X, y) 
    # Get the list of predictions obtained while validating
    predicted = cross_val_predict(model, X, y, cv=LeaveOneOut())
    # Return the metrics
    return mean_squared_error(y, predicted)


# Function to perform feature selection from those given as parameter. It choses those
# features that better complements kaolinite content for achieving the best adj. R2 and MSE
def feature_selection(data, features, days=[1,3,7,28,90], print_report=False):
    # Empty dataframe to be fill with all the results
    results = pd.DataFrame(index=features) 
    # For every day we want to do feature selection
    for i in days:    
        day     = 'day_'+str(i)
        mses    = []
        r2s     = []
        bestR2  = -1 
        bestMse = float('inf')
        
        # Go for every feature given and check what results we get with it
        for feature in features:   
            # IMPORTANT! Metrics can cheat us if we drop NaNs!!
            # IMPORTANT! That's what we have to rely the features we are testing!!
            df = data[['Kaolinite_content', feature, str(i)+'D']].dropna()
            df['Kaolinite_content_square'] = (df['Kaolinite_content'].values)**2
            df = df.rename(columns={str(i)+'D':'day_'+str(i)})
            
            # Kaolinite content is always in our features in degree one and two
            X = df[['Kaolinite_content', 'Kaolinite_content_square', feature]].values
            y = df[day].values
        
            # Get the metrics
            mse = get_model_validation(X,y)
            r2  = smf.ols(formula=day+' ~ Kaolinite_content + Kaolinite_content_square + '+feature, data=df).fit().rsquared_adj
            
            mses.append(mse)
            r2s.append(r2)
            
            # Keep the bests
            if r2 > bestR2:
                bestR2         = r2
                bestR2_mse     = mse
                bestR2_feature = feature
            
            if mse < bestMse:
                bestMse        = mse
                bestMse_r2     = r2
                bestMse_feature= feature
        
        # Add this day results to the results dataframe
        results[day+'_mse']   = mses
        results[day+'_adjR2'] = r2s
        
        if print_report:
            print('=============================================================================')
            print('Best features for compression strength at day '+str(i))
            print('=============================================================================')
            print()
            print('Best R2  achieved for degree two kaolinite content and '+bestR2_feature)
            print('AR2: '+str(bestR2))
            print('MSE: '+str(bestR2_mse))
            print()
            print('Best MSE achieved for degree two kaolinite content and '+bestMse_feature)
            print('AR2: '+str(bestMse_r2))
            print('MSE: '+str(bestMse))
            print()
            
    return results


# Funcion for returning the data ready for creating models with kaolinite and a given feature
def get_model_data(data, feature, day, normalize=False, drop_nan=True, replace_nan=False):
    # Get kaolinite content in degree one and two and the parameter feature
    df_aux = data[['Kaolinite_content', feature, day]]
    df_aux.insert(1, 'Kaolinite_content_square', data['Kaolinite_content']**2, True)
    # Copy for data integrity if we replace NaN and when renaming
    df_aux = df_aux.copy()
    df_aux.rename(columns = {day : 'day_'+day[0]}, inplace = True)
    
    if drop_nan:
        df_aux = df_aux.dropna()
    elif replace_nan:
        df_aux.fillna(value=df_aux[feature].mean(), inplace=True)    
    if normalize:
        df_aux =(df_aux-df_aux.min())/(df_aux.max()-df_aux.min())
    
    return df_aux


# Function for ploting 0.9, 0.8, 0.7 and 0.6 confidence intervals of a model
# based on kaolinite content for a given day
def plot_confidence_intervals(data, day):
    X = data[['Kaolinite_content','Kaolinite_content_square']].values
    y = data['day_'+str(day)]
    
    res = smf.ols(formula='day_'+str(day)+' ~ Kaolinite_content + Kaolinite_content_square', data=data).fit()
    
    conf90 = res.conf_int(alpha=0.1)
    conf80 = res.conf_int(alpha=0.2)
    conf70 = res.conf_int(alpha=0.3)
    conf60 = res.conf_int(alpha=0.4)
    
    # This could be generalized but degree n >= 3 leads to overfitting!
    print('f(x) = {0} + {1}*x + {2}*x^2'.format(res.params[0],res.params[1],res.params[2]))
        
    # Get the list of predictions obtained while validating
    model = LinearRegression()
    model.fit(X,y)
    
    predicted = cross_val_predict(model, X, y, cv=LeaveOneOut())
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(X[:,0], y, edgecolors=(0, 0, 0))
    
    ax.plot(X[:,0], np.dot(X,[conf60[0][1],conf60[0][2]]) + conf60[0][0],'b-')
    ax.plot(X[:,0], np.dot(X,[conf70[0][1],conf70[0][2]]) + conf70[0][0],'b-')
    ax.plot(X[:,0], np.dot(X,[conf80[0][1],conf80[0][2]]) + conf80[0][0],'b-')
    ax.plot(X[:,0], np.dot(X,[conf90[0][1],conf90[0][2]]) + conf90[0][0],'b-')
    ax.plot(X[:,0], np.dot(X,model.coef_) + model.intercept_,'r-')
    ax.plot(X[:,0], np.dot(X,[conf90[1][1],conf90[1][2]]) + conf90[1][0],'b-')
    ax.plot(X[:,0], np.dot(X,[conf80[1][1],conf80[1][2]]) + conf80[1][0],'b-')
    ax.plot(X[:,0], np.dot(X,[conf70[1][1],conf70[1][2]]) + conf70[1][0],'b-')
    ax.plot(X[:,0], np.dot(X,[conf60[1][1],conf60[1][2]]) + conf60[1][0],'b-')
    
    ax.set_xlabel('% Kaolinite content')
    ax.set_ylabel('Compressive strength')
    plt.show()
    
    # Metrics for the model
    print("MSE: {}".format(mean_squared_error(y, predicted)))
    print("R^2: {}".format(model.score(X,y)))
    print()



