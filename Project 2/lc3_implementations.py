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
from scipy import stats

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


def create_weights(data_dx, std_squared=False):
    """
    Helper function to compute the weights
    
    :data_dx: dataframe only with columns to model
              [day_x, STD_xD, Kaolinite_content]
    
    :returns: array of weights to input in the WLS
    """
    
    std_head = [i for i in data_dx if i.startswith('STD_')][-1]
    std = data_dx[std_head].values
    norm_std = (std-std.min())/(std.max()-std.min())
    norm_std[norm_std==0] = 1e-4
    
    if(std_squared):
        return 1./(norm_std**2)
    else:
        return 1./(norm_std)

def compute_WLS_model(data, day, summary=False, plot=False, show_bounds=False, show_ols=False, weights2=False):
    """
    Helper function to compute the Weighted Least
    Squares model. It returns the model and the results.
    It can print the summary and show a plot describing
    the wls. It can also provide in the plot the bounds
    given by the weights but also a comparison with OLS 
    
    :data: dataframe, this may contain all columns but
         must contain at_least columns day_x, STD_xD, 
         Kaolinite_content with that specific sintax
    :day: int with the day we want to model, must be
          1, 3, 7, 28, or 90.
    :summary: if True prints the summary of the model
    :plot: if True plots the wls linear model prediction
    :show_bounds: shows the upper and lower quantiles
                  with the weights given and 0,975 ci
    :show_ols: if True shows a comparison with an
               Ordinary Leat Squares model prediction
    :weights2: when True weights are 1/std^2 instead of
               1/std
              
    :returns: modelWLS: the model, resultWLS: its results
    """
    data_dx = data[["Kaolinite_content", f"day_{day}", f"STD_{day}D"]]
    data_dx = data_dx.dropna()
    
    x = data_dx["Kaolinite_content"].values
    X = sm.add_constant(x, has_constant='skip')
    Yx  = data_dx[f"day_{day}"].values
    weightsx = create_weights(data_dx)
    if(weights2):
        weightsx = create_weights(data_dx, std_squared=True)

    mod_wls = sm.WLS(Yx, X, weights=weightsx)
    res_wls = mod_wls.fit()
    
    if(summary):
        print(res_wls.summary())
    if(plot):
        # plot WLS
        prstd, iv_l, iv_u = wls_prediction_std(res_wls)

        fig, ax = plt.subplots(figsize=(8,6))

        #Datapoints
        ax.plot(x, Yx, 'o', label="Data")

        # WLS (weighted least squares) prediction
        ax.plot(x, res_wls.fittedvalues, 'g--.', label="WLS")
        if(show_bounds):
            ax.plot(x, iv_u, 'g--')
            ax.plot(x, iv_l, 'g--')
        ax.set_xlabel('% Kaolinite content')
        ax.set_ylabel('Compressive strength')
        
        if(show_ols):
            # OLS (ordinary least squares) prediction
            res_ols = sm.OLS(Yx, X).fit()

            ax.plot(x, res_ols.fittedvalues, 'r--', label="OLS")
            if(show_bounds):
                # Calculate prediction interval
                tppf = stats.t.ppf(0.975, res_ols.df_resid)
                prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(res_ols)
                ax.plot(x, iv_u_ols, 'r--')
                ax.plot(x, iv_l_ols, 'r--')

        ax.legend(loc="best");
        plt.show()
    return mod_wls, res_wls
    
# Plots linear regression model and print model function and metrics
def leave_one_out_validation(X, y, day, model=LinearRegression()):
    # Train the model
    model.fit(X, y) 
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))
    #OPC REFERENCIAS
    if(day==1): 
        OPC = 23.75
    if(day==3): 
        OPC = 35.52
    if(day==7): 
        OPC = 40.38
    if(day==28): 
        OPC = 50.25
    if(day==90):
        OPC = 50.42
    plt.axhline(y = OPC, color = 'darkorange', linestyle = '--', label='OPC') # OPC reference
    plt.legend()
    
    if X.shape[1]==1:
        print(f"f(x) = {model.intercept_} + {model.coef_[0]}*x")
        ax.plot(X, np.dot(X,model.coef_) + model.intercept_,'r-',label='regression line')
        ax.scatter(X, y, edgecolors=(0, 0, 0))
    else:
        # This could be generalized but degree n >= 3 leads to overfitting!
        print(f"f(x) = {model.intercept_} + {model.coef_[0]}*x + {model.coef_[1]}*x^2")
        ax.plot(X[:,0], np.dot(X,model.coef_) + model.intercept_,'r-', label='regression line')
        ax.scatter(X[:,0], y, edgecolors=(0, 0, 0))

    ax.set_xlabel('% Kaolinite content')
    ax.set_ylabel('Compressive strength')
    ax.legend()
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


# Function to perform feature selection from those given as parameter over a model with base features
# the Kaolinite content (in degree one and two) and those given in "other_base_features" parameter. 
# It choses those features that better complements kaolinite content for achieving the best adj. R2 and MSE
def feature_selection(data, features, days=[1,3,7,28,90], print_report=False, other_base_features=[]):
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
            df = data[['Kaolinite_content', feature] + other_base_features + [day]].dropna()
            df['Kaolinite_content_square'] = (df['Kaolinite_content'].values)**2
            
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
        # Select cols to highlight min MSE and max R2 by columns
        cols_mse= results.columns.str.endswith("mse")
        cols_R2 = results.columns.str.endswith("R2")
        subset_mse=pd.IndexSlice[:, cols_mse]
        subset_R2=pd.IndexSlice[:, cols_R2]
       

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
        
    return results.style.highlight_min(color = 'lightgreen', axis = 0, subset=subset_mse).highlight_max(color = 'red', axis = 0, subset=subset_R2)

# Funcion for returning the data ready for creating models with kaolinite and a given feature
def get_model_data(data, feature, day, normalize=False, drop_nan=True, replace_nan=False):
    # Get kaolinite content in degree one and two and the parameter feature
    df_aux = data[['Kaolinite_content', feature, day]]
    df_aux.insert(1, 'Kaolinite_content_square', data['Kaolinite_content']**2, True)
    # Copy for data integrity if we replace NaN and when renaming
    df_aux = df_aux.copy()
#     df_aux.rename(columns = {day : 'day_'+day[0]}, inplace = True)
    
    if drop_nan:
        df_aux = df_aux.dropna()
    elif replace_nan:
        df_aux.fillna(value=df_aux[feature].mean(), inplace=True)    
    if normalize:
        df_aux =(df_aux-df_aux.min())/(df_aux.max()-df_aux.min())
    
    return df_aux

# Funcion for returning the data ready for creating models with kaolinite 
def get_model_data_kaolinite(data, day, normalize=False, drop_nan=True, replace_nan=False):
    # Get kaolinite content in degree one and two and the parameter feature
    df_aux = data[['Kaolinite_content', day]]
    df_aux.insert(1, 'Kaolinite_content_square', data['Kaolinite_content']**2, True)
    # Copy for data integrity if we replace NaN and when renaming
    df_aux = df_aux.copy()
#     df_aux.rename(columns = {day : 'day_'+day[0]}, inplace = True)
    
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
    
    #OPC REFERENCIAS
    if(day==1): 
        OPC = 23.75
    if(day==3): 
        OPC = 35.52
    if(day==7): 
        OPC = 40.38
    if(day==28): 
        OPC = 50.25
    if(day==90):
        OPC = 50.42
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(X[:,0], y, edgecolors=(0, 0, 0))
    
    ax.plot(X[:,0], np.dot(X,[conf60[0][1],conf60[0][2]]) + conf60[0][0],'b--',label='CI 60%')
    ax.plot(X[:,0], np.dot(X,[conf70[0][1],conf70[0][2]]) + conf70[0][0],'c--',label='CI 70%')
    ax.plot(X[:,0], np.dot(X,[conf80[0][1],conf80[0][2]]) + conf80[0][0],'m--',label='CI 80%')
    ax.plot(X[:,0], np.dot(X,[conf90[0][1],conf90[0][2]]) + conf90[0][0],color='navy', linestyle='dashed',label='CI 90%')
    
    ax.plot(X[:,0], np.dot(X,model.coef_) + model.intercept_,'r-')
    ax.plot(X[:,0], np.dot(X,[conf90[1][1],conf90[1][2]]) + conf90[1][0],color='navy', linestyle='dashed')
    ax.plot(X[:,0], np.dot(X,[conf80[1][1],conf80[1][2]]) + conf80[1][0],'m--')
    ax.plot(X[:,0], np.dot(X,[conf70[1][1],conf70[1][2]]) + conf70[1][0],'c--')
    ax.plot(X[:,0], np.dot(X,[conf60[1][1],conf60[1][2]]) + conf60[1][0],'b--')
    ax.legend()
    plt.axhline(y = OPC, color = 'darkorange', linestyle = '--', label='OPC') # OPC reference
    plt.legend()
    ax.set_xlabel('% Kaolinite content')
    ax.set_ylabel('Compressive strength')
    plt.xticks(range(0, 100, 10))
    plt.grid(color='lightgrey',which= 'both', linestyle='-', linewidth=1)
    plt.show()
    
    # Metrics for the model
    print("MSE: {}".format(mean_squared_error(y, predicted)))
    print("R^2: {}".format(model.score(X,y)))
    print()

# Create a r-style formula
def create_r_formula(day, variable):
    formula = f"day_{day} ~ Kaolinite_content + Kaolinite_content_square + {variable}"
#     equals = "+".join(variables)
    return formula

# Get the adjusted R squared using the sm library
def get_model_r2_adj(formula, df):
    mods = smf.ols(formula=formula, data=df)
    res = mods.fit()
    return res.rsquared_adj

# Get the information about the model (sm library)
def get_model_summary(formula, df, param_plot=False):
    mods = smf.ols(formula=formula, data=df)
    res = mods.fit()
    
    if(param_plot):
        # feature names
        variables = res.params.index

        # quantifying uncertainty!
        coefficients = res.params.values
        p_values = res.pvalues
        standard_errors = res.bse.values

        # sort them all by coefficients
        l1, l2, l3, l4 = zip(*sorted(zip(coefficients[1:], variables[1:], standard_errors[1:], p_values[1:])))

        # plot
        plt.errorbar(l1, np.array(range(len(l1))), xerr= 2*np.array(l3), linewidth = 1,
                     linestyle = 'none',marker = 'o',markersize= 3,
                     markerfacecolor = 'black',markeredgecolor = 'black', capsize= 5)

        plt.vlines(0,0, len(l1), linestyle = '--')

        plt.yticks(range(len(l2)),l2);
        plt.show()
    return res.summary()

# Rename columns in dataframe to work with them in statistical analisys
def rename_cols(data): 
    # We rename some columns for having an easier reference
    data.rename(columns = {'Calcined kaolinite content (%)':'Kaolinite_content'}, inplace = True)
    data.rename(columns = {'Dv,50 (µm)':'Dv50'                                 }, inplace = True)
    data.rename(columns = {'BET Specific surface (m2/g)':'BET_specific_surface'}, inplace = True)
    data.rename(columns = {'Span (-)':'span'                                   }, inplace = True)

    data.rename(columns = {'STD'  : 'STD_1D'}, inplace = True)
    data.rename(columns = {'STD.1': 'STD_3D'}, inplace = True)
    data.rename(columns = {'STD.2': 'STD_7D'}, inplace = True)
    data.rename(columns = {'STD.3':'STD_28D'}, inplace = True)
    data.rename(columns = {'STD.4':'STD_90D'}, inplace = True)

    # Sorting allows us to plot functions more easily
    data = data.sort_values('Kaolinite_content')

def rename_std_cols(data):
    # We rename some columns for having an easier reference
    data.rename(columns = {'Calcined kaolinite content (%)':'Kaolinite_content'}, inplace = True)
    data.rename(columns = {'Dv,50 (µm)':'Dv50'                                 }, inplace = True)
    data.rename(columns = {'BET Specific surface (m2/g)':'BET_specific_surface'}, inplace = True)

    data.rename(columns = {'1D'  : 'day_1'}, inplace = True)
    data.rename(columns = {'3D': 'day_3'}, inplace = True)
    data.rename(columns = {'7D': 'day_7'}, inplace = True)
    data.rename(columns = {'28D':'day_28'}, inplace = True)
    data.rename(columns = {'90D':'day_90'}, inplace = True)

    data.rename(columns = {'STD'  : 'STD_1D'}, inplace = True)
    data.rename(columns = {'STD.1': 'STD_3D'}, inplace = True)
    data.rename(columns = {'STD.2': 'STD_7D'}, inplace = True)
    data.rename(columns = {'STD.3':'STD_28D'}, inplace = True)
    data.rename(columns = {'STD.4':'STD_90D'}, inplace = True)

    # Sorting allows us to plot functions more easily
    data = data.sort_values('Kaolinite_content')


def deviated_points_detection(data, day):    
    data = get_model_data_kaolinite(data, 'day_'+str(day))
    
    X = data[['Kaolinite_content','Kaolinite_content_square']].values
    y = data['day_'+str(day)]
    
    res = smf.ols(formula='day_'+str(day)+' ~ Kaolinite_content + Kaolinite_content_square', data=data).fit()
    
    conf90 = res.conf_int(alpha=0.1)
    
    lower_bound = np.dot(X,[conf90[0][1],conf90[0][2]]) + conf90[0][0]
    upper_bound = np.dot(X,[conf90[1][1],conf90[1][2]]) + conf90[1][0]
    
    print("Optimistic deviated points:")
    print(data[data['day_'+str(day)] > upper_bound])
    print("Pesimistic deviated points:")
    print(data[data['day_'+str(day)] < lower_bound])


def visualize_data(x1,y1,x3,y3,x7,y7,x28,y28,x90,y90):
    # Show points using matplotlib.pyplot library
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(x1,y1,'c^',x3,y3,'bs',x7,y7,'r^',x28,y28,'go', x90,y90,'m^' )
    plt.xlabel('%Kaolinite Content')
    plt.ylabel('Compressive Strenght')
    
    d1_patch  = mpatches.Patch(color='cyan',      label='After  1 day')
    d3_patch  = mpatches.Patch(color='blue',      label='After  3 days')
    d7_patch  = mpatches.Patch(color='red',       label='After  7 days')
    d28_patch = mpatches.Patch(color='darkgreen', label='After 28 days')
    d90_patch = mpatches.Patch(color='purple',    label='After 90 days')
    plt.legend(handles=[d1_patch,d3_patch,d7_patch,d28_patch,d90_patch])

    plt.show()


def load_full_data(path):
    # Read full data and remove empty lines
    data_full = pd.read_excel(path,sheet_name='Clays_CS',na_values=['-'])
    data_full.dropna(how="all", inplace=True)

    # Read clay properties
    data_clay   = pd.read_excel(path,sheet_name='Clays_properties', na_values=['-'])

    # Merge to have the whole dataset
    data_full_clay = pd.merge(data_full, data_clay, left_on='Clay', right_on='Clay', how='left')
    data_full_clay = data_full_clay.sort_values("Calcined kaolinite content (%)")

    # We rename some columns for having an easier reference
    rename_cols(data_full_clay)

    return data_full_clay




