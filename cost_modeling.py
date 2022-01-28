import sys
import numpy as np
import pandas as pd
from os import chdir
import seaborn as sns 
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import PureWindowsPath, Path

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.tools.eval_measures import meanabs
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import OLSInfluence as influence

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate


# Read in data
project_dir = PureWindowsPath(r"D:\GitHubProjects\medical_cost_prediction\\")
chdir(project_dir)
dataset = pd.read_csv('./input/insurance.csv')
models_output_dir = Path(project_dir, Path('./output/models'))

# Import my data science helper functions (relative dir based on project_dir)
my_module_dir = str(Path.resolve(Path('../my_ds_modules')))
sys.path.insert(0, my_module_dir)
import ds_helper as dh


# ====================================================================================================================
# Categorize and process features
# ====================================================================================================================
# Separate categorical and numerical features
categorical_cols = [cname for cname in dataset.columns if dataset[cname].dtype == "object"]
numerical_cols = [cname for cname in dataset.columns if not dataset[cname].dtype == "object"]

# See if there are any 'numerical' columns that actually contain encoded categorical data
num_uniques = dataset[numerical_cols].nunique()

# Feature 'children' is discrete and contains whole numbers 0 to 5, inclusive
dataset['children'].unique()

# Create list of categorical variables with target and one without target
num_cols_w_target = numerical_cols.copy()
numerical_cols.remove('charges')

# Create list of categorical + ordinal variables for certain tasks
cat_ord_cols = categorical_cols.copy()
cat_ord_cols.append('children')

# Create list of continuous variables with target and one without target
cont_cols = numerical_cols.copy()
cont_cols.remove('children')
cont_cols_w_target = cont_cols.copy()
cont_cols_w_target.append('charges')

# Create formatted columns dictionary in helper module
custom_dict = {}
custom_dict['bmi'] = 'BMI'
custom_dict['bmi_>=_30'] = 'BMI >= 30'
dh.create_formatted_cols_dict(dataset.columns, custom_dict)

# ====================================================================================================================
# Visualization helper functions
# ====================================================================================================================
# Function returning the formatted version of column name
def format_col(col_name):
    return dh.format_col(col_name)
    
# ====================================================================================================================
# Data preprocessing function via pipeline
# ====================================================================================================================
def create_pipeline(model_name, model):
    # Preprocessing for numerical data (SimpleImputer default strategy='mean')
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()), 
        ('scale', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
    ])
    
    # Bundle preprocessor and model
    my_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        (model_name, model)
    ])
    return my_pipeline

# ====================================================================================================================
# Data preprocessing function without using pipeline
# ====================================================================================================================
def manual_preprocess(X_train, X_valid):
    # =============================
    # Numerical preprocessing
    # =============================
    X_train_num = X_train[numerical_cols]
    X_valid_num = X_valid[numerical_cols]
    
    # Imputation (Not relevant in this dataset, but keeping for future application)
    #num_imputer = SimpleImputer(strategy='mean')
    #imputed_X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    #imputed_X_valid_num = pd.DataFrame(num_imputer.transform(X_valid_num), columns=X_valid_num.columns, index=X_valid_num.index)
    
    # Scaling
    ss = StandardScaler()
    scaled_X_train_num = pd.DataFrame(ss.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    scaled_X_valid_num = pd.DataFrame(ss.transform(X_valid_num), columns=X_valid_num.columns, index=X_valid_num.index)
    
    # =============================
    # Categorical preprocessing
    # =============================
    X_train_cat = X_train[categorical_cols]
    X_valid_cat = X_valid[categorical_cols]
    
    # Imputation (Not relevant in this dataset, but keeping for future application)
    #cat_imputer = SimpleImputer(strategy='most_frequent')
    #imputed_X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train_cat), columns=X_train_cat.columns, index=X_train_cat.index)
    #imputed_X_valid_cat = pd.DataFrame(cat_imputer.transform(X_valid_cat), columns=X_valid_cat.columns, index=X_valid_cat.index)
    
    # One-hot encoding
    OH_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_cat), index=X_train_cat.index, columns=OH_encoder.get_feature_names_out())
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid_cat), index=X_valid_cat.index, columns=OH_encoder.get_feature_names_out())
    
    # Add preprocessed categorical columns back to preprocessed numerical columns
    X_train_processed = pd.concat([scaled_X_train_num, OH_cols_train], axis=1)
    X_valid_processed = pd.concat([scaled_X_valid_num, OH_cols_valid], axis=1)
    
    return X_train_processed, X_valid_processed

# Preprocessing of all indepedent variable data together (no train/test split) for use with statmodels (sm) data analysis
def manual_preprocess_sm(X):
    # =============================
    # Numerical preprocessing
    # =============================
    X_num = X[numerical_cols]
        
    # Scaling
    ss = StandardScaler()
    scaled_X_num = pd.DataFrame(ss.fit_transform(X_num), columns=X_num.columns, index=X_num.index)
    
    # =============================
    # Categorical preprocessing
    # =============================
    X_cat = X[categorical_cols]
        
    # One-hot encoding
    OH_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
    OH_X_cat = pd.DataFrame(OH_encoder.fit_transform(X_cat), index=X_cat.index, columns=OH_encoder.get_feature_names_out())
    
    # Add preprocessed categorical columns back to preprocessed numerical columns
    X_processed = pd.concat([scaled_X_num, OH_X_cat], axis=1)
    
    # Add constant (required for statsmodels linear regression model)
    X_processed = sm.add_constant(X_processed)
    
    return X_processed

# ====================================================================================================================
# Model evaluation functions
# ====================================================================================================================
# Written for sklearn linear regression models, but only require y, y_pred, and X, so could be used for any model
# Using the 'evaluate_model_sk()' and 'evaluate_model_sm()' on the SAME sm model yield almost identical results
def evaluate_model_sk(y_valid, y_pred, X, round_results=3, print_results=False):      
    metrics = {}
    metrics['max_e'] = max_error(y_valid, y_pred).round(round_results)
    metrics['mae'] = mean_absolute_error(y_valid, y_pred).round(round_results)
    metrics['mse'] = mean_squared_error(y_valid, y_pred).round(round_results)
    metrics['rmse'] = np.sqrt(metrics['mse']).round(round_results)
    metrics['med_abs_e'] = median_absolute_error(y_valid, y_pred).round(round_results)
    metrics['r2'] = r2_score(y_valid, y_pred).round(round_results)
    metrics['r2_adj'] = 1 - ((1-metrics['r2'])*(len(y_valid)-1)/(len(y_valid)-X.shape[1]-1)).round(round_results)
    
    if print_results:
        print('Max Error: ' + str(metrics['max_e']))
        print('Mean Absolute Error: ' + str(metrics['mae']))
        print('Mean Squared Error: ' + str(metrics['mse']))
        print('Root Mean Squared Error: ' + str(metrics['rmse']))
        print('Median Absolute Error: ' + str(metrics['med_abs_e']))
        print('R-squared: ' + str(metrics['r2']))
        print('R-squared (adj): ' + str(metrics['r2_adj']))
    return metrics

# Written for statsmodels model
# Using the 'evaluate_model_sk()' and 'evaluate_model_sm()' on the same sm model yield almost identical results
def evaluate_model_sm(y, y_pred, sm_lr_model, round_results=3, print_results=False):      
    metrics = {}
    metrics['max_e'] = np.round(max(sm_lr_model.resid), round_results)
    metrics['mae'] = meanabs(y, y_pred).round(round_results)
    metrics['mse'] = sm_lr_model.mse_resid.round(round_results) # this is the closet metric to mse. It was within 3% of both my calculation and sklearn's. 'mse_model' and 'mse_total' were 99% and 75% different
    metrics['rmse'] = np.sqrt(metrics['mse']).round(round_results)
    metrics['med_abs_e'] = np.median(abs(sm_lr_model.resid)).round(round_results)
    metrics['r2'] = sm_lr_model.rsquared.round(round_results)
    metrics['r2_adj'] = sm_lr_model.rsquared_adj.round(round_results)
    
    # Quantify Heteroscedasticity using Breusch-Pagan test and White test 
    bp_test = het_breuschpagan(sm_lr_model.resid, sm_lr_model.model.exog)
    white_test = het_white(sm_lr_model.resid, sm_lr_model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    bp_test_results = dict(zip(labels, bp_test))
    white_test_results = dict(zip(labels, white_test))
    metrics['bp_lm_p'] = '{:0.3e}'.format(bp_test_results['LM-Test p-value'])
    metrics['white_lm_p'] = '{:0.3e}'.format(white_test_results['LM-Test p-value'])
    
    if print_results:
        print('Max Error: ' + str(metrics['max_e']))
        print('Mean Absolute Error: ' + str(metrics['mae']))
        print('Mean Squared Error: ' + str(metrics['mse']))
        print('Root Mean Squared Error: ' + str(metrics['rmse']))
        print('Median Absolute Error: ' + str(metrics['med_abs_e']))
        print('R-squared: ' + str(metrics['r2']))
        print('R-squared (adj): ' + str(metrics['r2_adj']))
        print('Breusch-Pagan LM p-val: ' + metrics['bp_lm_p'])
        print('White LM p-val: ' + metrics['white_lm_p'])
    return metrics

    
def calulate_vif(data, numerical_cols):
    # https://www.statology.org/multiple-linear-regression-assumptions/
    # https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
    fxn_dataset = data[numerical_cols].copy()
    fxn_dataset = add_constant(fxn_dataset)
    vif = pd.Series([variance_inflation_factor(fxn_dataset.values, i) for i in range(fxn_dataset.shape[1])], index=fxn_dataset.columns)
    return vif

# =======================================================================================
# Statsmodels functions
# =======================================================================================
# Plot standardized residuals vs. predicted values and true values vs. predicted values
# Parameter filename_unique to be added to the end of the filename if saved
# Parameter lr_model must be a statsmodels linear regression model
# Can only save image if combining plots
# Returns heteroscedasticity metrics 'het_metrics'
def sm_lr_model_results(lr_model, y, y_pred, combine_plots=False, plot_title='', save_img=False, filename_unique=None): 
    # Format text box for relevant metric of each plot
    box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.9}
    
    if combine_plots:
        # Create figure, gridspec, list of axes/subplots mapped to gridspec location
        fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=(10, 5))
    
    # =============================
    # Plot studentized residuals vs. predicted values
    # =============================
    standardized_residuals = pd.DataFrame(lr_model.get_influence().resid_studentized_internal)
    
    # Calculate heteroscedasticity metrics
    bp_test = het_breuschpagan(lr_model.resid, lr_model.model.exog)
    white_test = het_white(lr_model.resid, lr_model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    bp_test_results = dict(zip(labels, bp_test)) 
    white_test_results = dict(zip(labels, white_test))
    bp_lm_p_value = '{:0.2e}'.format(bp_test_results['LM-Test p-value'])
    white_lm_p_value = '{:0.2e}'.format(white_test_results['LM-Test p-value'])
    
    if not combine_plots:
        plt.scatter(y_pred, standardized_residuals)
        ax1 = plt.gca()
    else:
        ax1 = ax_array_flat[0]
        ax1.scatter(y_pred, standardized_residuals) 
        
    ax1.axhline(y=0, color='darkblue', linestyle='--')
    ax1.set_ylabel('Studentized Residuals')
    ax1.set_xlabel('Predicted Values')
    ax1.set_title('Scale-Location')
    textbox_text = f'BP: {bp_lm_p_value} \n White: {white_lm_p_value}' 
    ax1.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right')   
    if not combine_plots: plt.show()

    # =============================
    # Plot True Values vs. Predicted Values
    # =============================
    if not combine_plots:
        plt.scatter(y, y_pred)
        ax2 = plt.gca()
    else:
        ax2 = ax_array_flat[1]
        ax2.scatter(y, y_pred)
        
    largest_num = max(max(y), max(y_pred))
    smallest_num = min(min(y), min(y_pred))
    plot_limits = [smallest_num - (0.02*largest_num), largest_num + (0.02*largest_num)]
    ax2.set_xlim(plot_limits)
    ax2.set_ylim(plot_limits)
    ax2.plot([0, 1], [0, 1], color='darkblue', linestyle='--', transform=ax2.transAxes)
    ax2.set_title('True Values vs. Predicted Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_xlabel('True Values')
    textbox_text = r'$R^2$: %0.3f' %lr_model.rsquared
    ax2.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',)    
    if not combine_plots: plt.show()
    
    if combine_plots:
        fig.suptitle('LR Model Performance (' + plot_title + ')', fontsize=24)
        fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
        if save_img:
            save_filename = 'sm_lr_results_' + filename_unique
            dh.save_image(save_filename, models_output_dir)
        plt.show()
    
    het_metrics = dict(zip(['BP', 'White'], [bp_test_results, white_test_results]))
    return het_metrics

# Subgroup plots by smoking and obesity
# Parameter lr_model must be a statsmodels linear regression model
# Parameter plot_title will be added below the actual title in parentheses
# Parameter filename_unique to be added to the end of the filename if saved
# Returns heteroscedasticity metrics 'het_metrics'
def sm_lr_model_results_subgrouped(lr_model, X_data, y, y_pred, plot_title, save_img=False, filename_unique=None):
    # Organize relevant data
    standardized_residuals = pd.DataFrame(lr_model.get_influence().resid_studentized_internal, columns=['stand_resid'])
    #standardized_residuals = pd.DataFrame(lr_model.resid, columns=['stand_resid']) # to plot absolute residuals rather than studentized
    y_pred_series = pd.Series(y_pred, name='y_pred')
    y_series = pd.Series(y, name='y')
    relevant_data = pd.concat([X_data[['bmi_>=_30_yes', 'smoker_yes']], y_series, y_pred_series, standardized_residuals], axis=1)

    smoker_data = relevant_data[relevant_data['smoker_yes']==1]
    nonsmoker_data = relevant_data[relevant_data['smoker_yes']==0]
    smoker_obese_data = smoker_data[smoker_data['bmi_>=_30_yes']==1]
    smoker_nonobese_data = smoker_data[smoker_data['bmi_>=_30_yes']==0]
    nonsmoker_obese_data = nonsmoker_data[nonsmoker_data['bmi_>=_30_yes']==1]
    nonsmoker_nonobese_data = nonsmoker_data[nonsmoker_data['bmi_>=_30_yes']==0]
    
    # Quantify Heteroscedasticity using White test and Breusch-Pagan test
    bp_test = het_breuschpagan(lr_model.resid, lr_model.model.exog)
    white_test = het_white(lr_model.resid, lr_model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    bp_test_results = dict(zip(labels, bp_test))
    white_test_results = dict(zip(labels, white_test))
    bp_lm_p_value = '{:0.2e}'.format(bp_test_results['LM-Test p-value'])
    white_lm_p_value = '{:0.2e}'.format(white_test_results['LM-Test p-value'])
    
    # Format text box with relevant metric of each plot
    box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.8}
    
    # Create figure, gridspec, list of axes/subplots mapped to gridspec location
    fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=(12, 5))
        
    # =============================
    # Plot studentized residuals vs. predicted values
    # =============================
    ax1 = ax_array_flat[0]
    ax1.scatter(smoker_obese_data['y_pred'], smoker_obese_data['stand_resid'], alpha=0.5, label='obese smokers')
    ax1.scatter(smoker_nonobese_data['y_pred'], smoker_nonobese_data['stand_resid'], alpha=0.5, label='nonobese smokers')
    ax1.scatter(nonsmoker_obese_data['y_pred'], nonsmoker_obese_data['stand_resid'], alpha=0.5, label='obese nonsmokers')
    ax1.scatter(nonsmoker_nonobese_data['y_pred'], nonsmoker_nonobese_data['stand_resid'], alpha=0.5, label='nonobese nonsmokers')
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_ylabel('Studentized Residuals')
    ax1.set_xlabel('Predicted Values')
    ax1.set_title('Scale-Location')
    textbox_text = f'BP: {bp_lm_p_value} \n White: {white_lm_p_value}' 
    ax1.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right')  
    
    # =============================
    # True Values vs. Predicted Values 
    # =============================
    ax2 = ax_array_flat[1]
    ax2.scatter(smoker_obese_data['y'], smoker_obese_data['y_pred'], alpha=0.5, label='obese smokers')
    ax2.scatter(smoker_nonobese_data['y'], smoker_nonobese_data['y_pred'], alpha=0.5, label='nonobese smokers')
    ax2.scatter(nonsmoker_obese_data['y'], nonsmoker_obese_data['y_pred'], alpha=0.5, label='obese nonsmokers')
    ax2.scatter(nonsmoker_nonobese_data['y'], nonsmoker_nonobese_data['y_pred'], alpha=0.5, label='nonobese nonsmokers')
    largest_num = max(max(relevant_data['y']), max(relevant_data['y_pred']))
    smallest_num = min(min(relevant_data['y']), min(relevant_data['y_pred']))
    
    plot_limits = [smallest_num - (0.02*largest_num), largest_num + (0.02*largest_num)]
    ax2.set_xlim(plot_limits)
    ax2.set_ylim(plot_limits)
    ax2.plot([0, 1], [0, 1], color='darkblue', linestyle='--', transform=ax2.transAxes)
    
    #ax2.plot([smallest_num, largest_num], [smallest_num, largest_num], color='darkblue', linestyle='--')
    ax2.set_title('True Values vs. Predicted Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_xlabel('True Values')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Subgroup')
    textbox_text = r'$R^2$: %0.3f' %lr_model.rsquared
    ax2.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right') 
    
    # Format and save figure
    fig.suptitle('LR Model Performance (' + plot_title + ')', fontsize=24)
    fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
    if save_img:
        save_filename = 'sm_lr_results_' + filename_unique
        dh.save_image(save_filename, models_output_dir)
    plt.show()

    het_metrics = dict(zip(['BP', 'White'], [bp_test_results, white_test_results]))
    return het_metrics

# Combine statsmodels linear regression model creation, fitting, and returning results    
def fit_lr_model_results(fxn_X, fxn_y, plot_title, combine_plots=True, save_img=False, filename_unique=None):
    # Fit model
    fxn_lin_reg = sm.OLS(fxn_y, fxn_X).fit()
    
    # Predict target
    fxn_y_pred = fxn_lin_reg.predict(fxn_X) 
    
    # Plot results, get heteroscedasticity metrics
    het_results = sm_lr_model_results(fxn_lin_reg, fxn_y, fxn_y_pred, combine_plots=combine_plots, plot_title=plot_title, save_img=save_img, filename_unique=filename_unique)
  
    return fxn_lin_reg, fxn_y_pred, het_results

# Combine statsmodels linear regression model creation, fitting, and returning results    
def fit_lr_model_results_subgrouped(fxn_X, fxn_y, plot_title, save_img=False, filename_unique=None):
    # Fit model
    fxn_lin_reg = sm.OLS(fxn_y, fxn_X).fit()
    
    # Predict target
    fxn_y_pred = fxn_lin_reg.predict(fxn_X) 
    
    # Plot results subgrouped, get heteroscedasticity metrics
    het_results = sm_lr_model_results_subgrouped(fxn_lin_reg, fxn_X, fxn_y, fxn_y_pred, plot_title, save_img=save_img, filename_unique=filename_unique)
    
    return fxn_lin_reg, fxn_y_pred, het_results

# Convert statsmodels summary() output to pandas DataFrame
def sm_results_to_df(summary):
    # Statsmodels summary() returns list of tables, relevant data in second table as list of lists
    summary_data = summary.tables[1].data
    
    # Convert list of lists to dataframe
    return_df = pd.DataFrame(summary_data)
    
    # Extract column names, from first row. Convert to list to remove confusion with Series name
    col_names = return_df.iloc[0][1:].tolist()
    
    # Remove first row
    return_df.drop(0, axis=0, inplace=True)
    
    # Extract row names, from first column. Convert to list to remove confusion with Series name
    row_names = return_df[0].tolist()
    
    # Remove first column
    return_df.drop(0, axis=1, inplace=True)
    
    # Set column and row names
    return_df.columns = col_names
    return_df.index = row_names
    
    return return_df


# ====================================================================================================================
# Implement statsmodels package to test multiple linear regression model assumptions
# ====================================================================================================================
# https://datatofish.com/multiple-linear-regression-python/

# =======================================================================================
# Test for multicollinearity
# =======================================================================================
# Calculate VIF
vif = calulate_vif(dataset, numerical_cols).to_frame()

# All very close to 1, no multicollinearity. (Greater than 5-10 indicates multicollinearity)
# Row indeces normally not included in table image, so I inserted them as the first column
vif.insert(0, 'Feature', vif.index)

# Rename VIF columns
vif.rename(columns={0:'VIF'}, inplace=True)

# Round to 2 decimal places
vif = np.round(vif, decimals=2)

# Convert VIF values to string to avoid render_mpl_table() removing trailing zeroes
vif['VIF'] = vif['VIF'].map('{:,.2f}'.format)

# Create table image
dh.render_mpl_table(vif)
#dh.save_image('vif_table', models_output_dir, dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()


# =======================================================================================
# Quantify Heteroscedasticity
# =======================================================================================

# Quantify Heteroscedasticity using White test and Breusch-Pagan test
# https://medium.com/@remycanario17/tests-for-heteroskedasticity-in-python-208a0fdb04ab
# https://www.statology.org/breusch-pagan-test/

# Before feature engineering (below), both had a p-value <<< 0.05, indicating presence of heteroscedasticity. After feature
# engineering, both were well above 0.05. 

# I tried log transforming target before and after performing feature engineering. Did not work 
# either time

# =======================================================================================
# Test for linear relationships between predictors and target variables
# Involves plenty of feature engineering 
# =======================================================================================
# Separate target from predictors
y = dataset['charges']
X = dataset.drop(['charges'], axis=1)

# This is to test for linear model assumptions in entire data set, will not perform train/test split
sm_processed_X = manual_preprocess_sm(X)

# Fit linear regression model
sm_lin_reg_0 = sm.OLS(y, sm_processed_X).fit()

# Make predictions
sm_y_pred_0 = sm_lin_reg_0.predict(sm_processed_X)

# Plot model
title_0 = 'Original'
model_name_0 = 'original'
file_name_0 = '0_' + model_name_0
het_metrics_0 = sm_lr_model_results(sm_lin_reg_0, y, sm_y_pred_0, combine_plots=True, plot_title=title_0, save_img=False, filename_unique=file_name_0)

# Organize model performance metrics
summary_df_0 = sm_results_to_df(sm_lin_reg_0.summary())
coeff_0 = pd.Series(summary_df_0['coef'], name=model_name_0)
sm_lr_results_0 = pd.Series(evaluate_model_sm(y, sm_y_pred_0, sm_lin_reg_0), name=model_name_0)

# ==========================================================
# Based on EDA, created dichotomous feature 'bmi_>=_30'
# ==========================================================
# Create new feature
new_X_1 = X.copy()
new_X_1['bmi_>=_30'] = new_X_1['bmi'] >= 30
bmi_dict = {False:'no', True:'yes'}
new_X_1['bmi_>=_30'] = new_X_1['bmi_>=_30'].map(bmi_dict)

# Add the new feature to the columns lists (necessary for preprocessing)
#categorical_cols.append('bmi_>=_30')
#cat_ord_cols.append('bmi_>=_30')

# Preprocess with new feature
new_X_1 = manual_preprocess_sm(new_X_1)

# Plot model without subgrouping
title_1 = 'w [bmi>=30] feature'
model_name_1 = '[bmi_>=_30]'
file_name_1_0 = '1_bmi_30_feature'
sm_lin_reg_1_0, sm_y_pred_1_0, het_results_1_0 = fit_lr_model_results(new_X_1, y, title_1, combine_plots=True, save_img=False, filename_unique=file_name_1_0)

# Plot model with subgrouping
file_name_1 = '1_bmi_30_feature_grouped'
sm_lin_reg_1, sm_y_pred_1, het_results_1 = fit_lr_model_results_subgrouped(new_X_1, y, title_1, save_img=False, filename_unique=file_name_1)

# Organize model performance metrics
summary_df_1 = sm_results_to_df(sm_lin_reg_1.summary())
coeff_1 = pd.Series(summary_df_1['coef'], name=model_name_1)
sm_lr_results_1 = pd.Series(evaluate_model_sm(y, sm_y_pred_1, sm_lin_reg_1), name=model_name_1)

# Keep track of model performance for comparison later
coeff_df = pd.concat([coeff_0, coeff_1], axis=1)
sm_results_df = pd.concat([sm_lr_results_0, sm_lr_results_1], axis=1)


# ==========================================================
# BMI vs. Charges
# Smokers had a strong linear relationship between BMI and charges, nonsmokers had basically no linear relationship
# Will engineer new feature [smoker*bmi] 
# ==========================================================
# Create new feature
new_X_2 = new_X_1.copy()
new_X_2['bmi*smoker'] = new_X_2['smoker_yes'] * new_X_2['bmi']

# Plot model
title_2 = 'w [bmi*smoker] feature'
model_name_2 = '[bmi*smoker]'
file_name_2 = '2_smoke_bmi_feature'
sm_lin_reg_2, sm_y_pred_2, het_results_2 = fit_lr_model_results_subgrouped(new_X_2, y, title_2, save_img=False, filename_unique=file_name_2)

# Organize model performance metrics
summary_df_2 = sm_results_to_df(sm_lin_reg_2.summary())
coeff_2 = pd.Series(summary_df_2['coef'], name=model_name_2)
sm_lr_results_2 = pd.Series(evaluate_model_sm(y, sm_y_pred_2, sm_lin_reg_2), name=model_name_2)

# Keep track of model performance for comparison later
coeff_df = pd.concat([coeff_df, coeff_2], axis=1)
sm_results_df = pd.concat([sm_results_df, sm_lr_results_2], axis=1)

# Tried removing original 'bmi' feature, slightly worsened model performance

# ==========================================================
# Age vs. Charges
# Explored new feature incorporating relationship between between presence of obesity, smoking, and age
# ==========================================================
# Create new feature
new_X_3 = new_X_2.copy()
new_X_3['smoker*obese'] = new_X_3['smoker_yes'] * new_X_3['bmi_>=_30_yes']

# Plot model
title_3 = 'w [smoker*obese] feature'
model_name_3 = '[smoker*obese]'
file_name_3 = '3_smoke_ob_feature'
sm_lin_reg_3, sm_y_pred_3, het_results_3 = fit_lr_model_results_subgrouped(new_X_3, y, title_3, save_img=False, filename_unique=file_name_3)

# Organize model performance metrics
summary_df_3 = sm_results_to_df(sm_lin_reg_3.summary())
coeff_3 = pd.Series(summary_df_3['coef'], name=model_name_3)
sm_lr_results_3 = pd.Series(evaluate_model_sm(y, sm_y_pred_3, sm_lin_reg_3), name=model_name_3)

# Keep track of model performance for comparison later
coeff_df = pd.concat([coeff_df, coeff_3], axis=1)
sm_results_df = pd.concat([sm_results_df, sm_lr_results_3], axis=1)

# ==========================================================
# Age vs. Charges: accounting for curvilinear relationship between age and charges
# ==========================================================
# Create new feature
new_X_4 = new_X_3.copy()
# Age has already been scaled around 0 and squaring the values will make all the negative numbers positive
# So I will take the original ages, square them, then scale.
orig_ages = dataset['age'].to_frame()
squared_ages = np.power(orig_ages, 2)
scaled_sq_ages = pd.DataFrame(StandardScaler().fit_transform(squared_ages), columns=['age^2'])
new_X_4['age^2'] = scaled_sq_ages

# Plot model
title_4 = 'w [age^2] feature'
model_name_4 = '[age^2]'
file_name_4 = '4_age_sq_feature'
sm_lin_reg_4, sm_y_pred_4, het_results_4 = fit_lr_model_results_subgrouped(new_X_4, y, title_4, save_img=False, filename_unique=file_name_4)

# Organize model performance metrics
summary_df_4 = sm_results_to_df(sm_lin_reg_4.summary())
coeff_4 = pd.Series(summary_df_4['coef'], name=model_name_4)
sm_lr_results_4 = pd.Series(evaluate_model_sm(y, sm_y_pred_4, sm_lin_reg_4), name=model_name_4)

# Keep track of model performance for comparison later
coeff_df = pd.concat([coeff_df, coeff_4], axis=1)
sm_results_df = pd.concat([sm_results_df, sm_lr_results_4], axis=1)

# Removing original 'age' feature didn't change model as it already had a relatively small coefficient (-68) once
# age^2 was added. I also tried not scaling the age^2 feature. This didn't change model at all. Literally same 
# coefficients other than it's own being significantly smaller (3000 -> 3)


# Plot distribution of charges
sns.kdeplot(x=y, shade=True)
plt.hist(y, bins=50, density=True, label='charges', alpha=0.5)

# Plot distribution of predicted charges
sns.kdeplot(x=sm_y_pred_4, shade=True)
plt.hist(sm_y_pred_4, bins=50, density=True, label='pred charges', alpha=0.5)

# Plot y and predicted y histograms
plt.hist(y, bins=50, density=True, label='charges', alpha=0.5)
plt.hist(sm_y_pred_4, bins=50, density=True, label='pred charges', alpha=0.5)
plt.legend()
# =======================================================================================
# Compare coefficients before and after new features
# =======================================================================================
# =============================
# Drop variables whose coefficients don't change much
# =============================
# Variables that don't change much: children, sex_male, all regions, const.
# All features
coeff_df_new = coeff_df.apply(pd.to_numeric)

# Drop variables
drop_var = ['region_northwest', 'region_southeast', 'region_southwest', 'const']
coeff_df_new = coeff_df_new.drop(drop_var, axis=0)

# Replace NaN with 0
coeff_df_new = coeff_df_new.replace(np.nan, 0)

# Separate new and old features
orig_features_df = coeff_df_new.iloc[0:5]
new_features_df = coeff_df_new.iloc[5:len(coeff_df_new.index)]

# Plot combined
fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=3, num_cols=1, figsize=(9, 13))

smoker_df = orig_features_df.loc['smoker_yes'].to_frame().T
ax1 = ax_array_flat[0]
for feature in smoker_df.index:
    ax1.plot(smoker_df.columns, smoker_df.loc[feature].to_list(), label=feature, linewidth=3)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
plt.setp(ax1.get_xticklabels(), rotation=20, horizontalalignment='right')
ax1.set_ylabel('Coefficient', fontsize=16)
ax1.grid()

orig_features_no_smoker = orig_features_df.drop(['smoker_yes'], axis=0)
ax2 = ax_array_flat[1]
for feature in orig_features_no_smoker.index:
    ax2.plot(orig_features_no_smoker.columns, orig_features_no_smoker.loc[feature].to_list(), label=feature, linewidth=3)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
plt.setp(ax2.get_xticklabels(), rotation=20, horizontalalignment='right')
ax2.set_ylabel('Coefficient', fontsize=16)
ax2.grid()

ax3 = ax_array_flat[2]
for feature in new_features_df.index:
    ax3.plot(new_features_df.columns, new_features_df.loc[feature].to_list(), label=feature, linewidth=3)
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
plt.setp(ax3.get_xticklabels(), rotation=20, horizontalalignment='right')
ax3.set_xlabel('Additional Features', fontsize=16)
ax3.set_ylabel('Coefficient', fontsize=16)
ax3.grid()

fig.suptitle('Feature coeff w/ each additional feature', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'coeff_vert_5_no_outliers_2'
#dh.save_image(save_filename, models_output_dir)
plt.show()


# =======================================================================================
# Compare performance before and after new features
# =======================================================================================

sm_results_df = sm_results_df.apply(pd.to_numeric)

# Separate out metrics by scale
all__error_mets = sm_results_df.loc[['max_e', 'rmse', 'mae', 'med_abs_e']]
max_e_df = sm_results_df.loc['max_e'].to_frame().T
error_metrics = sm_results_df.loc[['rmse', 'mae', 'med_abs_e']]
r_metrics = sm_results_df.loc[['r2', 'r2_adj']]
het_stats = sm_results_df.loc[['bp_lm_p', 'white_lm_p']]

# Plot combined
fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=2, num_cols=2, figsize=(12, 7))

ax1 = ax_array_flat[0]
df_to_plot = max_e_df
for metric in df_to_plot.index:
    ax1.plot(df_to_plot.columns, df_to_plot.loc[metric].to_list(), label=metric, linewidth=3)
ax1.legend(loc='upper right', borderaxespad=0.5, title='Metric')
plt.setp(ax1.get_xticklabels(), rotation=20, horizontalalignment='right')
ax1.grid()

ax2 = ax_array_flat[1]
df_to_plot = error_metrics
for metric in df_to_plot.index:
    ax2.plot(df_to_plot.columns, df_to_plot.loc[metric].to_list(), label=metric, linewidth=3)
ax2.legend(loc='upper right', borderaxespad=0.5, title='Metric')
plt.setp(ax2.get_xticklabels(), rotation=20, horizontalalignment='right')
ax2.grid()

ax3 = ax_array_flat[2]
df_to_plot = r_metrics
for metric in df_to_plot.index:
    ax3.plot(df_to_plot.columns, df_to_plot.loc[metric].to_list(), label=metric, linewidth=3)
ax3.legend(loc='upper left', borderaxespad=0.5, title='Metric')
plt.setp(ax3.get_xticklabels(), rotation=20, horizontalalignment='right')
ax3.grid()

ax4 = ax_array_flat[3]
df_to_plot = het_stats
for metric in df_to_plot.index:
    ax4.plot(df_to_plot.columns, df_to_plot.loc[metric].to_list(), label=metric, linewidth=3)
ax4.legend(loc='upper left', borderaxespad=0.5, title='Metric')
plt.setp(ax4.get_xticklabels(), rotation=20, horizontalalignment='right')
ax4.grid()

fig.suptitle('LR Performance w/ each additional feature', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'performance_no_outliers_2'
#dh.save_image(save_filename, models_output_dir)
plt.show()

# =======================================================================================
# Outlier Analysis
# =======================================================================================
# https://towardsdatascience.com/linear-regression-model-with-python-481c89f0f05b
# Observation has a high influence if the Cook's distance is greater than 4/(N-k-1)
# N = number of observations, k = number of predictors, yellow horizontal line in the plot

# ==========================================================
# Identify Outliers
# ==========================================================
inf = influence(sm_lin_reg_4)
(cooks, d) = inf.cooks_distance
cooks_cutoff = 4 / (len(cooks) - (new_X_4.shape[1] - 1) - 1) # 0.00301

outlier_df = new_X_4.copy()
outlier_df['cooks'] = cooks
outlier_df['outlier'] = outlier_df['cooks'] > cooks_cutoff
outlier_dict = {False:'no', True:'yes'}
outlier_df['outlier'] = outlier_df['outlier'].map(outlier_dict)

num_outliers = outlier_df[outlier_df['outlier'] == 'yes'].shape[0] # 90
perc_outliers = num_outliers / outlier_df.shape[0] # 0.0672
outlier_df['true_values'] = y
outlier_df['y_pred'] = sm_y_pred_4
outlier_df['stud_resid'] = sm_lin_reg_4.get_influence().resid_studentized_internal

# Visualize Cook's Distances
plt.title("Cook's Distance Plot")
plt.stem(range(len(cooks)), cooks, markerfmt=",")
plt.plot([0, len(cooks)], [cooks_cutoff, cooks_cutoff], color='darkblue', linestyle='--', label='4 / (N-k-1)')
plt.xlabel("Observation")
plt.ylabel("Cook's Distance")
plt.legend(title="Cook's Distance Cutoff")
#dh.save_image('cooks_dist_plot', models_output_dir)
plt.show()

# ==========================================================
# Plot with respect to model results
# ==========================================================
outlier_data = outlier_df[outlier_df['outlier']=='yes']
nonoutlier_data = outlier_df[outlier_df['outlier']=='no']

# Stand Resid vs. Stud Residuals
plt.scatter(outlier_data['y_pred'], outlier_data['stud_resid'], alpha=0.7, label='Outliers')
plt.scatter(nonoutlier_data['y_pred'], nonoutlier_data['stud_resid'], alpha=0.7)
plt.ylabel('Standardized Residuals')
plt.xlabel('Predicted Values')
plt.title('Standardized Residuals vs. Predicted Values')
plt.legend()
#dh.save_image('outliers_pred_vs_resid', models_output_dir)
plt.show()

# ==========================================================
# Plot with respect to original data
# ==========================================================
# Make dataset of original data with new outlier information
orig_data_w_outlier = dataset.copy()
orig_data_w_outlier['outlier'] = outlier_df['outlier']

orig_data_w_outlier['bmi_>=_30'] = orig_data_w_outlier['bmi'] >= 30
bmi_dict = {False:'no', True:'yes'}
orig_data_w_outlier['bmi_>=_30'] = orig_data_w_outlier['bmi_>=_30'].map(bmi_dict)

# =============================
# Distribution of charges in outliers vs. not
# =============================
# Make DataFrame of just charges and outlier category
y_outlier_df = pd.DataFrame({'charges':y, 'outlier':outlier_df['outlier']})

# Split outlier and nonoutlier data
y_outlier_data = y_outlier_df[y_outlier_df['outlier']=='yes']
y_nonoutlier_data = y_outlier_df[y_outlier_df['outlier']=='no']

# Compare charges distribution with outlier and nonoutlier data
sns.kdeplot(x=y_outlier_data['charges'], shade=True, alpha=0.5, label='outlier data')
sns.kdeplot(x=y_nonoutlier_data['charges'], shade=True, alpha=0.5, label='nonoutlier data')
plt.hist(y_outlier_data['charges'], bins=50, density=True, alpha=0.5)
plt.hist(y_nonoutlier_data['charges'], bins=50, density=True, alpha=0.5)
plt.title("Distribution of Charges")
plt.legend()
#dh.save_image('outliers_dist_charges', models_output_dir)


# =============================
# Scatterplots of numerical variables
# =============================
# Age vs. charges
sns.lmplot(x='age', y='charges', hue="outlier", data=orig_data_w_outlier, ci=None, line_kws={'alpha':0}, legend=False) # LM plot just makes it easier to color by outlier
plt.title("Age vs. Charges")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title="Cook's Outlier")
#dh.save_image('outliers_age_v_charges', models_output_dir)

# Nonsmoker age vs. charges
nonsmoker_outlier_df = orig_data_w_outlier[orig_data_w_outlier['smoker']=='no']
sns.lmplot(x='age', y='charges', hue="outlier", data=nonsmoker_outlier_df, ci=None, line_kws={'alpha':0}, legend=False) # LM plot just makes it easier to color by outlier
plt.title("Age vs. Charges in nonsmokers")
#dh.save_image('outliers_age_v_charges_nonsmoker', models_output_dir)

# Obese smoker age vs. charges
ob_smoker_outlier_df = orig_data_w_outlier[(orig_data_w_outlier['smoker']=='yes') & (orig_data_w_outlier['bmi_>=_30']=='yes')]
sns.lmplot(x='age', y='charges', hue="outlier", data=ob_smoker_outlier_df, ci=None, line_kws={'alpha':0}, legend=False)
plt.title("Age vs. Charges in obese smokers")
#dh.save_image('outliers_age_v_charges_ob_smoker', models_output_dir)

# Nonobese smoker age vs. charges
nonob_smoker_outlier_df = orig_data_w_outlier[(orig_data_w_outlier['smoker']=='yes') & (orig_data_w_outlier['bmi_>=_30']=='no')]
sns.lmplot(x='age', y='charges', hue="outlier", data=nonob_smoker_outlier_df, ci=None, line_kws={'alpha':0}, legend=False)
plt.title("Age vs. Charges in obese smokers")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title="Cook's Outlier")
#dh.save_image('outliers_age_v_charges_nonob_smoker', models_output_dir)

# In the next two graphs, the outliers are scattered around, there is no obvious grouping
# Smoker bmi vs. charges
smoker_outlier_df = orig_data_w_outlier[orig_data_w_outlier['smoker'] >= 'yes']
sns.lmplot(x='bmi', y='charges', hue="outlier", data=smoker_outlier_df)#, ci=None, line_kws={'alpha':0})
plt.plot()

# Children vs. charges
sns.lmplot(x='children', y='charges', hue="outlier", data=orig_data_w_outlier)#, ci=None, line_kws={'alpha':0})
plt.plot()

# =============================
# Boxplots of categorical variables
# =============================
# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=1, num_cols=5, figsize=(18, 4))

target_col = 'outlier'
i = 0
for col in cat_ord_cols:
    df_grouped = dh.dataframe_percentages(orig_data_w_outlier, target_col, col)
    ax = ax_array_flat[i]
    sns.barplot(x=df_grouped[col], y=df_grouped[(df_grouped[target_col]=='yes')]['percent_of_cat_var'], ax=ax)
    ax.axline(xy1=(0, (perc_outliers*100)), slope=0, color='darkblue', linestyle='--', label='Dataset % Outliers')
    ax.set_title('Percent Outlier by ' + format_col(col))
    ax.set_xlabel(format_col(col))
    ax.set_ylabel('Percent Outlier')
    if col=='region':
        plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right')
    if i==(len(cat_ord_cols)-1):
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)#, title=target_col)
    i+=1
fig.suptitle('Percent Outliers in Each Subcategory', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
save_filename = 'perc_outlier_subcat'
#dh.save_image(save_filename, models_output_dir)

# Subcategory of 4 children has 15% outliers whereas basically all other subcategories range between 5-8%
# You can also see in 'Categorical Variable Relationships with Target' figure that samples with 4 kids 
# have a different distribution than the rest
# However, that only represents 4 outlieres out of 90, so unsurprisingly, further exploration didn't lead anywhere


# Inverse of the above graph
# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=1, num_cols=5, figsize=(18, 4))

target_col = 'outlier'
i = 0
for col in cat_ord_cols:
    df_grouped = dh.dataframe_percentages(orig_data_w_outlier, target_col, col)
    ax = ax_array_flat[i]
    sns.barplot(x=df_grouped[target_col], y=df_grouped['perc_of_target_cat'], hue=df_grouped[col], ax=ax)
    ax.set_title('Comp. by ' + format_col(col) + ' Subcategory')
    ax.set_xlabel(format_col(col))
    ax.set_ylabel('Percent Outlier')
    ax.set_xlabel('Outlier')
    ax.set_ylabel('Percent Subcategory')
    ax.legend(title=format_col(col), framealpha=0.5)
    i+=1
    
fig.suptitle('Outlier Composition by Subcategory', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
save_filename = 'perc_subcat_by_outlier'
#dh.save_image(save_filename, models_output_dir)


# ==========================================================
# Remove outliers and compare models
# ==========================================================
merge_newX_y = pd.concat([new_X_4, y, outlier_df['outlier']], axis=1)
no_outliers_df = merge_newX_y[merge_newX_y['outlier']=='no']

# Separate target from predictors
no_outliers_y = no_outliers_df['charges']
no_outliers_X = no_outliers_df.drop(['charges', 'outlier'], axis=1)

# Plot model
title_5 = 'removed outliers'
model_name_5 = 'no_out'
file_name_5 = '5_no_outliers'
sm_lin_reg_5, sm_y_pred_5, het_results_5 = fit_lr_model_results_subgrouped(no_outliers_X, no_outliers_y, title_5, save_img=False, filename_unique=file_name_5)

# Organize model performance metrics
summary_df_5 = sm_results_to_df(sm_lin_reg_5.summary())
coeff_5 = pd.Series(summary_df_5['coef'], name=model_name_5)
sm_lr_results_5 = pd.Series(evaluate_model_sm(no_outliers_y, sm_y_pred_5, sm_lin_reg_5), name=model_name_5)

# Keep track of model performance for comparison later
coeff_df = pd.concat([coeff_df, coeff_5], axis=1)
sm_results_df = pd.concat([sm_results_df, sm_lr_results_5], axis=1)

# (I just ran the code to plot coefficients and performance metrics again after updating the coeff and results dataframes)

# =======================================================================================
# Outlier Analysis Round 2
# =======================================================================================
# ==========================================================
# Identify Outliers Again
# ==========================================================
inf_5 = influence(sm_lin_reg_5)
(cooks_5, d) = inf_5.cooks_distance
cooks_cutoff_5 = 4 / (len(cooks_5) - (no_outliers_X.shape[1] - 1) - 1) # 0.00323

outlier_df_5 = no_outliers_X.copy()
outlier_df_5['cooks'] = cooks_5
outlier_df_5['outlier'] = outlier_df_5['cooks'] > cooks_cutoff_5
outlier_dict = {False:'no', True:'yes'}
outlier_df_5['outlier'] = outlier_df_5['outlier'].map(outlier_dict)

num_outliers_5 = outlier_df_5[outlier_df_5['outlier'] == 'yes'].shape[0] # 29
perc_outliers_5 = num_outliers_5 / outlier_df_5.shape[0] # 0.0232
outlier_df_5['true_values'] = y
outlier_df_5['y_pred'] = sm_y_pred_5
outlier_df_5['stud_resid'] = sm_lin_reg_5.get_influence().resid_studentized_internal

# Visualiz Cook's Distances
plt.title("Cook's Distance Plot (#2)")
plt.stem(range(len(cooks_5)), cooks_5, markerfmt=",")
plt.plot([0, len(cooks_5)], [cooks_cutoff_5, cooks_cutoff_5], color='darkblue', linestyle='--', label='4 / (N-k-1)')
plt.xlabel("Observation")
plt.ylabel("Cook's Distance")
plt.legend(title="Cook's Distance Cutoff")
#dh.save_image('cooks_dist_plot_2', models_output_dir)
plt.show()

# ==========================================================
# Plot with respect to model results again
# ==========================================================
outlier_data_5 = outlier_df_5[outlier_df_5['outlier']=='yes']
nonoutlier_data_5 = outlier_df_5[outlier_df_5['outlier']=='no']

# Stand Resid vs. Stud Residuals
plt.scatter(outlier_data_5['y_pred'], outlier_data_5['stud_resid'], alpha=0.7, label='Outliers')
plt.scatter(nonoutlier_data_5['y_pred'], nonoutlier_data_5['stud_resid'], alpha=0.7)
plt.ylabel('Studentized Residuals')
plt.xlabel('Predicted Values')
plt.title('Studentized Residuals vs. Predicted Values (#2)')
plt.legend()
#dh.save_image('outliers_pred_vs_resid_2', models_output_dir)
plt.show()

# ==========================================================
# Remove outliers and compare models again
# ==========================================================
merge_newX_y_2 = pd.concat([no_outliers_X, no_outliers_y, outlier_df_5['outlier']], axis=1)
no_outliers_df_2 = merge_newX_y_2[merge_newX_y_2['outlier']=='no']

# Separate target from predictors
no_outliers_y_2 = no_outliers_df_2['charges']
no_outliers_X_2 = no_outliers_df_2.drop(['charges', 'outlier'], axis=1)

# Plot model
title_6 = 'removed outliers x2'
model_name_6 = 'no_out_2'
file_name_6 = '6_no_outliers_2'
sm_lin_reg_6, sm_y_pred_6, het_results_6 = fit_lr_model_results_subgrouped(no_outliers_X_2, no_outliers_y_2, title_6, save_img=False, filename_unique=file_name_6)

# Organize model performance metrics
summary_df_6 = sm_results_to_df(sm_lin_reg_6.summary())
coeff_6 = pd.Series(summary_df_6['coef'], name=model_name_6)
sm_lr_results_6 = pd.Series(evaluate_model_sm(no_outliers_y_2, sm_y_pred_6, sm_lin_reg_6), name=model_name_6)

# Keep track of model performance for comparison later
coeff_df = pd.concat([coeff_df, coeff_6], axis=1)
sm_results_df = pd.concat([sm_results_df, sm_lr_results_6], axis=1)

# Coefficients with minimal change, performance shows perfect fit of model

# Plot distribution of charges with outliers removed a second time
sns.kdeplot(x=no_outliers_y_2, shade=True, alpha=0.5, label='nonoutlier data')

# ==========================================================
# Influence plots
# ==========================================================

# =============================
# Before removing Cook's outliers
# =============================
inf = influence(sm_lin_reg_4)
fig, ax = plt.subplots()
ax.axhline(-2.5, linestyle='-', color='C1')
ax.axhline(2.5, linestyle='-', color='C1')
ax.scatter(inf.hat_matrix_diag, inf.resid_studentized_internal, s=1000 * np.sqrt(inf.cooks_distance[0]), alpha=0.5)
ax.set_xlabel('H Leverage')
ax.set_ylabel('Studentized Residuals')
ax.set_title('Influence Plot Before Removing Outliers')
plt.tight_layout()
dh.save_image('influence_plot_1', models_output_dir)
plt.show()

# =============================
# After removing Cook's outliers first time
# =============================
inf = influence(sm_lin_reg_5)
fig, ax = plt.subplots()
ax.axhline(-2.5, linestyle='-', color='C1')
ax.axhline(2.5, linestyle='-', color='C1')
ax.scatter(inf.hat_matrix_diag, inf.resid_studentized_internal, s=1000 * np.sqrt(inf.cooks_distance[0]), alpha=0.5)
ax.set_xlabel('H Leverage')
ax.set_ylabel('Studentized Residuals')
ax.set_title('Influence Plot After Removing Outliers (first time)')
plt.tight_layout()
dh.save_image('influence_plot_2', models_output_dir)
plt.show()

# =============================
# After removing Cook's outliers second time
# =============================
inf = influence(sm_lin_reg_6)
fig, ax = plt.subplots()
ax.axhline(-2.5, linestyle='-', color='C1')
ax.axhline(2.5, linestyle='-', color='C1')
ax.scatter(inf.hat_matrix_diag, inf.resid_studentized_internal, s=1000 * np.sqrt(inf.cooks_distance[0]), alpha=0.5)
ax.set_xlabel('H Leverage')
ax.set_ylabel('Studentized Residuals')
ax.set_title('Influence Plot After Removing Outliers x2')
plt.tight_layout()
plt.show()


# =======================================================================================
# Test for normality of residuals
# =======================================================================================
# https://www.statology.org/multiple-linear-regression-assumptions/
# https://towardsdatascience.com/linear-regression-model-with-python-481c89f0f05b

# =============================
# A few helper functions
# =============================

# Q-Q plot (default is normal dist)
def my_qq(data, my_data_str='Residuals', dist_obj=stats.norm, fit_params=None, dist_str='Normal Dist', 
          ax=None, y=1, save_img=False, img_filename=None): 
    
    if not fit_params:
        # Fit my data to dist_obj and get fit parameters
        fit_params = dist_obj.fit(data)
    
    # Specify scipy distribution shape, location, and scale based on the parameters calculated from fit()
    loc = fit_params[-2]
    scale = fit_params[-1]
    shape_params = fit_params[:-2]
    
    # Q-Q Plot
    qqplot(data, line='45', fit=False, dist=dist_obj, loc=loc, scale=scale, distargs=shape_params, ax=ax)
    
    if not ax:
        ax = plt.gca()

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(f'Q-Q Plot {my_data_str} vs. {dist_str}', y=y)
    
    if save_img:
        dh.save_image(img_filename, models_output_dir)

    if not ax:
        plt.show()

# Plots a scipy distribution vs. histogram of my_data
def hist_vs_dist_plot(my_data, my_data_str='Residuals', dist_obj=stats.norm, fit_params=None, dist_str='Normal Dist', 
                      bins=200, ax=None, test_interp_str=None, save_img=False, img_filename=None):    
    
    if not fit_params:
        # Fit my data to dist_obj and get fit parameters
        fit_params = dist_obj.fit(my_data)
    
    # Specify scipy distribution shape, location, and scale based on the parameters calculated from fit()
    loc = fit_params[-2]
    scale = fit_params[-1]
    shape_params = fit_params[:-2]
    
    # Specify scipy distribution shape, location, and scale based on the parameters calculated from fit() above
    rv = dist_obj(*shape_params, loc, scale)
    
    # Use the distribution to create x values for the plot
    # ppf() is the inverse of cdf(). So if cdf(10) = 0.1, then ppf(0.1)=10
    # ppf(0.1) is the x-value at which 10% of the values are less than or equal to it
    x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
    
    if not ax:
        ax = plt.gca()
    
    # Plot distribution on top of histogram of charges in order to compare
    ax.hist(my_data, bins=bins, density=True, histtype='stepfilled', alpha=0.9, label=my_data_str)
    ax.plot(x, rv.pdf(x), 'r-', lw=2.5, alpha=1, label=dist_str)
    ax.set_title(f'{my_data_str} vs. {dist_str}', y=1.05)
    ax.set_xlabel(f'{my_data_str}')
    
    if test_interp_str:
        # Add normality test interpretation text
        box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.8}
        ax.text(1.05, 0.99, test_interp_str, bbox=box_style, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left') 
    
    
    ax.legend()
    
    if save_img:
        dh.save_image(img_filename, models_output_dir)

# Plot both qq and hist vs. dist plots in same figure
def plot_qq_hist_dist_combined(my_data, my_data_str='Residuals', dist_obj=stats.norm, dist_str='Normal Dist', 
                               bins=50, test_interp_str=None, fig_title=None, title_fontsize = 24, figsize=(10, 5), save_img=False, img_filename=None):
    
    # Create figure, gridspec, list of axes/subplots mapped to gridspec location
    fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=figsize)

    # Fit my data to dist_obj and get fit parameters
    fit_params = dist_obj.fit(my_data)
    
    # Plot Q-Q, add to figure
    my_qq(my_data, my_data_str=my_data_str, dist_obj=dist_obj, fit_params=fit_params, 
          ax=ax_array_flat[0], y=1.05) # Increase title space to match hist_vs_dist_plot()
    
    # Plot hist vs. dist, add to figure
    hist_vs_dist_plot(my_data, my_data_str=my_data_str, dist_obj=dist_obj, fit_params=fit_params, 
                      dist_str=dist_str, bins=bins, ax=ax_array_flat[1], test_interp_str=test_interp_str)
    
    # Figure title
    if fig_title:
        fig.suptitle(fig_title, fontsize=title_fontsize)
    else:
        fig.suptitle(f'{my_data_str} vs. {dist_str}', fontsize=title_fontsize)
    fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
    
    if save_img:
        dh.save_image(img_filename, models_output_dir)
    plt.show()



# ==========================================================
# Test for normality before removing Cook's outliers
# ==========================================================

resid4 = sm_lin_reg_4.resid_pearson
#resid4 = sm_lin_reg_4.resid
#resid4 = sm_lin_reg_4.get_influence().resid_studentized_internal

# =============================
# Statistical test for normality
# =============================
# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/

# Shapiro-Wilk test for normality
# not useful for big samples(>5000), since it tends to reject normality too often. Not an issue here
sw_stat4, sw_pval4 = stats.shapiro(sm_lin_reg_4.resid_pearson)

# D’Agostino’s K-squared test
dk_stat4, dk_pval4 = stats.normaltest(sm_lin_reg_4.resid_pearson)

# Anderson-Darling Normality Test
ad_stat4, ad_critvals4, ad_siglevels4 = stats.anderson(sm_lin_reg_4.resid_pearson, dist='norm')

# Chi-Square Normality Test
cs_stat4, cs_pval4 = stats.chisquare(sm_lin_reg_4.resid_pearson)

# Jarque–Bera test for Normality
js_stat4, js_pval4 = stats.jarque_bera(sm_lin_reg_4.resid_pearson)

# Kolmogorov-Smirnov test for Normality
ks_stat4, ks_pval4 = stats.kstest(sm_lin_reg_4.resid_pearson, 'norm')
# According to KS-table, for alpha of 0.05 and with n > 40, we want the test-statistic to be 
# less than 1.36/sqrt(n). This will give us 95% confidence that our data comes from the 
# given distribution
# https://oak.ucc.nau.edu/rh83/Statistics/ks1/
ks_stat_cutoff = 1.36 / np.sqrt(len(dataset['charges'])) # = (1.36 / sqrt(1338)) = 0.0372

# Lilliefors Test for Normality 
# Same as Kolmogorov-Smirnov?
lt_stat4, lt_pval4 = lilliefors(sm_lin_reg_4.resid_pearson, dist='norm')

# Function combining above normaly tests and interpreting results
normal_results4, normal_interpret4, nml_interpret_txt4 = dh.normality_tests(resid4)

# Q-Q plot and Residual Histogram vs. Normal
qqhist_filename_1 = 'qqhist1_orig'
plot_qq_hist_dist_combined(resid4, fig_title='Residual Distribution', test_interp_str=nml_interpret_txt4, 
                           save_img=False, img_filename=qqhist_filename_1)

# Plot y and predicted y histograms
plt.hist(y, bins=50, density=True, label='charges', alpha=0.5)
plt.hist(sm_y_pred_4, bins=50, density=True, label='pred charges', alpha=0.5)
plt.hist(resid4, bins=50, density=True, label='resid', alpha=0.5)
plt.legend()

# ==========================================================
# Test for normality after removing Cook's outliers first time
# ==========================================================
resid5 = sm_lin_reg_5.resid_pearson
#resid5 = sm_lin_reg_5.resid
#resid5 = sm_lin_reg_5.get_influence().resid_studentized_internal

# Function combining normality tests and interpreting results
normal_results5, normal_interpret5, nml_interpret_txt5 = dh.normality_tests(resid5)

# Q-Q plot and Residual Histogram vs. Normal
qqhist_filename_3 = 'qqhist3_outlier_1'
plot_qq_hist_dist_combined(resid5, fig_title='Residual Dist After Outlier Removal',  test_interp_str=nml_interpret_txt5, 
                           save_img=False, img_filename=qqhist_filename_3)


# ==========================================================
# Test for normality after removing Cook's outliers second time
# ==========================================================
resid6 = sm_lin_reg_6.resid_pearson
#resid6 = sm_lin_reg_6.resid
#resid6 = sm_lin_reg_6.get_influence().resid_studentized_internal

# Function combining normality tests and interpreting results
normal_results6, normal_interpret6, nml_interpret_txt6 = dh.normality_tests(resid6)

# Q-Q plot and Residual Histogram vs. Normal
qqhist_filename_4 = 'qqhist4_outlier_2'
plot_qq_hist_dist_combined(resid6, fig_title='Residual Dist After Outlier Removal x2', test_interp_str=nml_interpret_txt6,
                           save_img=False, img_filename=qqhist_filename_4)


# ==========================================================
# Try applying nonlinear transformations to variables rather than outlier removal
# Independent variables are already close to normal or uniform distributions
# ==========================================================

# Plot distribution of nontransformed y
sns.distplot(dataset['charges'], bins=50)
# plt.hist(y, bins=50, density=True, alpha=0.8)
# sns.kdeplot(x=y, shade=True)

# Plot nontransformed y and predicted y histograms
plt.hist(y, bins=50, density=True, label='charges transformed', alpha=0.5)
plt.hist(sm_y_pred_4, bins=50, density=True, label='pred charges', alpha=0.5)
plt.legend()

# New dataset
new_X_7 = new_X_4.copy()

# =============================
# Box-Cox transformation of y
# =============================
# Boxcox 'charges'
y_bc, lambd = stats.boxcox(y)

# Plot distribution of 'charges' boxcox transformed
sns.distplot(y_bc, bins=50)
plt.title('Charges Box-Cox Transformed', fontsize=20, y=1.04)
plt.xlabel('charges')
dh.save_image('charges_boxcox', models_output_dir, dpi=300, bbox_inches='tight', pad_inches=0.1)

# Plot model
title_7 = 'box-cox charges'
model_name_7 = 'normalized [charges]'
file_name_7 = '7_bc_charges'
sm_lin_reg_7, sm_y_pred_7, het_results_7 = fit_lr_model_results_subgrouped(new_X_7, y_bc, title_7, save_img=True, filename_unique=file_name_7)

# Function combining normality tests and interpreting results
normal_results7, normal_interpret7, nml_interpret_txt7 = dh.normality_tests(sm_lin_reg_7.resid_pearson)

# Q-Q plot and Residual Histogram vs. Normal
qqhist_filename_2 = 'qqhist2_boxcox_y'
plot_qq_hist_dist_combined(sm_lin_reg_7.resid_pearson, fig_title='Residual Dist After Normalizing Target', 
                           test_interp_str=nml_interpret_txt7, save_img=False, img_filename=qqhist_filename_2)

# Plot y and predicted y histograms
plt.hist(y_bc, bins=50, density=True, label='charges transformed', alpha=0.5)
plt.hist(sm_y_pred_7, bins=50, density=True, label='pred charges', alpha=0.5)
plt.legend()



# =============================
# Other transformation of y
# =============================


# ====================================================================================================================
# Back to sklearn models
# ====================================================================================================================

# ====================================================================================================================
# Split and preprocess data
# ====================================================================================================================
# Separate target from predictors
y = dataset['charges']
X = dataset.drop(['charges'], axis=1)

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# Preprocess data
X_train_processed, X_valid_processed = manual_preprocess(X_train, X_valid)

# ====================================================================================================================
# Initial modeling with sklearn Multiple Linear Regression
# ====================================================================================================================

# Fit linear regression model
lin_reg = LinearRegression()
fit = lin_reg.fit(X_train_processed, y_train)

# Make predictions
y_pred = lin_reg.predict(X_valid_processed)

# Evaluate model
lr_eval = evaluate_model_sk(y_valid, y_pred, 'lin_reg', 'LR')

# =======================================================================================
# Test multiple linear regression model assumptions
# =======================================================================================



# =============================
# Other stuff
# =============================
lin_reg.coef_
lin_reg.intercept_













