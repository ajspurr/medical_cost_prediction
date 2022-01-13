import numpy as np
import pandas as pd
from os import chdir
import seaborn as sns 
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
from statsmodels.stats.diagnostic import het_white
from statsmodels.tools.eval_measures import meanabs
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

# ====================================================================================================================
# Visualization helper functions
# ====================================================================================================================
# Create 2d array of given size, used for figures with gridspec
def create_2d_array(num_rows, num_cols):
    matrix = []
    for r in range(0, num_rows):
        matrix.append([0 for c in range(0, num_cols)])
    return matrix

# Initialize figure, grid spec, axes variables
def initialize_fig_gs_ax(num_rows, num_cols, figsize=(16, 8)):
    # Create figure, gridspec, and 2d array of axes/subplots with given number of rows and columns
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    ax_array = create_2d_array(num_rows, num_cols)
    gs = fig.add_gridspec(num_rows, num_cols)

    # Map each subplot/axis to gridspec location
    for r in range(len(ax_array)):
        for c in range(len(ax_array[r])):
            ax_array[r][c] = fig.add_subplot(gs[r,c])

    # Flatten 2d array of axis objects to iterate through easier
    ax_array_flat = np.array(ax_array).flatten()
    
    return fig, gs, ax_array_flat

# Standardize image saving parameters
def save_image(filename, dir=models_output_dir, dpi=300, bbox_inches='tight'):
    plt.savefig(models_output_dir/filename, dpi=dpi, bbox_inches=bbox_inches)
    print("\nSaved image to '" + str(dir/filename) +"'\n")
    
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
# Parameter 'model_display_name' will be used for plot labels 
# Using the 'evaluate_model_sk()' and 'evaluate_model_sm()' on the SAME sm model yield almost identical results
def evaluate_model_sk(y_valid, y_pred, X, model_display_name, round_results=3, print_results=False):      
    metrics = {}
    metrics['max_e'] = max_error(y_valid, y_pred).round(round_results)
    metrics['mean_abs_e'] = mean_absolute_error(y_valid, y_pred).round(round_results)
    metrics['mse'] = mean_squared_error(y_valid, y_pred).round(round_results)
    metrics['rmse'] = np.sqrt(metrics['mse']).round(round_results)
    metrics['med_abs_e'] = median_absolute_error(y_valid, y_pred).round(round_results)
    metrics['r2'] = r2_score(y_valid, y_pred).round(round_results)
    metrics['r2_adj'] = 1 - ((1-metrics['r2'])*(len(y_valid)-1)/(len(y_valid)-X.shape[1]-1)).round(round_results)
    
    if print_results:
        print(model_display_name + ' Evaluation')
        print('Max Error: ' + str(metrics['max_e']))
        print('Mean Absolute Error: ' + str(metrics['mean_abs_e']))
        print('Mean Squared Error: ' + str(metrics['mse']))
        print('Root Mean Squared Error: ' + str(metrics['rmse']))
        print('Median Absolute Error: ' + str(metrics['med_abs_e']))
        print('R-squared: ' + str(metrics['r2']))
        print('R-squared (adj): ' + str(metrics['r2_adj']))
    return metrics

# Written for statsmodels model
# Using the 'evaluate_model_sk()' and 'evaluate_model_sm()' on the SAME sm model yield almost identical results
def evaluate_model_sm(y, y_pred, sm_lr_model, model_display_name, round_results=3, print_results=False):      
    metrics = {}
    metrics['max_e'] = np.round(max(sm_lr_model.resid), round_results)
    metrics['mean_abs_e'] = meanabs(y, y_pred).round(round_results)
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
        print(model_display_name + ' Evaluation')
        print('Max Error: ' + str(metrics['max_e']))
        print('Mean Absolute Error: ' + str(metrics['mean_abs_e']))
        print('Mean Squared Error: ' + str(metrics['mse']))
        print('Root Mean Squared Error: ' + str(metrics['rmse']))
        print('Median Absolute Error: ' + str(metrics['med_abs_e']))
        print('R-squared: ' + str(metrics['r2']))
        print('R-squared (adj): ' + str(metrics['r2_adj']))
        print('Breusch-Pagan LM p-val: ' + metrics['bp_lm_p'])
        print('White LM p-val: ' + metrics['white_lm_p'])
    return metrics

# Takes evalution metrics from evaluate_model() and plots confusion matrix, ROC, PRC, and precision/recall vs. threshold
# Parameter 'model_name' will be used for coding and saving images
# Parameter 'model_display_name' will be used for plot labels
def plot_model_metrics(model_name, model_display_name, conmat, conmat_df_perc, fpr, tpr, 
                       AUC, precision, recall, prc_thresholds, AUPRC, baseline, export_graphs):

    return

# Parameter 'model_name' will be used for coding and saving images
# Parameter 'model_display_name' will be used for plot labels
def plot_model_metrics_combined(model_name, model_display_name, conmat, conmat_df_perc, fpr, tpr, 
                                AUC, precision, recall, prc_thresholds, AUPRC, baseline, export_graphs):

    return
    
def calculate_residuals(y_valid, y_pred):    
    return y_valid - y_pred

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
        fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=(10, 5))
    
    # =============================
    # Plot standardized residuals vs. predicted values
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
        #plot1 = plt.scatter(y_pred, standardized_residuals)   
        #ax = plot1.axes
        plt.scatter(y_pred, standardized_residuals)
        ax1 = plt.gca()
    else:
        ax1 = ax_array_flat[0]
        ax1.scatter(y_pred, standardized_residuals) 
        
    ax1.axhline(y=0, color='darkblue', linestyle='--')
    ax1.set_ylabel('Standardized Residuals')
    ax1.set_xlabel('Predicted Values')
    ax1.set_title('Standardized Residuals vs. Predicted Values')
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
            save_image(save_filename)
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
    fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=(12, 5))
        
    # =============================
    # Plot standardized residuals vs. predicted values
    # =============================
    ax1 = ax_array_flat[0]
    ax1.scatter(smoker_obese_data['y_pred'], smoker_obese_data['stand_resid'], alpha=0.5, label='obese smokers')
    ax1.scatter(smoker_nonobese_data['y_pred'], smoker_nonobese_data['stand_resid'], alpha=0.5, label='nonobese smokers')
    ax1.scatter(nonsmoker_obese_data['y_pred'], nonsmoker_obese_data['stand_resid'], alpha=0.5, label='obese nonsmokers')
    ax1.scatter(nonsmoker_nonobese_data['y_pred'], nonsmoker_nonobese_data['stand_resid'], alpha=0.5, label='nonobese nonsmokers')
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_ylabel('Standardized Residuals')
    ax1.set_xlabel('Predicted Values')
    ax1.set_title('Standardized Residuals vs. Predicted Values')
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
        save_image(save_filename)
    plt.show()

    het_metrics = dict(zip(['BP', 'White'], [bp_test_results, white_test_results]))
    return het_metrics

# Combine statsmodels linear regression model creation, fitting, and returning results    
def fit_lr_model_results_subgrouped(fxn_X, fxn_y, plot_title, save_img=False, filename_unique=None):
    # Fit model
    fxn_lin_reg = sm.OLS(fxn_y, fxn_X).fit()
    
    # Predict target
    fxn_y_pred = fxn_lin_reg.predict(fxn_X) 
    
    #fxn_white_test_results, fxn_bp_test_results = subgroup_quantify_heteroscedasticity(fxn_lin_reg, orig_dataset, fxn_y_pred, fxn_y, plot_title)
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
title_0 = 'True Original'
model_name_0 = 'original'
het_metrics_0 = sm_lr_model_results(sm_lin_reg_0, y, sm_y_pred_0, combine_plots=True, plot_title=title_0, save_img=True, filename_unique=model_name_0)

# Organize model performance metrics
summary_df_0 = sm_results_to_df(sm_lin_reg_0.summary())
coeff_0 = pd.Series(summary_df_0['coef'], name=model_name_0)
sm_lr_results_0 = pd.Series(evaluate_model_sm(y, sm_y_pred_0, sm_lin_reg_0, 'LR (sm)'), name=model_name_0)

# ====================================================================================================================
# Feature engineering (more below)
# Based on EDA, created dichotomous feature 'bmi_>=_30'
# ====================================================================================================================

# Create new feature
new_X_1 = X.copy()
new_X_1['bmi_>=_30'] = new_X_1['bmi'] >= 30
bmi_dict = {False:'no', True:'yes'}
new_X_1['bmi_>=_30'] = new_X_1['bmi_>=_30'].map(bmi_dict)

# Add the new feature to the columns lists (necessary for preprocessing)
categorical_cols.append('bmi_>=_30')
cat_ord_cols.append('bmi_>=_30')

# Preprocess
new_X_1 = manual_preprocess_sm(new_X_1)

# Plot model
title_1 = 'w [bmi>=30] feature'
model_name_1 = '[bmi_>=_30]'
sm_lin_reg_1, sm_y_pred_1, het_results_1 = fit_lr_model_results_subgrouped(new_X_1, y, title_1, save_img=False, filename_unique='bmi_30_feature')

# Organize model performance metrics
summary_df_1 = sm_results_to_df(sm_lin_reg_1.summary())
coeff_1 = pd.Series(summary_df_1['coef'], name=model_name_1)
sm_lr_results_1 = pd.Series(evaluate_model_sm(y, sm_y_pred_1, sm_lin_reg_1, f'LR ({model_name_1})'), name=model_name_1)

# Keep track of model performance for comparison later
coeff_df = pd.concat([coeff_0, coeff_1], axis=1)
sm_results_df = pd.concat([sm_lr_results_0, sm_lr_results_1], axis=1)

# =======================================================================================
# Test for linear relationships between predictors and target variables
# =======================================================================================

# This analysis was done in the EDA section. Will explain findings here.

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
sm_lin_reg_2, sm_y_pred_2, het_results_2 = fit_lr_model_results_subgrouped(new_X_2, y, title_2, save_img=False, filename_unique='smoke_bmi_feature')

# Organize model performance metrics
summary_df_2 = sm_results_to_df(sm_lin_reg_2.summary())
coeff_2 = pd.Series(summary_df_2['coef'], name=model_name_2)
sm_lr_results_2 = pd.Series(evaluate_model_sm(y, sm_y_pred_2, sm_lin_reg_2, f'LR ({model_name_2})'), name=model_name_2)

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
title_3 = 'w [smoker*obese] Feature'
model_name_3 = '[smoker*obese]'
sm_lin_reg_3, sm_y_pred_3, het_results_3 = fit_lr_model_results_subgrouped(new_X_3, y, title_3, save_img=False, filename_unique='smoke_ob_feature')

# Organize model performance metrics
summary_df_3 = sm_results_to_df(sm_lin_reg_3.summary())
coeff_3 = pd.Series(summary_df_3['coef'], name=model_name_3)
sm_lr_results_3 = pd.Series(evaluate_model_sm(y, sm_y_pred_3, sm_lin_reg_3, f'LR ({model_name_3})'), name=model_name_3)

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
sm_lin_reg_4, sm_y_pred_4, het_results_4 = fit_lr_model_results_subgrouped(new_X_4, y, title_4, save_img=False, filename_unique='age_sq_feature')

# Organize model performance metrics
summary_df_4 = sm_results_to_df(sm_lin_reg_4.summary())
coeff_4 = pd.Series(summary_df_4['coef'], name=model_name_4)
sm_lr_results_4 = pd.Series(evaluate_model_sm(y, sm_y_pred_4, sm_lin_reg_4, f'LR ({model_name_4})'), name=model_name_4)

# Keep track of model performance for comparison later
coeff_df = pd.concat([coeff_df, coeff_4], axis=1)
sm_results_df = pd.concat([sm_results_df, sm_lr_results_4], axis=1)

# Removing original 'age' feature didn't change model as it already had a relatively small coefficient (-68) once
# age^2 was added. I also tried not scaling the age^2 feature. This didn't change model at all. Literally same 
# coefficients other than it's own being significantly smaller (3000 -> 3)

# =======================================================================================
# Compare coefficients before and after new features
# =======================================================================================
# All features
coeff_df = coeff_df.apply(pd.to_numeric)

# Filter out original features
orig_features_df = coeff_df.iloc[0:9]

# Separate larger and smaller coefficients (scale makes smaller coefficients harder to visualize)
orig_small_features_list = ['const', 'age', 'smoker_yes']
large_coeff_df = orig_features_df.loc[orig_small_features_list]
small_coeff_df = orig_features_df.drop(orig_small_features_list, axis=0)

# Filter out new features
new_features = coeff_df.iloc[9:len(coeff_df.index)]
new_features = new_features.replace(np.nan, 0)

# =============================
# Plot separately
# =============================
for feature in small_coeff_df.index:
    plt.plot(small_coeff_df.columns, small_coeff_df.loc[feature].to_list(), label=feature)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
plt.show()

for feature in large_coeff_df.index:
    plt.plot(large_coeff_df.columns, large_coeff_df.loc[feature].to_list(), label=feature)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
plt.show()

for feature in new_features.index:
    plt.plot(new_features.columns, new_features.loc[feature].to_list(), label=feature)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
plt.show()

# =============================
# Plot together horizontal
# =============================
# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=3, figsize=(16, 6))

# Small coeff features
ax1 = ax_array_flat[0]
for feature in small_coeff_df.index:
    ax1.plot(small_coeff_df.columns, small_coeff_df.loc[feature].to_list(), label=feature)

ax1.legend(loc='upper right', borderaxespad=0.5, title='Variable')
ax1.set_ylabel('Coefficient', fontsize=16)
plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')

# Large coeff features
ax2 = ax_array_flat[1]
for feature in large_coeff_df.index:
    ax2.plot(large_coeff_df.columns, large_coeff_df.loc[feature].to_list(), label=feature)
ax2.legend(loc='upper right', borderaxespad=0.5, title='Variable')
ax2.set_xlabel('New Features', fontsize=16)
plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')

# New features
ax3 = ax_array_flat[2]
for feature in new_features.index:
    ax3.plot(new_features.columns, new_features.loc[feature].to_list(), label=feature)
ax3.legend(loc='upper left', borderaxespad=0.5, title='Variable')
plt.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')

fig.suptitle('Variable coefficients with each additional feature', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'coeff_new_feat_horiz'
#save_image(save_filename)
plt.show()

# =============================
# Plot together vertical
# =============================
# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=3, num_cols=1, figsize=(6, 10))

# Small coeff features
ax1 = ax_array_flat[0]
for feature in small_coeff_df.index:
    ax1.plot(small_coeff_df.columns, small_coeff_df.loc[feature].to_list(), label=feature)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Variable')
ax1.set_ylabel('Coefficient', fontsize=16)
plt.setp(ax1.get_xticklabels(), rotation=20, horizontalalignment='right')

# Large coeff features
ax2 = ax_array_flat[1]
for feature in large_coeff_df.index:
    ax2.plot(large_coeff_df.columns, large_coeff_df.loc[feature].to_list(), label=feature)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Variable')
ax2.set_ylabel('Coefficient', fontsize=16)
plt.setp(ax2.get_xticklabels(), rotation=20, horizontalalignment='right')

# New features
ax3 = ax_array_flat[2]
for feature in new_features.index:
    ax3.plot(new_features.columns, new_features.loc[feature].to_list(), label=feature)
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Variable')
ax3.set_xlabel('New Features', fontsize=16)
ax3.set_ylabel('Coefficient', fontsize=16)
plt.setp(ax3.get_xticklabels(), rotation=20, horizontalalignment='right')

fig.suptitle('Variable coeff w/ each additional feature', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'coeff_new_feat_vert'
#save_image(save_filename)
plt.show()

# =============================
# Drop variables whose coefficients don't change much
# =============================
# Variables that don't change much: # children, sex_male, all regions, const.
# All features
coeff_df_new = coeff_df.apply(pd.to_numeric)

# Drop variables
drop_var = ['children', 'region_northwest', 'region_southeast', 'region_southwest', 'const']
coeff_df_new = coeff_df_new.drop(drop_var, axis=0)

# Replace NaN with 0
coeff_df_new = coeff_df_new.replace(np.nan, 0)

# Separate new and old features
orig_features_df = coeff_df_new.iloc[0:4]
new_features_df = coeff_df_new.iloc[4:len(coeff_df_new.index)]

# Plot combined
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=3, num_cols=1, figsize=(9, 13))

smoker_df = orig_features_df['smoker_yes']
ax1 = ax_array_flat[0]
for feature in smoker_df.index:
    ax1.plot(smoker_df.columns, smoker_df.loc[feature].to_list(), label=feature, linewidth=3)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
plt.setp(ax1.get_xticklabels(), rotation=20, horizontalalignment='right')
ax1.set_title('New Feature Coefficients')
ax1.set_xlabel('Additional Features', fontsize=16)
ax1.set_ylabel('Coefficient', fontsize=16)
ax1.grid()

orig_features_no_smoker = orig_features_df.drop(['smoker_yes'], axis=0)
ax2 = ax_array_flat[1]
for feature in orig_features_no_smoker.index:
    ax2.plot(orig_features_no_smoker.columns, orig_features_no_smoker.loc[feature].to_list(), label=feature, linewidth=3)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
plt.setp(ax2.get_xticklabels(), rotation=20, horizontalalignment='right')
ax2.set_title('New Feature Coefficients')
ax2.set_xlabel('Additional Features', fontsize=16)
ax2.set_ylabel('Coefficient', fontsize=16)
ax2.grid()

ax2 = ax_array_flat[0]
for feature in new_features_df.index:
    ax2.plot(new_features_df.columns, new_features_df.loc[feature].to_list(), label=feature, linewidth=2.5)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Feature')
plt.setp(ax2.get_xticklabels(), rotation=20, horizontalalignment='right')
ax2.set_title('Original Feature Coefficients')
ax2.set_ylabel('Coefficient', fontsize=16)
ax2.grid()

fig.suptitle('Feature coeff w/ each additional feature', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'coeff_new_feat_vert_3'
#save_image(save_filename)
plt.show()


# =======================================================================================
# Test for multicollinearity
# =======================================================================================
# Calculate VIF
vif = calulate_vif(dataset, numerical_cols)

# All very close to 1, no multicollinearity. (Greater than 5-10 indicates multicollinearity)

smoker_dataset = dataset[dataset['smoker']=='yes']
plt.scatter(smoker_dataset['bmi'], smoker_dataset['age'])

# =============================
# Quantify Heteroscedasticity
# =============================

# Model Summary
sm_results = sm_lin_reg.summary()

# Quantify Heteroscedasticity using White test and Breusch-Pagan test
# https://medium.com/@remycanario17/tests-for-heteroskedasticity-in-python-208a0fdb04ab
# https://www.statology.org/breusch-pagan-test/
white_test = het_white(sm_lin_reg.resid, sm_lin_reg.model.exog)
bp_test = het_breuschpagan(sm_lin_reg.resid, sm_lin_reg.model.exog)

# Results returned as tuple of results: LM Statistic, LM-Test p-value, F-Statistic, F-Test p-value
white_lm_p_value = white_test[1]
formatted = '{:0.2e}'.format(white_lm_p_value)

labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
white_test_results = dict(zip(labels, white_test))
bp_test_results = dict(zip(labels, bp_test))

# Both have a p-value <<< 0.05, indicating presence of heteroscedasticity

# Before performingfeature engineering, tried log transforming target. Did not work either before
# or after feature engineering.












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
lr_eval = evaluate_model(y_valid, y_pred, 'lin_reg', 'LR')

# =======================================================================================
# Test multiple linear regression model assumptions
# =======================================================================================



# =============================
# Other stuff
# =============================
lin_reg.coef_
lin_reg.intercept_













