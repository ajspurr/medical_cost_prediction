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
output_dir = Path(project_dir, Path('./output/models'))

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
# Feature engineering
# ====================================================================================================================

# Based on EDA, created dichotomous column 'bmi_>=_30'
dataset['bmi_>=_30'] = dataset['bmi'] >= 30
bmi_dict = {False:'no', True:'yes'}
dataset['bmi_>=_30'] = dataset['bmi_>=_30'].map(bmi_dict)

# Add the new feature to the columns lists
categorical_cols.append('bmi_>=_30')
cat_ord_cols.append('bmi_>=_30')



# ********************************************************************************************
# ********************************************************************************************
# MORE FEATURE ENGINEERING IDEAS
# Feature engineering based on smoking, bmi relationships with age and charges etc.
# ********************************************************************************************
# ********************************************************************************************





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
def save_image(dir, filename, dpi=300, bbox_inches='tight'):
    plt.savefig(dir/filename, dpi=dpi, bbox_inches=bbox_inches)
    print("Saved image to '" + dir/filename +"'")
    
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
    
    # Imputation (Not relevant in this dataset, but keeping for future application)
    #num_imputer = SimpleImputer(strategy='mean')
    #imputed_X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    #imputed_X_valid_num = pd.DataFrame(num_imputer.transform(X_valid_num), columns=X_valid_num.columns, index=X_valid_num.index)
    
    # Scaling
    ss = StandardScaler()
    scaled_X_num = pd.DataFrame(ss.fit_transform(X_num), columns=X_num.columns, index=X_num.index)
    
    # =============================
    # Categorical preprocessing
    # =============================
    X_cat = X[categorical_cols]
    
    # Imputation (Not relevant in this dataset, but keeping for future application)
    #cat_imputer = SimpleImputer(strategy='most_frequent')
    #imputed_X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train_cat), columns=X_train_cat.columns, index=X_train_cat.index)
    #imputed_X_valid_cat = pd.DataFrame(cat_imputer.transform(X_valid_cat), columns=X_valid_cat.columns, index=X_valid_cat.index)
    
    # One-hot encoding
    OH_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
    OH_X_cat = pd.DataFrame(OH_encoder.fit_transform(X_cat), index=X_cat.index, columns=OH_encoder.get_feature_names_out())
    
    # Add preprocessed categorical columns back to preprocessed numerical columns
    X_processed = pd.concat([scaled_X_num, OH_X_cat], axis=1)
    
    return X_processed

# ====================================================================================================================
# Model evaluation function
# ====================================================================================================================
# Parameter 'model_name' will be used for coding and saving images
# Parameter 'model_display_name' will be used for plot labels

# def evaluate_model(X_train, X_valid, y_train, y_valid, y_pred, pipeline_or_model, model_name, 
#                    model_display_name, create_graphs=True, combine_graphs=True, export_graphs=False, round_results=3): 
    
def evaluate_model(y_valid, y_pred, model_name, model_display_name, create_graphs=True, 
                   combine_graphs=True, export_graphs=False, round_results=2):      
    metrics = {}
    metrics['max_e'] = max_error(y_valid, y_pred).round(round_results)
    metrics['mean_abs_e'] = mean_absolute_error(y_valid, y_pred).round(round_results)
    metrics['mse'] = mean_squared_error(y_valid, y_pred).round(round_results)
    metrics['rmse'] = np.sqrt(metrics['mse']).round(round_results)
    metrics['med_abs_e'] = median_absolute_error(y_valid, y_pred).round(round_results)
    metrics['r2'] = r2_score(y_valid, y_pred).round(round_results)
    
    print(model_display_name + ' Evaluation')
    print('Max Error: ' + str(metrics['max_e']))
    print('Mean Absolute Error: ' + str(metrics['mean_abs_e']))
    print('Mean Squared Error: ' + str(metrics['mse']))
    print('Root Mean Squared Error: ' + str(metrics['rmse']))
    print('Median Absolute Error: ' + str(metrics['med_abs_e']))
    print('R-squared: ' + str(metrics['r2']))
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
# ==========================================================
# Test for multicollinearity
# ==========================================================
# Calculate VIF
vif = calulate_vif(dataset, numerical_cols)

# All very close to 1, no multicollinearity

# ==========================================================
# Test for heteroscedasticity
# ==========================================================
# Plot residuals vs. predicted values
residuals = calculate_residuals(y_valid, y_pred)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.ylabel('Residuals')
plt.xlabel('Predicted Values')
plt.title('Residuals vs. Predicted Values')
plt.show()

# Tried to calculate 'standardized residuals' but didn't affect the distribution of data points
std = residuals.std()
mean = residuals.mean()
standardized_resid = (residuals - mean) / std
plt.scatter(y_pred, standardized_resid)
plt.axhline(y=0, color='red', linestyle='--')
plt.ylabel('Standardized Residuals')
plt.xlabel('Predicted Values')
plt.show()

# Plot True Values vs. Predicted Values to visualize the data differently
fig = plt.scatter(y_valid, y_pred)
plt.xlim([0, 50000])
plt.ylim([0, 50000])
ax = fig.axes
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', transform=ax.transAxes)
plt.title('True Values vs. Predicted Values')
plt.ylabel('Predicted Values')
plt.xlabel('True Values')
plt.show()


# ==========================================================
# Using statsmodels Multiple Linear Regression to test for heteroscedasticity
# ==========================================================
# https://datatofish.com/multiple-linear-regression-python/

# This is just to test for heteroscedasticity in entire data set, will not perform train/test split
sm_processed_X = manual_preprocess_sm(X)

# Add constant (required for statsmodels linear regression model)
sm_processed_X = sm.add_constant(sm_processed_X)

# Fit linear regression model
sm_lin_reg = sm.OLS(y, sm_processed_X).fit()

# Make predictions
sm_y_pred = sm_lin_reg.predict(sm_processed_X) 

# Plot residuals vs. predicted values
plt.scatter(sm_y_pred, sm_lin_reg.resid)
plt.axhline(y=0, color='red', linestyle='--')
plt.ylabel('SM Residuals')
plt.xlabel('SM Predicted Values')
plt.title('SM Residuals vs. Predicted Values')
plt.show()

# Plot standardized residuals vs. predicted values
standardized_residuals = sm_lin_reg.get_influence().resid_studentized_internal
plt.scatter(sm_y_pred, standardized_residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.ylabel('SM Residuals (standardized)')
plt.xlabel('SM Predicted Values')
plt.title('SM Residuals (standardized) vs. Predicted Values')
plt.show()

# Plot True Values vs. Predicted Values to visualize the data differently
fig = plt.scatter(y, sm_y_pred)
plt.xlim([0, 50000])
plt.ylim([0, 50000])
ax = fig.axes
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', transform=ax.transAxes)
plt.title('SM True Values vs. Predicted Values')
plt.ylabel('SM Predicted Values')
plt.xlabel('True Values')
plt.show()


# =============================
# Subgroup plots
# =============================
# Organize relevant data
standardized_residuals = pd.DataFrame(sm_lin_reg.get_influence().resid_studentized_internal)
standardized_residuals.columns = ['stand_resid']
relevant_data = pd.concat([dataset[['bmi_>=_30', 'smoker', 'charges']], sm_y_pred, standardized_residuals], axis=1)
relevant_data = relevant_data.rename(columns = {'charges':'y', 0:'y_pred'})

smoker_data = relevant_data[relevant_data['smoker']=='yes']
nonsmoker_data = relevant_data[relevant_data['smoker']=='no']
smoker_obese_data = smoker_data[smoker_data['bmi_>=_30']=='yes']
smoker_nonobese_data = smoker_data[smoker_data['bmi_>=_30']=='no']
nonsmoker_obese_data = nonsmoker_data[nonsmoker_data['bmi_>=_30']=='yes']
nonsmoker_nonobese_data = nonsmoker_data[nonsmoker_data['bmi_>=_30']=='no']

# True Values vs. Predicted Values subgrouped by smoking and bmi
plt.scatter(smoker_obese_data['y'], smoker_obese_data['y_pred'], alpha=0.5, label='obese smokers')
plt.scatter(smoker_nonobese_data['y'], smoker_nonobese_data['y_pred'], alpha=0.5, label='nonobese smokers')
plt.scatter(nonsmoker_obese_data['y'], nonsmoker_obese_data['y_pred'], alpha=0.5, label='obese nonsmokers')
plt.scatter(nonsmoker_nonobese_data['y'], nonsmoker_nonobese_data['y_pred'], alpha=0.5, label='nonobese nonsmokers')
largest_num = max(max(relevant_data['y']), max(relevant_data['y_pred']))
plt.plot([0, largest_num], [0, largest_num], color='darkblue', linestyle='--')
plt.title('SM True Values vs. Predicted Values')
plt.ylabel('SM Predicted Values')
plt.xlabel('True Values')
plt.legend()
plt.show()

# Plot standardized residuals vs. predicted values subgrouped by smoking and bmi
plt.scatter(smoker_obese_data['y_pred'], smoker_obese_data['stand_resid'], alpha=0.5, label='obese smokers')
plt.scatter(smoker_nonobese_data['y_pred'], smoker_nonobese_data['stand_resid'], alpha=0.5, label='nonobese smokers')
plt.scatter(nonsmoker_obese_data['y_pred'], nonsmoker_obese_data['stand_resid'], alpha=0.5, label='obese nonsmokers')
plt.scatter(nonsmoker_nonobese_data['y_pred'], nonsmoker_nonobese_data['stand_resid'], alpha=0.5, label='nonobese nonsmokers')
plt.axhline(y=0, color='red', linestyle='--')
plt.ylabel('SM Residuals (standardized)')
plt.xlabel('SM Predicted Values')
plt.title('SM Residuals (standardized) vs. Predicted Values')
plt.legend()
plt.show()

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

labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
white_test_results = dict(zip(labels, bp_test))
bp_test_results = dict(zip(labels, white_test))

# Both have a p-value <<< 0.05, indicating presence of heteroscedasticity

# =============================
# Subgroup plots and quantify heteroscedasticity function
# =============================

def subgroup_quantify_heteroscedasticity(sm_lr_model, orig_dataset, sm_y_pred, sm_y, plot_title):
    # Organize relevant data
    standardized_residuals = pd.DataFrame(sm_lr_model.get_influence().resid_studentized_internal)
    standardized_residuals.columns = ['stand_resid']
    #relevant_data = pd.concat([orig_dataset[['bmi_>=_30', 'smoker', 'charges']], sm_y_pred, standardized_residuals], axis=1)
    y_pred_series = pd.Series(sm_y_pred, name='y_pred')
    y_series = pd.Series(sm_y, name='y')
    relevant_data = pd.concat([orig_dataset[['bmi_>=_30', 'smoker']], y_series, y_pred_series, standardized_residuals], axis=1)
    #relevant_data = relevant_data.rename(columns = {'charges':'y', 0:'y_pred'})
    
    smoker_data = relevant_data[relevant_data['smoker']=='yes']
    nonsmoker_data = relevant_data[relevant_data['smoker']=='no']
    smoker_obese_data = smoker_data[smoker_data['bmi_>=_30']=='yes']
    smoker_nonobese_data = smoker_data[smoker_data['bmi_>=_30']=='no']
    nonsmoker_obese_data = nonsmoker_data[nonsmoker_data['bmi_>=_30']=='yes']
    nonsmoker_nonobese_data = nonsmoker_data[nonsmoker_data['bmi_>=_30']=='no']
    
    # True Values vs. Predicted Values subgrouped by smoking and bmi
    plt.scatter(smoker_obese_data['y'], smoker_obese_data['y_pred'], alpha=0.5, label='obese smokers')
    plt.scatter(smoker_nonobese_data['y'], smoker_nonobese_data['y_pred'], alpha=0.5, label='nonobese smokers')
    plt.scatter(nonsmoker_obese_data['y'], nonsmoker_obese_data['y_pred'], alpha=0.5, label='obese nonsmokers')
    plt.scatter(nonsmoker_nonobese_data['y'], nonsmoker_nonobese_data['y_pred'], alpha=0.5, label='nonobese nonsmokers')
    largest_num = max(max(relevant_data['y']), max(relevant_data['y_pred']))
    smallest_num = min(min(relevant_data['y']), min(relevant_data['y_pred']))
    plt.plot([smallest_num, largest_num], [smallest_num, largest_num], color='darkblue', linestyle='--')
    plt.title('SM True Values vs. Predicted Values\n(' + plot_title + ')')
    plt.ylabel('SM Predicted Values')
    plt.xlabel('True Values')
    plt.legend()
    plt.show()
    
    
    # Plot standardized residuals vs. predicted values subgrouped by smoking and bmi
    plt.scatter(smoker_obese_data['y_pred'], smoker_obese_data['stand_resid'], alpha=0.5, label='obese smokers')
    plt.scatter(smoker_nonobese_data['y_pred'], smoker_nonobese_data['stand_resid'], alpha=0.5, label='nonobese smokers')
    plt.scatter(nonsmoker_obese_data['y_pred'], nonsmoker_obese_data['stand_resid'], alpha=0.5, label='obese nonsmokers')
    plt.scatter(nonsmoker_nonobese_data['y_pred'], nonsmoker_nonobese_data['stand_resid'], alpha=0.5, label='nonobese nonsmokers')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.ylabel('SM Residuals (standardized)')
    plt.xlabel('SM Predicted Values')
    plt.title('SM Residuals (standardized) vs. Predicted Values\n(' + plot_title + ')')
    plt.legend()
    plt.show()
    
    # Quantify Heteroscedasticity using White test and Breusch-Pagan test
    white_test = het_white(sm_lr_model.resid, sm_lr_model.model.exog)
    bp_test = het_breuschpagan(sm_lr_model.resid, sm_lr_model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    white_test_results = dict(zip(labels, bp_test))
    bp_test_results = dict(zip(labels, white_test))
    
    return white_test_results, bp_test_results

# Combine statsmodels linear regression model creation, fitting, and returning results    
def fit_ols_test_heteroscedasticity(fxn_X, fxn_y, orig_dataset, plot_title):
    fxn_lin_reg = sm.OLS(fxn_y, fxn_X).fit()
    fxn_y_pred = fxn_lin_reg.predict(fxn_X) 
    
    fxn_white_test_results, fxn_bp_test_results = subgroup_quantify_heteroscedasticity(fxn_lin_reg, orig_dataset, fxn_y_pred, fxn_y, plot_title)
    
    return fxn_lin_reg, fxn_y_pred, fxn_white_test_results, fxn_bp_test_results

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
    
title_1 = 'Original'   
white_test_results_1, bp_test_results_1 = subgroup_quantify_heteroscedasticity(sm_lin_reg, dataset, sm_y_pred, y, title_1)
summary_1 = sm_lin_reg.summary()
summary_df_1 = sm_results_to_df(summary_1)
sm_lin_reg.rsquared

# =============================
# New model with feature incorporating relationship between BMI and smoking status
# =============================
# I initially tried to do this by added a new categorical column 'bmi_smoke_combo'
# Where 0=nonsmoker/nonobese, 1=smoker/nonobese, 2=nonsmoker/obese, 3=smoker/obese
# But that didn't work
# But this way, you multiply BMI by smoking status. So if they don't smoke, BMI doesn't affect predicted charge
# Which makes sense given the BMI vs. charge scatterplot
new_X_2 = sm_processed_X.copy()
new_X_2['bmi*smoker'] = new_X_2['smoker_yes'] * new_X_2['bmi']
title_2 = 'w [bmi*smoker] Feature'

sm_lin_reg_2, sm_y_pred_2, white_test_results_2, bp_test_results_2 = fit_ols_test_heteroscedasticity(new_X_2, y, dataset, title_2)
summary_df_2 = sm_results_to_df(sm_lin_reg_2.summary())
sm_lin_reg_2.rsquared

# =============================
# Compare coefficients before and after new feature
# =============================
compared_df = pd.DataFrame({'coef_orig':summary_df_1['coef'], 'coef_bmi*smok':summary_df_2['coef']}, index=summary_df_1.index)
compared_df.loc['bmi*smoker'] = [np.nan, summary_df_2.loc['bmi*smoker'][0]]
compared_df = compared_df.apply(pd.to_numeric)
compared_df['diff'] = compared_df['coef_bmi*smok'] - compared_df['coef_orig']

# =============================
# New model with feature incorporating relationship between between presence of obesity, smoking (and age)
# =============================
# This model (smoker*obese) worked very well
# It worked better than smoker*obese*age, which makes sense if you look at the age vs. charges graphs you'll see that they're almost
# horizontal lines. So it's not the age itself that is predictive, it's the difference between the three  
# groups: nonsmokers, obese smokers, and nonobese smokers. With this variable, the model can give it a coefficient that 
# is the average difference between obese smokers and nonobese smokers
new_X_3 = sm_processed_X.copy()
new_X_3['smoker*obese'] = new_X_3['smoker_yes'] * new_X_3['bmi_>=_30_yes']# * new_X_3['age']
title_3 = 'w [smoker*obese] Feature'

sm_lin_reg_3, sm_y_pred_3, white_test_results_3, bp_test_results_3 = fit_ols_test_heteroscedasticity(new_X_3, y, dataset, title_3)
summary_df_3 = sm_results_to_df(sm_lin_reg_3.summary())
sm_lin_reg_3.rsquared

# Tried incorporating group nonobese smoker
# This didn't do anything because the model already adds ~15,000 to the charges if you're a smoker
# Then with the new [smoker*obese] feature, it adds another ~20,000

# pd.set_option("display.max_rows", None)
# pd.reset_option("display.max_rows")

# =============================
# Add to coefficient comparison dataframe
# =============================
compared_df['coef_smok*obese'] = summary_df_3['coef']
compared_df.loc['smoker*obese'] = [np.nan, np.nan, np.nan, summary_df_3.loc['smoker*obese'][0]]

# =============================
# Try adding both new features 
# =============================
# COMPARED TO JUST OBESE/SMOKING FEATURE, THIS MADE GRAPH LOOK BETTER, R-SQUARED SLIGHTLY BETTER BY 0.004, WHITE TEST P-VALUES WORSE
# BY ABOUT 0.05
new_X_4 = new_X_3.copy()
new_X_4['bmi*smoker'] = new_X_2['bmi*smoker']
title_4 = 'w [bmi*smoker] and [smoker*obese] Feature'

sm_lin_reg_4, sm_y_pred_4, white_test_results_4, bp_test_results_4 = fit_ols_test_heteroscedasticity(new_X_4, y, dataset, title_4)
summary_df_4 = sm_results_to_df(sm_lin_reg_4.summary())
sm_lin_reg_4.rsquared

# =============================
# Add to coefficient comparison dataframe
# =============================
compared_df['coef_comb'] = summary_df_4['coef']

compared_df[['coef_orig', 'coef_bmi*smok', 'coef_smok*obese', 'coef_comb']]


# =============================
# Still some curvature to each group of residuals. Will try squaring age - WORKED
# =============================
new_X_5 = new_X_4.copy()
title_5 = 'w/ Both Features + age^2'

# Unfortunately, age has already been scalled around 0 and squaring will make all the negative numbers positive
# Will have to take the original ages, square, then scale

orig_ages = dataset['age'].to_frame()
squared_ages = np.power(orig_ages, 2)
scaled_ages = pd.DataFrame(StandardScaler().fit_transform(squared_ages), columns=['age^2'])

# plt.scatter(new_X_5['age'], y, alpha=0.5, label='orig age')
# plt.scatter(scaled_ages, y, alpha=0.5, label='squared age')

new_X_5['age'] = scaled_ages
new_X_5 = new_X_5.rename(columns={'age':'age^2'})

sm_lin_reg_5, sm_y_pred_5, white_test_results_5, bp_test_results_5 = fit_ols_test_heteroscedasticity(new_X_5, y, dataset, title_5)
summary_df_5 = sm_results_to_df(sm_lin_reg_5.summary())
sm_lin_reg_5.rsquared










# =============================
# Try log tranformation of original data THIS DIDN'T WORK
# =============================
new_X_3 = sm_processed_X.copy()
new_y = np.log(y)
sns.distplot(new_y)
plt.title('Charges Histogram (log tranformed)', fontsize=20, y=1.04)
plt.show()
title_3 = 'Log Transform Target (no new feature)'

sm_lin_reg_3, sm_y_pred_3, white_test_results_3, bp_test_results_3 = fit_ols_test_heteroscedasticity(new_X_3, new_y, dataset, title_3)
summary_df_3 = sm_results_to_df(sm_lin_reg_3.summary())
sm_lin_reg_3.rsquared

# =============================
# Try new features with log tranformation THIS DIDN'T WORK
# =============================
new_X_6 = new_X_5.copy()
new_y = np.log(y)
sns.distplot(new_y)
plt.title('Charges Histogram (log tranformed)', fontsize=20, y=1.04)
plt.show()
title_6 = 'Log Transform Target w new features'

sm_lin_reg_6, sm_y_pred_6, white_test_results_6, bp_test_results_6 = fit_ols_test_heteroscedasticity(new_X_6, new_y, dataset, title_6)
summary_df_6 = sm_results_to_df(sm_lin_reg_6.summary())
sm_lin_reg_6.rsquared

# =============================
# Other stuff
# =============================
lin_reg.coef_
lin_reg.intercept_














