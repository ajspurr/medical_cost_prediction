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

from statsmodels.tools.tools import add_constant
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
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
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
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_cat), index=X_train_cat.index, columns=OH_encoder.get_feature_names_out())
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid_cat), index=X_valid_cat.index, columns=OH_encoder.get_feature_names_out())
    
    # Add preprocessed categorical columns back to preprocessed numerical columns
    X_train_processed = pd.concat([scaled_X_train_num, OH_cols_train], axis=1)
    X_valid_processed = pd.concat([scaled_X_valid_num, OH_cols_valid], axis=1)
    
    return X_train_processed, X_valid_processed

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

# ====================================================================================================================
# Initial modeling with Multiple Linear Regression
# ====================================================================================================================
# Separate target from predictors
y = dataset['charges']
X = dataset.drop(['charges'], axis=1)

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# =======================================================================================
# Initial multiple linear regression model
# =======================================================================================
# Preprocess data
X_train_processed, X_valid_processed = manual_preprocess(X_train, X_valid)

# Fit logistic regression model
lin_reg = LinearRegression()
fit = lin_reg.fit(X_train_processed, y_train)

# Make predictions
y_pred = lin_reg.predict(X_valid_processed)

# Evaluate model
lr_eval = evaluate_model(y_valid, y_pred, 'lin_reg', 'LR')

# ==========================================================
# Test multiple linear regression model assumptions
# ==========================================================
# =============================
# Test for multicollinearity
# =============================
# Calculate VIF
# https://www.statology.org/multiple-linear-regression-assumptions/
# https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
new_X = X.copy()
new_X_num = new_X[numerical_cols]
new_X_num = add_constant(new_X_num)

vif = pd.Series([variance_inflation_factor(new_X_num.values, i) for i in range(new_X_num.shape[1])], index=new_X_num.columns)

# =============================
# Test for heteroscedasticity
# =============================
# Plot residuals vs. predicted values
residuals = calculate_residuals(y_valid, y_pred)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.ylabel('Residuals')
plt.xlabel('Predicted Values')
plt.title('Residuals vs. Predicted Values')
plt.show()

# Tried to calculate 'standardized residuals' but didn't affect the distribution of data points
# std = residuals.std()
# mean = residuals.mean()
# standardized_resid = (residuals - mean) / std
# plt.scatter(y_pred, standardized_resid)
# plt.ylabel('Standardized Residuals')
# plt.xlabel('Predicted Values')
# plt.show()

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



# =============================
# Other stuff
# =============================
lin_reg.coef_
lin_reg.intercept_

lin_reg.score(X_train_processed, y_train)




