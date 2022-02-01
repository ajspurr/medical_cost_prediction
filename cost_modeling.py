import sys
import numpy as np
import pandas as pd
from os import chdir
import seaborn as sns 
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import PureWindowsPath, Path

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
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

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

# Make my own colormap
from matplotlib import cm
from matplotlib.colors import ListedColormap
Set1 = cm.get_cmap('Set1', 8)
tab10  = cm.get_cmap('tab10', 8)
my_cmap = new_cmap3 = ListedColormap(np.vstack([tab10.colors[0:2],  Set1.colors[0], tab10.colors[2]]))

# Read in data
project_dir = PureWindowsPath(r"D:\GitHubProjects\medical_cost_prediction\\")
chdir(project_dir)
dataset = pd.read_csv('./input/insurance.csv')
ml_models_output_dir = Path(project_dir, Path('./output/models/ml'))

# Import my data science helper functions (relative dir based on project_dir)
my_module_dir = str(Path.resolve(Path('../my_ds_modules')))
sys.path.insert(0, my_module_dir)
import ds_helper as dh

# ====================================================================================================================
# Categorize and process features
# ====================================================================================================================

categorical_cols = []
numerical_cols = []

# Made into function for debugging
def reset_column_categories_create_format_dict(fxn_X):
    """
    Updates global variables categorical_cols and numerical_cols to include names of categorical data and 
    numerical data, respectively.

    Parameters
    ----------
    fxn_X : pandas.DataFrame
        Dataset features and their values. This dataframe should not include the target.

    Returns
    -------
    None.

    """
    global categorical_cols
    global numerical_cols
    
    # Separate categorical and numerical features
    categorical_cols = [cname for cname in fxn_X.columns if fxn_X[cname].dtype == "object"]
    numerical_cols = [cname for cname in fxn_X.columns if not fxn_X[cname].dtype == "object"]
    
    print_num_cat_cols()
    
    # Create formatted columns dictionary in dh module
    dh.create_formatted_cols_dict(dataset.columns)
    dh.add_edit_formatted_col('bmi', 'BMI')

def print_num_cat_cols():
    global categorical_cols
    global numerical_cols
    print(f"Numerical Columns: {numerical_cols}")
    print(f"Categorical Columns: {categorical_cols}")
    
    
reset_column_categories_create_format_dict(dataset.drop('charges', axis=1))


def add_feature(name, data, feature_type, fxn_dataset):
    if feature_type == 'c':
        categorical_cols.append(name)
    elif feature_type == 'n':
        numerical_cols.append(name)
    else:
        raise Exception("Parameter 'feature_type' must be 'c' for categorical or 'n' for numerical")
    
    fxn_dataset[name] = data    

def remove_feature(name, feature_type, fxn_dataset):
    if feature_type == 'c':
        categorical_cols.remove(name)
    elif feature_type == 'n':
        numerical_cols.remove(name)
    else:
        raise Exception("Parameter 'feature_type' must be 'c' for categorical or 'n' for numerical")
    
    fxn_dataset.drop(name, axis=1, inplace=True)
    return    

# ====================================================================================================================
# Visualization helper functions
# ====================================================================================================================
# Function returning the formatted version of column name
def format_col(col_name):
    return dh.format_col(col_name)

# Takes dataframe of X values, uses them to create Series with categories:
# 'obese smokers', 'nonobese smokers', 'obese nonsmokers', 'nonobese nonsmokers'
# Returns a series of said categories which corresponds to 'X_df' parameter
# https://datagy.io/pandas-conditional-column/
def create_obese_smoker_category(X_df):
    conditions = [
        (X_df['bmi_>=_30_yes'] == 1) & (X_df['smoker_yes'] == 1),
        (X_df['bmi_>=_30_yes'] == 0) & (X_df['smoker_yes'] == 1),
        (X_df['bmi_>=_30_yes'] == 1) & (X_df['smoker_yes'] == 0),
        (X_df['bmi_>=_30_yes'] == 0) & (X_df['smoker_yes'] == 0)
    ]
    
    category_names = ['obese smokers', 'nonobese smokers', 'obese nonsmokers', 'nonobese nonsmokers']
    return pd.Series(np.select(conditions, category_names), name='grouping') 

def create_obese_smoker_category_2(X_df):
    conditions = [
        (X_df['bmi_>=_30'] == 'yes') & (X_df['smoker'] == 'yes'),
        (X_df['bmi_>=_30'] == 'no') & (X_df['smoker'] == 'yes'),
        (X_df['bmi_>=_30'] == 'yes') & (X_df['smoker'] == 'no'),
        (X_df['bmi_>=_30'] == 'no') & (X_df['smoker'] == 'no')
    ]
    
    category_names = ['obese smokers', 'nonobese smokers', 'obese nonsmokers', 'nonobese nonsmokers']
    return pd.Series(np.select(conditions, category_names), name='grouping') 

def create_obese_smoker_category_3(X_df):
    conditions = [
        (X_df['bmi_>=_30'] == True) & (X_df['smoker'] == 'yes'),
        (X_df['bmi_>=_30'] == False) & (X_df['smoker'] == 'yes'),
        (X_df['bmi_>=_30'] == True) & (X_df['smoker'] == 'no'),
        (X_df['bmi_>=_30'] == False) & (X_df['smoker'] == 'no')
    ]
    
    category_names = ['obese smokers', 'nonobese smokers', 'obese nonsmokers', 'nonobese nonsmokers']
    return pd.Series(np.select(conditions, category_names), name='grouping') 

def create_obese_smoker_category_4(X_df):
    conditions = [
        (X_df['bmi_>=_30'] == 1) & (X_df['smoker'] == 'yes'),
        (X_df['bmi_>=_30'] == 0) & (X_df['smoker'] == 'yes'),
        (X_df['bmi_>=_30'] == 1) & (X_df['smoker'] == 'no'),
        (X_df['bmi_>=_30'] == 0) & (X_df['smoker'] == 'no')
    ]
    
    category_names = ['obese smokers', 'nonobese smokers', 'obese nonsmokers', 'nonobese nonsmokers']
    return pd.Series(np.select(conditions, category_names), name='grouping') 

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


# class MultiplyTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         print('\n>>>>>init() called\n')
    
#     def fit(self, X, y=None):
#         print('\n>>>>>fit() called\n')
#         return self
    
#     def transform(self, X, y=None):
#         print('\n>>>>>transform() called\n')
#         #X_ = X.copy()
#         print(X)
#         X['bmi'] = X['bmi'] * X['smoker']
#         # for i, thing in enumerate(X):
#         #     print(thing)
#         #     if i > 20: break
#         #X.bmi = X.bmi * X.smoker
#         print('\n>>>>>transform() finished\n')
#         return X
 
# Works if I use create_pipeline_bmi_smoker() that uses the column transformer for
# my custom transformer
# class MultiplyTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         print('\n>>>>>init() called\n')
#         self.means = None
#         self.std = None
    
#     def fit(self, X, y=None):
#         print('\n>>>>>fit() called\n')
#         self.means = X.to_numpy().mean(axis=0)[0] # Because right now its calculating mean for both 'bmi' and 'smoker'
#         self.std = X.to_numpy().std(axis=0)[0]
        
#         return self
    
#     def transform(self, X, y=None):
#         print('\n>>>>>transform() called\n')
#         bmi = X['bmi'] * X['smoker']
#         for index, value in enumerate(bmi):
#             if value != 0:
#                 # Scale values
#                 bmi.iat[index] = (value - self.means) / self.std

#         X['bmi'] = bmi
#         print('\n>>>>>transform() finished\n')
#         return X

class MultiplyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('\n>>>>>init() called\n')
        self.means = None
        self.std = None
    
    def fit(self, X, y=None):
        print('\n>>>>>fit() called\n')
        self.means = X.bmi.to_numpy().mean(axis=0)
        print(f"mean bmi = {self.means}")
        self.std = X.bmi.to_numpy().std(axis=0)
        print(f"std bmi = {self.std}")
        return self
    
    def transform(self, X, y=None):
        print('\n>>>>>transform() called\n')
        bmi_smoker = X['bmi'] * X['smoker']
        
        # Scale to original 'bmi' mean and std
        for index, value in enumerate(bmi_smoker):
            if value != 0:
                # Scale values
                bmi_smoker.iat[index] = (value - self.means) / self.std
        
        # Copy so as not to affect original data
        X_copy = X.copy()
        X_copy['bmi*smoker'] = bmi_smoker
        print('\n>>>>>transform() finished\n')
        return X_copy

def create_pipeline_bmi_smoker(model_name, model):
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
    
    # Try as feature union
    #union = FeatureUnion([('bmi_smoker', MultiplyTransformer())])
    
    # Bundle preprocessor and model
    my_pipeline = Pipeline([
        ('bmi_smoker', MultiplyTransformer()),
        #('union', union),
        ('preprocessor', preprocessor),
        (model_name, model)
    ])

    return my_pipeline

# This one worked, but using MultiplyTransformer() as a column transformer, I needed
# to pass it both 'bmi' and 'smoker' columns so it duplicated smoker column
# def create_pipeline_bmi_smoker(model_name, model):
#     # Preprocessing for numerical data (SimpleImputer default strategy='mean')
#     numerical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer()), 
#         ('scale', StandardScaler())
#     ])
    
#     # Preprocessing for categorical data
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')), 
#         ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False))
#     ])
    
#     # Bundle preprocessing for numerical and categorical data
#     preprocessor = ColumnTransformer(transformers=[
#             ('bmi_smoker', MultiplyTransformer(), ['bmi', 'smoker']),
#             ('num', numerical_transformer, numerical_cols),
#             ('cat', categorical_transformer, categorical_cols)
#     ])
    
#     # Bundle preprocessor and model
#     my_pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         (model_name, model)
#     ])

    # return my_pipeline


# def create_pipeline_bmi_smoker(model_name, model):
#     # Preprocessing for numerical data (SimpleImputer default strategy='mean')
#     numerical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer()), 
#         ('scale', StandardScaler())
#     ])
    
#     # Preprocessing for categorical data
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')), 
#         ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False))
#     ])
    
#     # Bundle preprocessing for numerical and categorical data
#     preprocessor_num = ColumnTransformer(transformers=[
#             ('num', numerical_transformer, numerical_cols)
#     ])
    
#     # Bundle preprocessing for numerical and categorical data
#     preprocessor_cat = ColumnTransformer(transformers=[
#             ('cat', categorical_transformer, categorical_cols)
#     ])
    
#     # Bundle preprocessor and model
#     my_pipeline = Pipeline([
#         ('bmi_smoker', MultiplyTransformer()),
#         ('preprocessor_num', preprocessor_num),
#         ('preprocessor_cat', preprocessor_cat),
#         (model_name, model)
#     ])
#     return my_pipeline

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

def manual_preprocess_bmi_smoker(X_train, X_valid):
    
    X_train_num = X_train[numerical_cols]
    X_valid_num = X_valid[numerical_cols]
    
    X_train_cat = X_train[categorical_cols]
    X_valid_cat = X_valid[categorical_cols]
    
    # =============================
    # Numerical preprocessing
    # =============================
    # Imputation (Not relevant in this dataset, but keeping for future application)
    #num_imputer = SimpleImputer(strategy='mean')
    #imputed_X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    #imputed_X_valid_num = pd.DataFrame(num_imputer.transform(X_valid_num), columns=X_valid_num.columns, index=X_valid_num.index)
    
    # Scaling
    ss = StandardScaler()
    scaled_X_train_num = pd.DataFrame(ss.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    scaled_X_valid_num = pd.DataFrame(ss.transform(X_valid_num), columns=X_valid_num.columns, index=X_valid_num.index)
    
    # Create ['bmi*smoker'] feature, specifically after scaling so it scales BMI properly first
    scaled_X_train_num['bmi*smoker'] = X_train_cat['smoker'] * scaled_X_train_num['bmi']
    scaled_X_valid_num['bmi*smoker'] = X_valid_cat['smoker'] * scaled_X_valid_num['bmi']
    
    # =============================
    # Categorical preprocessing
    # =============================
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

# Preprocessing of all independent variable data together (no train/test split) for use with statmodels (sm) data analysis
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
# Recreate three datasets from cost_lin_reg.py
# One with the new features I created in cost_lin_reg.py and with their source features removed
# One with the original set of Cook's outliers removed
# One with both sets of Cook's outliers removed
# ====================================================================================================================

# ==========================================================
# Dataset with new features I created in cost_lin_reg.py and with their source features removed
# ==========================================================

new_features_data = dataset.copy()

# Separate target from predictors
y = new_features_data['charges']
X = new_features_data.drop(['charges'], axis=1)

# MAKE SURE LIST OF CATEGORICAL AND NUMERICAL COLUMNS IS UPDATED BEFORE USING PREPROCESSING FUNCTION
reset_column_categories_create_format_dict(X)

# Create a few of the features before preprocessing

# Create ['age^2'] feature
age_sq = np.power(X['age'], 2)
add_feature('age^2', age_sq, 'n', X)
remove_feature('age', 'n', X)

# Create feature ['bmi_>=_30'] temporarily to create the ob_smoke_series using create_obese_smoker_category()
# and to use for obese*smoker feature later
bmi_30 = X['bmi'] >= 30
obese_dict = {False:0, True:1}
bmi_30 = bmi_30.map(obese_dict)
add_feature('bmi_>=_30', bmi_30, 'c', X)
ob_smoke_series = create_obese_smoker_category_4(X)

# Create ['smoker*obese'] feature
smoker_dict = {'no':0, 'yes':1}
X['smoker'] = X['smoker'].map(smoker_dict)
smoke_obese = X['smoker'] * X['bmi_>=_30']
add_feature('smoker*obese', smoke_obese, 'c', X)

print_num_cat_cols()

# I originally wanted to preprocess the data before adding the ['bmi*smoker'] feature
# because I wanted 'bmi' to be scaled with all 'bmi' values, rather than just scaled with the bmi
# of smokers, which would happen if I created the feature first. But the way sklearn functions work
# you have to create the preprocess pipeline after all feature engineering is done, especially if you 
# plan on using cross-validation. So I will see if I can add the feature first. I'll see if 'bmi' has a similar
# distribution in smokers and nonsmokers, and I'll see how adding the feature before preprocessing affects
# the new features min, max, med, mean

# Check dist of bmi in smoker vs not: they are almost identical
sns.kdeplot(x=X[X['smoker']==1]['bmi'], label='smokers', shade=True, alpha=0.8)
sns.kdeplot(x=X[X['smoker']==0]['bmi'], label='nonsmokers', shade=True, alpha=0.8)
plt.legend()

# Compare two orders of operation:
    # 1. Scale 'bmi' first, then create ['bmi*smoker']
    # 2. Create ['bmi*smoker'], then scale it

# 1. Scale 'bmi' first, then create ['bmi*smoker']
ss = StandardScaler()
scaled_bmi = pd.DataFrame(ss.fit_transform(X['bmi'].to_frame()))[0]
bmi_smoker_1 = X['smoker'] * scaled_bmi

# 2. Create ['bmi*smoker'], then scale it
bmi_smoker_2 = X['smoker'] * X['bmi']
ss = StandardScaler()
bmi_smoker_2 = pd.DataFrame(ss.fit_transform(bmi_smoker_2.to_frame()))[0]

# Compare: NOT the same distribution because it takes the zeros into account in the scaling, which throws it all off
sns.kdeplot(x=bmi_smoker_1, label='scale before', shade=True, alpha=0.8)
sns.kdeplot(x=bmi_smoker_2, label='scale after', shade=True, alpha=0.8)
plt.legend()


# I can't scale before because I need to use a pipeline for cross-val scores and it needs to fit_transform
# the test data but only transform the validation data. So try to get standard scaler to ignore the zeroes 
# try converting zeroes to nan? but then the model won't use them? let's see
bmi_smoker_3 = X['smoker'] * X['bmi']
bmi_smoker_3.replace(0, np.nan, inplace=True)
ss = StandardScaler()
bmi_smoker_3 = pd.DataFrame(ss.fit_transform(bmi_smoker_3.to_frame()))[0]

# Compare distributions:
sns.kdeplot(x=bmi_smoker_1, label='scale before', shade=True, alpha=0.8)
sns.kdeplot(x=bmi_smoker_3, label='scale after (0=NaN)', shade=True, alpha=0.8)
plt.legend()

diff = bmi_smoker_3 - bmi_smoker_1
perc_diff = (diff / bmi_smoker_1) * 100

diff_df = pd.concat([diff, perc_diff], axis=1)
diff_df.dropna(inplace=True)
diff_df.rename(columns={0:'diff', 1:'perc_diff'}, inplace=True)

# Distribution of differences mostly very close to zero, a couple huge outliers in perc_diff though
sns.kdeplot(x=diff_df['diff'], label='diff', shade=True, alpha=0.8)
sns.kdeplot(x=diff_df['perc_diff'], label='% diff', shade=True, alpha=0.8)
plt.legend()


# =============================
# What about using the pipeline to create the feature? So it can happen after the scaling
# =============================

# First compare pipeline preprocess to manual, so I can make sure everything is happening correctly
#test_pipe_data = dataset.copy()

# Separate target from predictors
y = dataset['charges']
X_pipe_test = X.copy() # use the X with 3 out of 4 new features created above

# Categorize columns 
# smoker_dict = {0:'no', 1:'yes'}
# X_pipe_test['bmi_>=_30'] = X_pipe_test['bmi_>=_30'].map(smoker_dict)
# X_pipe_test['smoker*obese'] = X_pipe_test['smoker*obese'].map(smoker_dict)
# X_pipe_test['smoker'] = X_pipe_test['smoker'].map(smoker_dict)
# reset_column_categories_create_format_dict(X_pipe_test)

print_num_cat_cols()

# Train/valid split
X_train, X_valid, y_train, y_valid = train_test_split(X_pipe_test, y, train_size=0.8, test_size=0.2, random_state=15)

# Manual preprocess
X_train_processed, X_valid_processed = manual_preprocess_bmi_smoker(X_train, X_valid)



# Compare distributions: manual_preprocess_bmi_smoker() works exactly the same as scaling bmi first, then creating bmi*smoker
ss = StandardScaler()
bmi_train_scaled_old = pd.DataFrame(ss.fit_transform(X_train['bmi'].to_frame()))[0]
bmi_valid_scaled_old = pd.DataFrame(ss.transform(X_valid['bmi'].to_frame()))[0]

smoker_train_reset_index = X_train['smoker'].reset_index(drop=True)
smoker_valid_reset_index = X_valid['smoker'].reset_index(drop=True)
bmi_smoker_train = bmi_train_scaled_old * smoker_train_reset_index
bmi_smoker_valid = bmi_valid_scaled_old * smoker_valid_reset_index

sns.kdeplot(x=X_train_processed['bmi*smoker'], label='train', shade=True, alpha=0.8)
sns.kdeplot(x=X_valid_processed['bmi*smoker'], label='valid', shade=True, alpha=0.8)
sns.kdeplot(x=bmi_smoker_train, label='train_old', shade=True, alpha=0.8)
sns.kdeplot(x=bmi_smoker_valid, label='valid_old', shade=True, alpha=0.8)
plt.legend()

sns.kdeplot(x=X_train_processed['bmi*smoker'], label='train', shade=True, alpha=0.8)
sns.kdeplot(x=bmi_smoker_train, label='train_old', shade=True, alpha=0.8)
plt.legend()

sns.kdeplot(x=X_valid_processed['bmi*smoker'], label='valid', shade=True, alpha=0.8)
sns.kdeplot(x=bmi_smoker_valid, label='valid_old', shade=True, alpha=0.8)
plt.legend()
# manual_preprocess_bmi_smoker() works exactly the same as scaling bmi first, then creating bmi*smoker



# Compare current pipeline to manual preprocess
my_pipeline = create_pipeline('LR', LinearRegression())

results = my_pipeline.fit(X_train, y_train)
y_pred = my_pipeline.predict(X_valid)

# See intermediate pipeline data
# https://stackoverflow.com/questions/45626780/getting-transformer-results-from-sklearn-pipeline-pipeline
new_pipe = my_pipeline.named_steps['preprocessor']
pipeline_transformed_train_data = pd.DataFrame(new_pipe.transform(X_train))
pipeline_transformed_valid_data = pd.DataFrame(new_pipe.transform(X_valid))

# Compare intermediate pipeline data to manually processed data - its the same. Other than not adding bmi*smoker yet
X_train_processed.iloc[:,0]
pipeline_transformed_train_data.iloc[:,0]

X_train_processed.iloc[:,1]
pipeline_transformed_train_data.iloc[:,1]

X_train_processed.iloc[:,:-6]
pipeline_transformed_train_data.iloc[:,:-6]

X_valid_processed.iloc[:,:-6]
pipeline_transformed_valid_data.iloc[:,:-6]

X_train.columns


# Try to create a pipeline version of manual_preprocess_bmi_smoker()
my_pipeline2 = create_pipeline_bmi_smoker('LR', LinearRegression())
results2 = my_pipeline2.fit(X_train, y_train)
y_pred2 = my_pipeline2.predict(X_valid)
# See intermediate pipeline data
new_pipe2 = my_pipeline2.named_steps['preprocessor']
pipeline_transformed_train_data2 = pd.DataFrame(new_pipe2.transform(X_train))
pipeline_transformed_valid_data2 = pd.DataFrame(new_pipe2.transform(X_valid))

X_train_processed.iloc[:,:-8]
pipeline_transformed_train_data2.iloc[:,:-7]

# at this point, the transform works, but it's not actually adding bmi*smoker as a new column
# even though I tell transform to do it. So I could try just replacing 'bmi' with 'bmi*smoker'

















results.feature_names_in_

my_pipeline.steps
my_pipeline.named_steps['preprocessor']















# Preprocess data
sm_processed_X = manual_preprocess_sm(X)

# Put dataset back together with preprocessed X
new_features_data = pd.concat([sm_processed_X, y], axis=1)








# Create ['bmi*smoker'] feature, specifically after preprocessing so it scales BMI properly first
new_features_data['bmi*smoker'] = new_features_data['smoker_1'] * new_features_data['bmi']

# Remove old variables from dataset: 'bmi_>=_30', 'bmi', and 'age' 
remove_var = ['bmi', 'bmi_>=_30_1']
new_features_data = new_features_data.drop(remove_var, axis=1)

# Remove old variables from categorical and numerical columns lists (relevant in case I used preprocessing again)
list_remove = ['bmi', 'age']
numerical_cols = [col for col in numerical_cols if col not in list_remove]
cont_cols = [col for col in cont_cols if col not in list_remove]
cont_cols_w_target = [col for col in cont_cols_w_target if col not in list_remove]
categorical_cols.remove('bmi_>=_30')


# ==========================================================
# Dataset with new features and first set of Cook's outliers removed
# ==========================================================
first_no_outliers_data = new_features_data.copy()

# Need to build model to calculate Cook's distances

# Separate target from predictors
y = first_no_outliers_data['charges']
X = first_no_outliers_data.drop(['charges'], axis=1)

# Already preprocessed data above

# =============================
# Create a few plots to ensure outlier identification is working propertly
# =============================
# Plot model without subgrouping
title_1 = 'w all new features'
model_name_1 = 'new_vars'
file_name_1 = '1_all_new_vars'
lin_reg_1, y_pred_1, het_results_1 = dh.fit_lr_model_results(X, y, title_1, save_img=False, ob_smoke_series=ob_smoke_series, 
                                                                   filename=file_name_1, save_dir=ml_models_output_dir,
                                                                   cmap=my_cmap, subgroup=True)


# Calculate Cook's distances
inf = influence(lin_reg_1)
(cooks, d) = inf.cooks_distance
cooks_cutoff = 4 / (len(cooks) - (X.shape[1] - 1) - 1)

outlier_df = X.copy()
outlier_df['cooks'] = cooks
outlier_df['outlier'] = outlier_df['cooks'] > cooks_cutoff
outlier_dict = {False:'no', True:'yes'}
outlier_df['outlier'] = outlier_df['outlier'].map(outlier_dict)

num_outliers = outlier_df[outlier_df['outlier'] == 'yes'].shape[0] # 85
perc_outliers = num_outliers / outlier_df.shape[0] # 0.0635
outlier_df['true_values'] = y
outlier_df['y_pred'] = y_pred_1
outlier_df['stud_resid'] = lin_reg_1.get_influence().resid_studentized_internal

# Visualize Cook's Distances
plt.title("Cook's Distance Plot")
plt.stem(range(len(cooks)), cooks, markerfmt=",")
plt.plot([0, len(cooks)], [cooks_cutoff, cooks_cutoff], color='darkblue', linestyle='--', label='4 / (N-k-1)')
plt.xlabel("Observation")
plt.ylabel("Cook's Distance")
plt.legend(title="Cook's Distance Cutoff")
#dh.save_image('cooks_dist_plot', models_output_dir)
plt.show()

# Plot outliers with respect to model results
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


# =============================
# Finalize DataFrame with first set of outliers removed
# =============================
outlier_df.columns
X.columns
first_no_outliers_data.columns

first_no_outliers_data['outlier'] = outlier_df['outlier']
first_no_outliers_data = first_no_outliers_data[first_no_outliers_data['outlier']=='no']
first_no_outliers_data = first_no_outliers_data.drop(['outlier'], axis=1)


# ==========================================================
# Dataset with new features and SECOND set of Cook's outliers removed
# ==========================================================

second_no_outliers_data = first_no_outliers_data.copy()

# Need to build model to calculate Cook's distances

# Separate target from predictors
y = second_no_outliers_data['charges']
X = second_no_outliers_data.drop(['charges'], axis=1)

# Already preprocessed data above

# =============================
# Create a few plots to ensure outlier identification is working propertly
# =============================
# Plot model without subgrouping
title_2 = 'w outliers removed x1'
model_name_2 = 'no_out_1'
file_name_2 = '2_no_out_1'
lin_reg_2, y_pred_2, het_results_2 = dh.fit_lr_model_results(X, y, title_2, save_img=False, ob_smoke_series=ob_smoke_series, 
                                                                   filename_unique=file_name_2, save_dir=ml_models_output_dir,
                                                                   cmap=my_cmap, subgroup=True)


# Calculate Cook's distances
inf = influence(lin_reg_2)
(cooks, d) = inf.cooks_distance
cooks_cutoff = 4 / (len(cooks) - (X.shape[1] - 1) - 1)

outlier_df = X.copy()
outlier_df['cooks'] = cooks
outlier_df['outlier'] = outlier_df['cooks'] > cooks_cutoff
outlier_dict = {False:'no', True:'yes'}
outlier_df['outlier'] = outlier_df['outlier'].map(outlier_dict)

num_outliers = outlier_df[outlier_df['outlier'] == 'yes'].shape[0] # 34
perc_outliers = num_outliers / outlier_df.shape[0] # 0.0271
outlier_df['true_values'] = y
outlier_df['y_pred'] = y_pred_2
outlier_df['stud_resid'] = lin_reg_2.get_influence().resid_studentized_internal

# Visualize Cook's Distances
plt.title("Cook's Distance Plot")
plt.stem(range(len(cooks)), cooks, markerfmt=",")
plt.plot([0, len(cooks)], [cooks_cutoff, cooks_cutoff], color='darkblue', linestyle='--', label='4 / (N-k-1)')
plt.xlabel("Observation")
plt.ylabel("Cook's Distance")
plt.legend(title="Cook's Distance Cutoff", loc="upper left")
#dh.save_image('cooks_dist_plot', models_output_dir)
plt.show()

# Plot outliers with respect to model results
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


# =============================
# Finalize DataFrame with SECOND set of outliers removed
# =============================
outlier_df.columns
X.columns
second_no_outliers_data.columns

second_no_outliers_data['outlier'] = outlier_df['outlier']
second_no_outliers_data = second_no_outliers_data[second_no_outliers_data['outlier']=='no']
second_no_outliers_data = second_no_outliers_data.drop(['outlier'], axis=1)


# ====================================================================================================================
# Machine Learning 
# ====================================================================================================================

# Back to preprocessing training and validation sets slightly differently, so starting feature engineering from the beginning
new_df = dataset.copy()

# Separate target from predictors
y = new_df['stroke']
X = new_df.drop(['stroke'], axis=1)

# MAKE SURE LIST OF CATEGORICAL AND NUMERICAL COLUMNS IS UPDATED BEFORE USING PREPROCESSING FUNCTION
reset_column_categories_create_format_dict()











# Create a few of the features before preprocessing
# Create ['age^2'] feature
new_features_data['age^2'] = np.power(new_features_data['age'], 2)
new_features_data = new_features_data.drop('age', axis=1)
numerical_cols = [col for col in numerical_cols if col not in ['age']]
numerical_cols.append('age^2')

# Create feature ['bmi_>=_30'] temporarily to create the ob_smoke_series using create_obese_smoker_category()
new_features_data['bmi_>=_30'] = new_features_data['bmi'] >= 30
obese_dict = {False:0, True:1}
new_features_data['bmi_>=_30'] = new_features_data['bmi_>=_30'].map(obese_dict)
categorical_cols.append('bmi_>=_30')
ob_smoke_series = create_obese_smoker_category_4(new_features_data)

# # Create ['smoker*obese'] feature
smoker_dict = {'no':0, 'yes':1}
new_features_data['smoker'] = new_features_data['smoker'].map(smoker_dict)
new_features_data['smoker*obese'] = new_features_data['smoker'] * new_features_data['bmi_>=_30']
categorical_cols.append('smoker*obese')

# Separate target from predictors
y = new_features_data['charges']
X = new_features_data.drop(['charges'], axis=1)

# Preprocess data
sm_processed_X = manual_preprocess_sm(X)

# Put dataset back together with preprocessed X
new_features_data = pd.concat([sm_processed_X, y], axis=1)

# Create ['smoker*obese'] feature
# smoker_dict = {'no':0, 'yes':1}
# new_features_data['smoker'] = new_features_data['smoker'].map(smoker_dict)
# new_features_data['smoker*obese'] = new_features_data['smoker'] * new_features_data['bmi_>=_30']

# Create ['bmi*smoker'] feature, specifically after preprocessing so it scales BMI properly first
new_features_data['bmi*smoker'] = new_features_data['smoker_1'] * new_features_data['bmi']

# Remove old variables from dataset: 'bmi_>=_30', 'bmi', and 'age' 
remove_var = ['bmi', 'bmi_>=_30_1']
new_features_data = new_features_data.drop(remove_var, axis=1)

# Remove old variables from categorical and numerical columns lists (relevant in case I used preprocessing again)
list_remove = ['bmi', 'age']
numerical_cols = [col for col in numerical_cols if col not in list_remove]
cont_cols = [col for col in cont_cols if col not in list_remove]
cont_cols_w_target = [col for col in cont_cols_w_target if col not in list_remove]
categorical_cols.remove('bmi_>=_30')












# ====================================================================================================================
# Decision Tree
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