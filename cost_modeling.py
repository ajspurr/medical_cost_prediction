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

from sklearn.linear_model import Ridge
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

# Initialize global variables
categorical_cols = []
numerical_cols = []

# Import my data science helper functions (relative dir based on project_dir)
my_module_dir = str(Path.resolve(Path('../my_ds_modules')))
sys.path.insert(0, my_module_dir)
import ds_helper as dh

# ====================================================================================================================
# Categorize and process features
# ====================================================================================================================

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

class MultiplyTransformer(BaseEstimator, TransformerMixin):
    """
    Allows use of a pipeline to create a new feature ('bmi*smoker') by multiplying 'bmi' feature by 'smoker' feature. 'bmi' needs to be scaled
    before multiplying by 'smoker' because there is no easy way to scale 'bmi*smoker' in such a way that the zeroes are ignored, as far as I 
    know. I initially tried to create a pipeline where 'bmi' was scaled in a ColumnTransformer, then 'bmi*smoker' was created using this
    transformer as a FeatureUnion. This didn't work when I added this tranformer (as a FeatureUnion) to the pipeline as a
    ColumnTransformer, because I had to include two columns ('bmi' and 'smoker') as parameters and ColumnTransformer requires you to return at least
    an equal number of columns to the number which you included as parameters. So 'bmi' and 'smoker' were duplicated in the final processed
    dataset. This also didn't work when I added this tranformer (as a FeatureUnion) directly to the pipeline (not as a ColumnTransformer)
    because the DataFrames were converted to numpy arrays at different points in the pipeline which I could not change, and as such 
    columns could not be accessed by their string. 
    
    So I ultimately decided to create this transformer which is meant to be used as a FeatureUnion and added to the pipeline as a
    ColumnTransformer that takes 'bmi' and 'smoker' as parameters. Those two columns will be preprocessed in this tranformer and NOT
    preprocessed in the usual categorical and numerical column transformers. 


    Parameters
    ----------
    BaseEstimator : sklearn.base.BaseEstimator
        BaseEstimator class.
    TransformerMixin : sklearn.base.TransformerMixin
        TransformerMixin class.


    Returns
    -------
    None.
        
    
    """
    
    def __init__(self):
        print('\n>>>>>init() called\n')
        self.means = None
        self.std = None
    
    def fit(self, X, y=None):
        print('\n>>>>>fit() called\n')
        
        # Scaling for 'bmi'
        self.ss = StandardScaler()
        self.ss.fit(X[['bmi']])
        
        # One-hot encoding for 'smoker'
        self.OH_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
        self.OH_encoder.fit(X[['smoker']])
        return self
    
    def transform(self, X, y=None):
        print('\n>>>>>transform() called\n')
       
        # Copy so as not to affect original data
        X_copy = X.copy()
        
        # Scaling 'bmi'
        X_copy['bmi'] = self.ss.transform(X_copy[['bmi']])
        
        #OH Encoding for 'smoker'
        X_copy.drop(['smoker'], axis=1, inplace=True)
        X_copy['smoker_1'] = pd.DataFrame(self.OH_encoder.transform(X[['smoker']]), index=X.index, columns=['smoker_1'])
        
        # Multiply 'bmi' and 'smoker' columns
        X_copy['bmi*smoker'] = X_copy['bmi'] * X_copy['smoker_1']

        print('\n>>>>>transform() finished\n')
        return X_copy


# Uses MultiplyTransformer, which takes 'bmi' and 'smoker' and transforms both of them and creates bmi*smoker feature
def create_pipeline_bmi_smoker(model_name, model): 
    # Use FeatureUnion preprocessing for 'bmi' and 'smoker' and creates 'bmi*smoker' feature
    union = FeatureUnion([('bmi_smoker', MultiplyTransformer())])
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()), 
        ('scale', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False))
    ])
    
    # Bundle preprocessing for union, numerical and categorical data
    preprocessor = ColumnTransformer(transformers=[
            ('dif', union, ['bmi', 'smoker']),
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
    ])
    
    # Bundle preprocessor and model
    my_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        (model_name, model)
    ])

    return my_pipeline

# Original pipeline creation. No feature engineering
def create_pipeline(model_name, model, num_cols, cat_cols):
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
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
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

def manual_preprocess_bmi_smoker(X_train, X_valid):
    
    X_train_num = X_train[numerical_cols]
    X_valid_num = X_valid[numerical_cols]
    
    X_train_cat = X_train[categorical_cols]
    X_valid_cat = X_valid[categorical_cols]
    
    # =============================
    # Numerical preprocessing
    # =============================
    
    # Scaling
    ss = StandardScaler()
    scaled_X_train_num = pd.DataFrame(ss.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    scaled_X_valid_num = pd.DataFrame(ss.transform(X_valid_num), columns=X_valid_num.columns, index=X_valid_num.index)
    
    # Create ['bmi*smoker'] feature, specifically after scaling so it scales BMI properly first
    # scaled_X_train_num['bmi*smoker'] = X_train_cat['smoker'] * scaled_X_train_num['bmi']
    # scaled_X_valid_num['bmi*smoker'] = X_valid_cat['smoker'] * scaled_X_valid_num['bmi']
    bmi_smoker_train = pd.Series(X_train_cat['smoker'] * scaled_X_train_num['bmi'], name='bmi*smoker')
    bmi_smoker_valid = pd.Series(X_valid_cat['smoker'] * scaled_X_valid_num['bmi'], name='bmi*smoker')
    
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
    # X_train_processed = pd.concat([scaled_X_train_num, OH_cols_train], axis=1)
    # X_valid_processed = pd.concat([scaled_X_valid_num, OH_cols_valid], axis=1)
    X_train_processed = pd.concat([scaled_X_train_num, OH_cols_train, bmi_smoker_train], axis=1)
    X_valid_processed = pd.concat([scaled_X_valid_num, OH_cols_valid, bmi_smoker_valid], axis=1)
    
    return X_train_processed, X_valid_processed

# Preprocessing of all independent variable data together (no train/test split) for use with statmodels (sm) data analysis
def manual_preprocess_sm(X, num_cols, cat_cols):
    # =============================
    # Numerical preprocessing
    # =============================
    X_num = X[num_cols]
        
    # Scaling
    ss = StandardScaler()
    scaled_X_num = pd.DataFrame(ss.fit_transform(X_num), columns=X_num.columns, index=X_num.index)
    
    # =============================
    # Categorical preprocessing
    # =============================
    X_cat = X[cat_cols]
        
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

# =======================================================================================
# Dataset with new features I created in cost_lin_reg.py and with their source features removed
# =======================================================================================

new_features_data = dataset.copy()

# Separate target from predictors
y = new_features_data['charges']
X = new_features_data.drop(['charges'], axis=1)

# Create most of the features before preprocessing

# Create ['age^2'] feature
# age_sq = np.power(X['age'], 2)
# add_feature('age^2', age_sq, 'n', X)
# remove_feature('age', 'n', X)
X['age'] = np.power(X['age'], 2)
X.rename(columns={'age':'age^2'}, inplace=True)

# Create feature ['bmi_>=_30'] (maybe temporarily) to create the ob_smoke_series using create_obese_smoker_category()
# and to use for obese*smoker feature later
# bmi_30 = X['bmi'] >= 30
# obese_dict = {False:0, True:1}
# bmi_30 = bmi_30.map(obese_dict)
# add_feature('bmi_>=_30', bmi_30, 'c', X)
# ob_smoke_series = create_obese_smoker_category_4(X)
X['bmi_>=_30'] = X['bmi'] >= 30
obese_dict = {False:0, True:1}
X['bmi_>=_30'] = X['bmi_>=_30'].map(obese_dict)
ob_smoke_series = create_obese_smoker_category_4(X)

# Create ['smoker*obese'] feature
# smoker_dict = {'no':0, 'yes':1}
# X['smoker'] = X['smoker'].map(smoker_dict)
# smoke_obese = X['smoker'] * X['bmi_>=_30']
# add_feature('smoker*obese', smoke_obese, 'c', X)
smoker_dict = {'no':0, 'yes':1}
X['smoker'] = X['smoker'].map(smoker_dict)
X['smoker*obese'] = X['smoker'] * X['bmi_>=_30']


# I originally wanted to preprocess the data before adding the ['bmi*smoker'] feature. If I added the feature first,
# then tried to scale it, the zeros would be included in the scaling (not sure how to prevent that). 
# And the way skealrn cross-validation works is that the preprocessing has to be included in the pipeline, since
# each fold will have a different test/train split. So I created my own custom transformer and pipeline to 
# create the ['bmi*smoker'] feature.


# =======================================================================================
# Test new custom transformer/pipeline against my manual preprocess function
# =======================================================================================

# Compare pipeline preprocess to manual preprocessing, so I can make sure everything is happening correctly
y = new_features_data['charges']

# use the X with 3 out of 4 new features created above
# For now I'll keep the 'bmi >= 30' feature
X_pipe_test = X.copy() 

# MAKE SURE LIST OF CATEGORICAL AND NUMERICAL COLUMNS IS UPDATED BEFORE USING PREPROCESSING FUNCTION
reset_column_categories_create_format_dict(X)
remove_from_numerical_cols = ['smoker', 'bmi_>=_30', 'smoker*obese']
numerical_cols = [col for col in numerical_cols if col not in remove_from_numerical_cols]
categorical_cols.extend(remove_from_numerical_cols)
print_num_cat_cols()

# Train/valid split
X_train, X_valid, y_train, y_valid = train_test_split(X_pipe_test, y, train_size=0.8, test_size=0.2, random_state=15)

# Manual preprocess
X_train_processed, X_valid_processed = manual_preprocess_bmi_smoker(X_train, X_valid)

# Remove bmi and smoker from numerical and cateorical columns lists for custom pipeline preprocessing
numerical_cols.remove('bmi')
categorical_cols.remove('smoker')
print_num_cat_cols()

# Create pipeline with custom transform to create ['bmi*smoker'] feature
my_pipeline = create_pipeline_bmi_smoker('LR', LinearRegression())
results = my_pipeline.fit(X_train, y_train)
y_pred = my_pipeline.predict(X_valid)

# View pipeline-transformed data
new_pipe = my_pipeline.named_steps['preprocessor']
pipeline_transformed_train_data = pd.DataFrame(new_pipe.transform(X_train))
pipeline_transformed_valid_data = pd.DataFrame(new_pipe.transform(X_valid))

X_train_processed[['bmi', 'smoker_1', 'bmi*smoker', 'age^2', 'children']]
X_train_processed[['sex_male', 'region_northwest', 'region_southeast']]
X_train_processed[['region_southwest', 'bmi_>=_30_1', 'smoker*obese_1']]

X_valid_processed[['bmi', 'smoker_1', 'bmi*smoker', 'age^2', 'children']]
X_valid_processed[['sex_male', 'region_northwest', 'region_southeast']]
X_valid_processed[['region_southwest', 'bmi_>=_30_1', 'smoker*obese_1']]

# Pipeline vs. manually preprocessed data looks identical 

# I can compare their model performance if there is still concern that the pipeline isn't working properly


# =======================================================================================
# Dataset with new features and first set of Cook's outliers removed
# =======================================================================================
# Copy to new dataset
first_no_outliers_X = X.copy()
first_no_outliers_X.columns

# Categorize num and cat column names
fno_num_cols = ['age^2', 'bmi', 'children']
fno_cat_cols = ['sex', 'smoker', 'region', 'bmi_>=_30', 'smoker*obese']

# Preprocess data and add ['smoker*bmi']
# The model needs to include the same features that were originally used to identify outliers
first_no_outliers_X = manual_preprocess_sm(first_no_outliers_X, fno_num_cols, fno_cat_cols)
first_no_outliers_X['smoker*bmi'] = first_no_outliers_X['smoker_1']  * first_no_outliers_X['bmi']
first_no_outliers_X.columns

# Need to build model to calculate Cook's distances

# Already preprocessed data above

# =============================
# Create a few plots to ensure outlier identification is working propertly
# =============================
# Plot model without subgrouping
title_1 = 'w all new features'
model_name_1 = 'new_vars'
file_name_1 = '1_all_new_vars'
lin_reg_1, y_pred_1, het_results_1 = dh.fit_lr_model_results(first_no_outliers_X, y, title_1, save_img=False, grouping=ob_smoke_series, 
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

num_outliers = outlier_df[outlier_df['outlier'] == 'yes'].shape[0] # 85 -> 88
perc_outliers = num_outliers / outlier_df.shape[0] # 0.0635 -> 0.0658
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
first_no_outliers_X.columns

first_no_outliers_data = pd.concat([X, y, outlier_df['outlier']], axis=1)
first_no_outliers_data = first_no_outliers_data[first_no_outliers_data['outlier']=='no']
first_no_outliers_data = first_no_outliers_data.drop(['outlier'], axis=1)


# ==========================================================
# Dataset with new features and SECOND set of Cook's outliers removed
# ==========================================================

second_no_outliers_data = first_no_outliers_data.copy()

# Need to build model to calculate Cook's distances

# Separate target from predictors
no_out_2_y = second_no_outliers_data['charges']
no_out_2_X = second_no_outliers_data.drop(['charges'], axis=1)

# Categorize num and cat column names
sno_num_cols = ['age^2', 'bmi', 'children']
sno_cat_cols = ['sex', 'smoker', 'region', 'bmi_>=_30', 'smoker*obese']

# Preprocess data and add ['smoker*bmi']
# The model needs to include the same features that were originally used to identify outliers
no_out_2_X = manual_preprocess_sm(no_out_2_X, sno_num_cols, sno_cat_cols)
no_out_2_X['smoker*bmi'] = no_out_2_X['smoker_1']  * no_out_2_X['bmi']


# =============================
# Create a few plots to ensure outlier identification is working propertly
# =============================
# Plot model without subgrouping
title_2 = 'w outliers removed x1'
model_name_2 = 'no_out_1'
file_name_2 = '2_no_out_1'
lin_reg_2, y_pred_2, het_results_2 = dh.fit_lr_model_results(no_out_2_X, no_out_2_y, title_2, save_img=False, grouping=ob_smoke_series, 
                                                                   filename=file_name_2, save_dir=ml_models_output_dir,
                                                                   cmap=my_cmap, subgroup=True)


# Calculate Cook's distances
inf = influence(lin_reg_2)
(cooks, d) = inf.cooks_distance
cooks_cutoff = 4 / (len(cooks) - (X.shape[1] - 1) - 1)

outlier_df = no_out_2_X.copy()
outlier_df['cooks'] = cooks
outlier_df['outlier'] = outlier_df['cooks'] > cooks_cutoff
outlier_dict = {False:'no', True:'yes'}
outlier_df['outlier'] = outlier_df['outlier'].map(outlier_dict)

num_outliers = outlier_df[outlier_df['outlier'] == 'yes'].shape[0] # 34 -> 31
perc_outliers = num_outliers / outlier_df.shape[0] # 0.0271 ->  0.0248
outlier_df['true_values'] = no_out_2_y
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
no_out_2_X.columns
second_no_outliers_data.columns

second_no_outliers_data['outlier'] = outlier_df['outlier']
second_no_outliers_data = second_no_outliers_data[second_no_outliers_data['outlier']=='no']
second_no_outliers_data = second_no_outliers_data.drop(['outlier'], axis=1)


# ====================================================================================================================
# Model Building
# ====================================================================================================================

# Scoring list for cross validation
scoring_list = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'max_error']

# Function for formatting model performance metrics
def average_cv_scores(score_dict, round=3):
    avg_cv_scores = {}
    for key, value in score_dict.items():
        avg_cv_scores[key] = np.round(np.mean(value), round)
    
    avg_cv_scores.pop('fit_time')
    avg_cv_scores.pop('score_time')
    return avg_cv_scores

# =============================
# Separate target from predictors
# =============================
# Original data with no feature engineering
y_orig = dataset['charges']
X_orig = dataset.drop(['charges'], axis=1)

# X and y already separated above. X includes 3/4 of the new features, and kept 'bmi_>=_30', but does not include 
# [smoker*bmi] as it needs to be created in the pipeline
y
X

# First no outlier X and y
fno_y = first_no_outliers_data['charges']
fno_X = first_no_outliers_data.drop(['charges'], axis=1)

# Second no outlier X and y
sno_y = second_no_outliers_data['charges']
sno_X = second_no_outliers_data.drop(['charges'], axis=1)

# MAKE SURE LIST OF CATEGORICAL AND NUMERICAL COLUMNS IS UPDATED BEFORE USING PREPROCESSING FUNCTION
# 'bmi' and 'smoker' shouldn't be included in the lists
print_num_cat_cols()
#reset_column_categories_create_format_dict()

# ====================================================================================================================
# Linear Regression
# ====================================================================================================================

# Keep track of performance
lr_results_list = []

# No new features
num_cols = ['age', 'bmi', 'children']
cat_cols = ['sex', 'smoker', 'region']

lin_reg0 = LinearRegression()
lr_pipeline0 = create_pipeline('LR', lin_reg0, num_cols, cat_cols)
lr_scores0 = cross_validate(lr_pipeline0, X_orig, y_orig, cv=10, scoring=scoring_list)
lr_results_list.append(average_cv_scores(lr_scores0))


# New features with no outliers removed
lin_reg1 = LinearRegression()
lr_pipeline1 = create_pipeline_bmi_smoker('LR', lin_reg1)
lr_scores1 = cross_validate(lr_pipeline1, X, y, cv=10, scoring=scoring_list)
lr_results_list.append(average_cv_scores(lr_scores1))

# First outliers removed
lin_reg2 = LinearRegression()
lr_pipeline2 = create_pipeline_bmi_smoker('LR', lin_reg2)
lr_scores2 = cross_validate(lr_pipeline2, fno_X, fno_y, cv=10, scoring=scoring_list)
lr_results_list.append(average_cv_scores(lr_scores2))

# Second outliers removed
lin_reg3 = LinearRegression()
lr_pipeline3 = create_pipeline_bmi_smoker('LR', lin_reg3)
lr_scores3 = cross_validate(lr_pipeline3, sno_X, sno_y, cv=10, scoring=scoring_list)
lr_results_list.append(average_cv_scores(lr_scores3))

# Keep track of performance
lr_results_df = pd.DataFrame(lr_results_list, index=['orig', 'new_feat', 'no_out1', 'no_out2']).T




# ====================================================================================================================
# Ridge Regression
# ====================================================================================================================
# Keep track of performance
rr_results_list = []
rr_model_name = 'RR'

# =======================================================================================
# No new features
# =======================================================================================
num_cols = ['age', 'bmi', 'children']
cat_cols = ['sex', 'smoker', 'region']

rid_reg0 = Ridge(alpha=1.0, random_state=15)
rr_pipeline0 = create_pipeline(rr_model_name, rid_reg0, num_cols, cat_cols)
rr_scores0 = cross_validate(rr_pipeline0, X_orig, y_orig, cv=10, scoring=scoring_list)
rr_results_list.append(average_cv_scores(rr_scores0))

# ==========================================================
# Hyperparameter tuning
# ==========================================================

# Determine hyperparameters to be tuned
rr_pipeline0.get_params()
rid_reg0.get_params()
rr_parameters = {rr_model_name + '__alpha': range(0, 10, 1)}

# Specify other GridSearchCV parameters
scoring = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'max_error']
refit = 'r2'
n_jobs = 10
cv = 10
verbose = 5

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=rr_pipeline0, param_grid=rr_parameters, scoring=scoring, refit=refit, n_jobs=n_jobs, cv=cv, verbose=verbose)

# Test/train split
X_train, X_valid, y_train, y_valid = train_test_split(X_orig, y_orig, train_size=0.8, test_size=0.2, random_state=15)

# Hyperparameter tuning using GridSearchCV
grid_search.fit(X_train, y_train)

# Access GridSearch results, store in df
gs_results = grid_search.cv_results_
gs_results_df = pd.DataFrame(gs_results)

# Extract meaningful results and reformat new df
relevant_gs_results_df = gs_results_df[['param_RR__alpha', 'mean_test_r2', 'rank_test_r2', 
                                        'mean_test_neg_root_mean_squared_error',
                                        'rank_test_neg_root_mean_squared_error',
                                        'mean_test_neg_mean_absolute_error',
                                        'rank_test_neg_mean_absolute_error',
                                        'mean_test_neg_median_absolute_error', 
                                        'rank_test_neg_median_absolute_error',
                                        'mean_test_max_error',
                                        'rank_test_max_error']]

relevant_gs_results_df = relevant_gs_results_df.rename(columns={'param_RR__alpha':'alpha',
                                                                'mean_test_r2':'r2', 'rank_test_r2':'r2_rank', 
                                                                'mean_test_neg_root_mean_squared_error':'rmse',
                                                                'rank_test_neg_root_mean_squared_error':'rmse_rank',
                                                                'mean_test_neg_mean_absolute_error':'mae',
                                                                'rank_test_neg_mean_absolute_error':'mae_rank',
                                                                'mean_test_neg_median_absolute_error':'med_ae', 
                                                                'rank_test_neg_median_absolute_error':'med_ae_rank',
                                                                'mean_test_max_error':'me',
                                                                'rank_test_max_error':'me_rank'})



# Graph performance metrics with each hyperparameter
list_metric_cols = ['r2', 'rmse', 'mae', 'med_ae', 'me']

for metric in list_metric_cols:
    plt.plot(relevant_gs_results_df['alpha'], relevant_gs_results_df[metric], marker='o', markersize=4)
    plt.title(metric)
    plt.grid()
    plt.show()

# plt.xlabel() and plt.ylabel() not working, this code outputs a list of callable functions in plt and 
# does not include them for some reason
object_methods = [method_name for method_name in dir(plt) if callable(getattr(plt, method_name))]












# Access results of best estimator
best_estimator_row = relevant_gs_results_df.loc[relevant_gs_results_df['r2_rank'] == 1]

# Gets the index of the best r2 score (to be used later as well)
#best_estimator_r2_index = best_estimator_row.index[0]
#best_estimator_r2_index = np.nonzero(gs_results['rank_test_r2'] == 1)[0][0]

# Best r2 score, which was used to select the best estimator
best_estimator_r2 = best_estimator_row['r2']
#best_estimator_r2 = gs_results['mean_test_r2'][best_estimator_r2_index]

# To get the mae score of the best estimator, use the same index as the best r2, since that corresponds to the same estimator
#best_estimator_mae = gs_results['mean_test_neg_median_absolute_error'][best_estimator_r2_index]
best_estimator_mae = best_estimator_row['mae']

# Using optimal model from GridSearch results, fit and run model again so that I can run my results functions
pipeline_gs = grid_search.best_estimator_
pipeline_gs.fit(X_train, y_train)
y_pred_gs = pipeline_gs.predict(X_valid)

# ====================================================================================================================
# Lasso Regression
# ====================================================================================================================


# ====================================================================================================================
# ElasticNet?
# ====================================================================================================================


# ====================================================================================================================
# Decision Tree
# ====================================================================================================================

