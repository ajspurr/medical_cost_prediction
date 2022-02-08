import sys
import numpy as np
import pandas as pd
from os import chdir
from pathlib import PureWindowsPath, Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
# Data preprocessing function via pipeline
# ====================================================================================================================

# def create_pipeline(model_name, model):
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
#             ('num', numerical_transformer, numerical_cols),
#             ('cat', categorical_transformer, categorical_cols)
#     ])
    
#     # Bundle preprocessor and model
#     my_pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         (model_name, model)
#     ])
#     return my_pipeline


# class MultiplyTransformer_old(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         print('\n>>>>>init() called\n')
#         self.means = None
#         self.std = None
    
#     def fit(self, X, y=None):
#         # Save original (before scaling) 'bmi' mean and std
#         print('\n>>>>>fit() called\n')
#         self.means = X.bmi.to_numpy().mean(axis=0)
#         #print(f"mean bmi = {self.means}")
#         self.std = X.bmi.to_numpy().std(axis=0)
#         #print(f"std bmi = {self.std}")
#         return self
    
#     def transform(self, X, y=None):
#         print('\n>>>>>transform() called\n')
#         print(type(X))
#         # Multiply 'bmi' and 'smoker' columns
#         bmi_smoker = X['bmi'] * X['smoker']
#         print(type(bmi_smoker))
#         # Scale bmi_smoker to original 'bmi' mean and std
#         for index, value in enumerate(bmi_smoker):
#             if value != 0:
#                 # Scale values
#                 bmi_smoker.iat[index] = (value - self.means) / self.std
        
#         # Copy so as not to affect original data
#         X_copy = X.copy()
#         print(type(X_copy))
#         X_copy['bmi*smoker'] = bmi_smoker
#         print(X_copy)
#         print('\n>>>>>transform() finished\n')
#         return X_copy



class MultiplyTransformer(BaseEstimator, TransformerMixin):
    """
    Allows use of a pipeline to create a new feature ('bmi*smoker') by multiplying 'bmi' feature by 'smoker' feature. 'bmi' needs to be scaled
    before multiplying by 'smoker' because there is no easy way to scale 'bmi*smoker' in such a way that the zeroes are ignored, as far as I 
    know. I initially tried to create a pipeline where 'bmi' was scaled in a ColumnTransformer, then 'bmi*smoker' was created using this
    transformer as a FeatureUnion. This didn't work when I added this tranformer (as a FeatureUnion) to the pipeline as a
    ColumnTransformer, because I had to include two columns ('bmi' and 'smoker') and ColumnTransformer requires you to return at least
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


# ====================================================================================================================
# Data preprocessing function without using pipeline
# ====================================================================================================================

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
    bmi_smoker_train = pd.Series(X_train_cat['smoker'] * scaled_X_train_num['bmi'], name='bmi*smoker')
    bmi_smoker_valid = pd.Series(X_valid_cat['smoker'] * scaled_X_valid_num['bmi'], name='bmi*smoker')
        
    # =============================
    # Categorical preprocessing
    # =============================
    # One-hot encoding
    OH_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_cat), index=X_train_cat.index, columns=OH_encoder.get_feature_names_out())
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid_cat), index=X_valid_cat.index, columns=OH_encoder.get_feature_names_out())
    
    # Add preprocessed categorical columns back to preprocessed numerical columns
    X_train_processed = pd.concat([scaled_X_train_num, OH_cols_train, bmi_smoker_train], axis=1)
    X_valid_processed = pd.concat([scaled_X_valid_num, OH_cols_valid, bmi_smoker_valid], axis=1)
    
    return X_train_processed, X_valid_processed


# ===================================================
# Try with all features 
# ===================================================

reset_column_categories_create_format_dict(dataset.drop('charges', axis=1))

my_df = dataset.copy()
y = my_df['charges']
X = my_df. drop(['charges'], axis=1)
reset_column_categories_create_format_dict(X)
print_num_cat_cols()

# Convert smoker to 0s and 1s
smoker_dict = {'no':0, 'yes':1}
X['smoker'] = X['smoker'].map(smoker_dict)

# Train/valid split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# Manual preprocess
X_train_processed, X_valid_processed = manual_preprocess_bmi_smoker(X_train, X_valid)

# Remove bmi and smoker from numerical and cateorical columns lists for custom pipeline preprocessing
numerical_cols.remove('bmi')
categorical_cols.remove('smoker')
print_num_cat_cols()

my_pipeline6 = create_pipeline_bmi_smoker('LR', LinearRegression())
results6 = my_pipeline6.fit(X_train, y_train)
y_pred6 = my_pipeline6.predict(X_valid)
# See intermediate pipeline data
new_pipe6 = my_pipeline6.named_steps['preprocessor']
pipeline_transformed_train_data6 = pd.DataFrame(new_pipe6.transform(X_train))
pipeline_transformed_valid_data6 = pd.DataFrame(new_pipe6.transform(X_valid))

X_train_processed[['bmi', 'smoker_1', 0, 'age', 'children']]
X_train_processed[['sex_male', 'region_northwest', 'region_southeast', 'region_southwest']]

X_valid_processed[['bmi', 'smoker_1', 0, 'age', 'children']]
X_valid_processed[['sex_male', 'region_northwest', 'region_southeast', 'region_southwest']]


