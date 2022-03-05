import sys
import numpy as np
import pandas as pd
from os import chdir
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import PureWindowsPath, Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate

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

# Made into function for debugging
def create_column_categories(fxn_X):
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
    
    # Separate categorical and numerical features
    numerical_cols = [cname for cname in fxn_X.columns if not fxn_X[cname].dtype == "object"]
    categorical_cols = [cname for cname in fxn_X.columns if fxn_X[cname].dtype == "object"]
    
    print(f"Numerical Columns: {numerical_cols}")
    print(f"Categorical Columns: {categorical_cols}")
    
    return numerical_cols, categorical_cols

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
        return

    
    def fit(self, X, y=None):      
        # Scaling for 'bmi'
        self.ss = StandardScaler()
        self.ss.fit(X[['bmi']])
        
        # One-hot encoding for 'smoker'
        self.OH_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
        self.OH_encoder.fit(X[['smoker']])
        return self
    
    def transform(self, X, y=None):       
        # Copy so as not to affect original data
        X_copy = X.copy()
        
        # Scaling 'bmi'
        X_copy['bmi'] = self.ss.transform(X_copy[['bmi']])
        
        #OH Encoding for 'smoker'
        X_copy.drop(['smoker'], axis=1, inplace=True)
        X_copy['smoker_1'] = pd.DataFrame(self.OH_encoder.transform(X[['smoker']]), index=X.index, columns=['smoker_1'])
        
        # Multiply 'bmi' and 'smoker' columns
        X_copy['bmi*smoker'] = X_copy['bmi'] * X_copy['smoker_1']

        #print('\n>>>>>transform() finished\n')
        return X_copy


# Uses MultiplyTransformer, which takes 'bmi' and 'smoker' and transforms both of them and creates bmi*smoker feature
def create_pipeline_bmi_smoker(model_name, model, num_cols, cat_cols): 
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
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
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
def manual_preprocess(X_train, X_valid, numerical_cols, categorical_cols):
    # =============================
    # Numerical preprocessing
    # =============================
    X_train_num = X_train[numerical_cols]
    X_valid_num = X_valid[numerical_cols]
       
    # Scaling
    ss = StandardScaler()
    scaled_X_train_num = pd.DataFrame(ss.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    scaled_X_valid_num = pd.DataFrame(ss.transform(X_valid_num), columns=X_valid_num.columns, index=X_valid_num.index)
    
    # =============================
    # Categorical preprocessing
    # =============================
    X_train_cat = X_train[categorical_cols]
    X_valid_cat = X_valid[categorical_cols]
    
    # One-hot encoding
    OH_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_cat), index=X_train_cat.index, columns=OH_encoder.get_feature_names_out())
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid_cat), index=X_valid_cat.index, columns=OH_encoder.get_feature_names_out())
    
    # Add preprocessed categorical columns back to preprocessed numerical columns
    X_train_processed = pd.concat([scaled_X_train_num, OH_cols_train], axis=1)
    X_valid_processed = pd.concat([scaled_X_valid_num, OH_cols_valid], axis=1)
    
    return X_train_processed, X_valid_processed

def manual_preprocess_bmi_smoker(X_train, X_valid, numerical_cols, categorical_cols):
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


# ====================================================================================================================
# Feature engineering from cost_lin_reg.py
# ====================================================================================================================
# Create formatted columns dictionary in dh module
dh.create_formatted_cols_dict(dataset.columns)
dh.update_formatted_cols('bmi', 'BMI')

# Copy dataset to leave original unaffected
new_features_data = dataset.copy()

# Separate target from predictors
y = new_features_data['charges']
X = new_features_data.drop(['charges'], axis=1)

# Create most of the features before preprocessing

# Create ['age^2'] feature
X['age'] = np.power(X['age'], 2)
X.rename(columns={'age':'age^2'}, inplace=True)

# Create feature ['bmi_>=_30'] (temporarily) to create the ob_smoke_series using create_obese_smoker_category()
# and to use for obese*smoker feature later
X['bmi_>=_30'] = X['bmi'] >= 30
obese_dict = {False:0, True:1}
X['bmi_>=_30'] = X['bmi_>=_30'].map(obese_dict)
ob_smoke_series = create_obese_smoker_category(X)

# Create ['smoker*obese'] feature
smoker_dict = {'no':0, 'yes':1}
X['smoker'] = X['smoker'].map(smoker_dict)
X['smoker*obese'] = X['smoker'] * X['bmi_>=_30']


# I originally wanted to preprocess the data before adding the ['bmi*smoker'] feature. If I added the feature first,
# then tried to scale it, the zeros would be included in the scaling (not sure how to prevent that). 
# And the way sklearn cross-validation works is that the preprocessing has to be included in the pipeline, since
# each fold will have a different test/train split. So I created my own custom transformer and pipeline to 
# create the ['bmi*smoker'] feature.


# ====================================================================================================================
# Model Building
# ====================================================================================================================

# =======================================================================================
# Model Building Functions
# =======================================================================================


def average_cv_scores(cv_results, score_abbv_dict, index_neg, round=3):  
    """
    Function for formatting model performance scores returned from sklearn cross_validate(). Takes the 
    mean of each score.

    Parameters
    ----------
    cv_results : Dict
        The dictionary of results that is returned from cross_validate().
    score_abbv_dict : Dict
        Values are the names of the cross_validate() scores, keys are the abbreviated names of the scores.
    index_neg : list
        List of integers, each representing the index of the score in score_abbv_dict that should be 
        multipled by -1 (as they are returned as negative numbers).
    round : integer, optional
        How to round the results. The default is 3.

    Returns
    -------
    return_df : DataFrame
        Single-column dataframe where the index is the abbreviated name of the cross_val score and the values 
        are the average values.

    """
    scores_list = ['test_' + score_name for score_name in list(score_abbv_dict.keys())]
    scores_rename_list = list(score_abbv_dict.values())
       
    avg_cv_scores = {}
    for i, key in enumerate(scores_list):
        avg_cv_scores[key] = np.round(np.mean(cv_results[key]), round)
        if i in index_neg:
            avg_cv_scores[key] = avg_cv_scores[key] * -1
    
    # Convert result dict to df
    return_df = pd.DataFrame.from_dict(avg_cv_scores, orient='index')    
    return_df.index = scores_rename_list
    return return_df


def model_scores_test_data(estimators, X_test, y_test, round=3):
    """
    Loops through estimators which have already been fit on training data, uses estimators to predict y and 
    calculates multiple model performance scores, returns scores as a dataframe

    Parameters
    ----------
    estimators : list of sklearn Estimators
        List of estimators (usually models or pipelines) that have already been fit on training data. Will be used
        to predict target.
    X_test : DataFrame
        The features of the test data.
    y_test : Series
        The target of the test data.
    round : Integer, optional
        How many decimals places to round results. The default is 3.

    Returns
    -------
    return_df : DataFrame
        DataFrame containing the model performance scores on the test data.

    """
    # Using a defaultdict allows you to create a dictionary where the values are lists, which you can 
    # access directly by their keys and append values to. This doesn't work on a normal dictionary.
    test_results_dict = defaultdict(list)
    
    for estimator in estimators:
        y_pred = estimator.predict(X_test)
        test_results_dict['r2'].append(r2_score(y_test, y_pred))
        test_results_dict['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_results_dict['mae'].append(mean_absolute_error(y_test, y_pred))
        test_results_dict['mape'].append(mean_absolute_percentage_error(y_test, y_pred))
        test_results_dict['med_ae'].append(median_absolute_error(y_test, y_pred))
        test_results_dict['me'].append(max_error(y_test, y_pred))
    
    # Each value in test_results_dict is a list of scores, one for each estimator, this loop averages each score
    scores = ['r2', 'rmse', 'mae', 'mape', 'med_ae', 'me']
    test_results_dict2 = defaultdict(list)
    for score in scores:
        test_results_dict2[score] = np.round(np.mean(test_results_dict[score]), round)
    
    # Convert result dict to df
    return_df = pd.DataFrame.from_dict(dict(test_results_dict2), orient='index')    
    
    return return_df

# Extracts relevant metrics from GridSearch cv_results_ and returns as a dataframe
# metric_dict keys are the names of the metrics in GridSearch cv_results_ and 
# metric_dict values are the updated names of the metrics (usually shorter strings)
def gs_relevant_results_to_df(gs_results, metric_dict, negative_list):
    # metric_dict =  {'param_RR__alpha':'alpha', 'mean_test_r2':'r2', 'rank_test_r2':'r2_rank',
    #                 'mean_test_neg_root_mean_squared_error':'rmse', 'rank_test_neg_root_mean_squared_error':'rmse_rank',
    #                 'mean_test_neg_mean_absolute_error':'mae', 'rank_test_neg_mean_absolute_error':'mae_rank',
    #                 'mean_test_neg_median_absolute_error':'med_ae', 'rank_test_neg_median_absolute_error':'med_ae_rank',
    #                 'mean_test_max_error':'me','rank_test_max_error':'me_rank'}
    
    gs_results_df = pd.DataFrame(gs_results)
    relevant_gs_results_df = gs_results_df[metric_dict.keys()]
    relevant_gs_results_df = relevant_gs_results_df.rename(columns=metric_dict)
    
    # Multiply negative metrics by -1
    for negative_metric in negative_list:
        relevant_gs_results_df[negative_metric] = relevant_gs_results_df[negative_metric] * -1
    
    return relevant_gs_results_df

# Takes parameters and performs cross-validation. Returns mean cv results and results when model applied to remaining test data
def cv_results(fxn_pipeline, X, y, score_abbv_dict, index_neg, cv=10, return_estimator=True):  
    # Test/train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)
    
    # Cross-validation
    cv_scores = cross_validate(fxn_pipeline, X_train, y_train, scoring=list(score_abbv_dict.keys()), cv=cv, return_estimator=return_estimator)
    
    # Cross-validation performance scores (means)
    avg_cv_scores_df = average_cv_scores(cv_scores, score_abbv_dict, index_neg)
    
    # Use CV model on test data
    cv_test_scores_df = model_scores_test_data(cv_scores['estimator'], X_test, y_test)
    
    return avg_cv_scores_df, cv_test_scores_df


# Takes parameters and performs hyperparamter tuning using GridSearchCV. 
# Returns best hyperparameter, mean cv results of best estimator, and results when best estimator applied to remaining test data
def perform_grid_search(pipeline, X_train, y_train, scoring_list, param_grid, plot_title, cv=10, refit='r2', verbose=5, n_jobs=-1):    
    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring_list, 
                               refit=refit, n_jobs=n_jobs, cv=cv, verbose=verbose)
       
    # Hyperparameter tuning using GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Access GridSearch results
    gs_results = grid_search.cv_results_
    
    return grid_search, gs_results


# Takes parameters and performs hyperparamter tuning using GridSearchCV. 
# Returns best hyperparameter, mean cv results of best estimator, and results when best estimator applied to remaining test data
def ridge_gs_results(pipeline, X, y, scoring_list, param_grid, plot_title, cv=10, refit='r2', verbose=5, n_jobs=-1):    
    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring_list, 
                               refit=refit, n_jobs=n_jobs, cv=cv, verbose=verbose)
    
    # Test/train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)
    
    # Hyperparameter tuning using GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Access GridSearch results
    gs_results = grid_search.cv_results_
    
    # metric_dict keys are the names of the metrics in GridSearch cv_results_ and 
    # values are the updated names of the metrics (usually shorter strings)
    # This is specific to ridge regression gs as the only parameter it includes is alpha
    metric_dict =  {'param_RR__alpha':'alpha', 'mean_test_r2':'r2', 'rank_test_r2':'r2_rank',
                    'mean_test_neg_root_mean_squared_error':'rmse', 'mean_test_neg_mean_absolute_error':'mae', 
                    'mean_test_neg_median_absolute_error':'med_ae', 'mean_test_max_error':'me'}
    
    # List of metrics that return negative, so will be multiplied by -1
    negative_list = ['rmse', 'mae', 'med_ae', 'me']
    
    # Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
    relevant_gs_results_df = gs_relevant_results_to_df(gs_results, metric_dict, negative_list)
    
    # Visualize change in r2 with each hyperparameter
    plt.plot(relevant_gs_results_df['alpha'], relevant_gs_results_df['r2'], marker='o', markersize=4)
    plt.ylabel('r2')
    plt.xlabel('alpha')
    plt.title(f'GS Results ({plot_title})')
    plt.grid()
    plt.show()
    
    # Access metrics of best estimator, this is based on 'refit' parameter which defaults to 'r2'
    best_estimator_row = relevant_gs_results_df.loc[relevant_gs_results_df['r2_rank'] == 1]
    
    # Hyperparameter of best estimator 
    best_estimator_alpha = best_estimator_row['alpha'].iloc[0]
    
    # Format all performance metrics of best estimator, these are technically means of the cv results for that estimator
    best_estimator_scores = best_estimator_row.drop(['alpha', 'r2_rank'], axis=1).T.round(decimals=3)
    
    # Using optimal model (best_estimator_) from GridSearch results, run model on test data to compare difference in metrics 
    best_estimator = grid_search.best_estimator_
    test_data_model_results = model_scores_test_data([best_estimator], X_test, y_test)
    
    return_dict = {}
    return_dict['best_estimator'] = best_estimator
    return_dict['best_estimator_alpha'] = best_estimator_alpha
    return_dict['best_estimator_scores'] = best_estimator_scores
    return_dict['test_data_model_results'] = test_data_model_results
    
    return return_dict

# =======================================================================================
# Model Building Variables
# =======================================================================================
# Categorize columns before new features added
num_cols_orig = ['age', 'bmi', 'children']
cat_cols_orig = ['sex', 'smoker', 'region']

# Remove bmi and smoker from numerical and categorical columns lists for custom pipeline 
# preprocessing after new features added (already done)
#num_cols_pipeline = [col for col in num_cols_orig if col!='bmi']
#cat_cols_pipeline = [col for col in cat_cols_orig if col != 'smoker']
num_cols_pipeline = ['age^2', 'children']
cat_cols_pipeline = ['sex', 'region', 'bmi_>=_30', 'smoker*obese']


# =============================
# Separate target from predictors for each dataset (original & new features)
# =============================
# Original data with no feature engineering
y_orig = dataset['charges']
X_orig = dataset.drop(['charges'], axis=1)

# X and y already separated above. X includes 3/4 of the new features, and kept 'bmi_>=_30', but does not include 
# [smoker*bmi] as it needs to be created in the pipeline
y
X

# =============================
# Variables for cross_validate
# =============================
# List of scores for cross_validate() function to return
scoring_list = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 
                'neg_median_absolute_error', 'max_error']

# Abbreviation for output of each score
score_abbvs = ['r2', 'rmse', 'mae', 'mape', 'med_ae', 'me']
score_abbv_dict = dict(zip(scoring_list, score_abbvs))

# Indeces of scores to be multiplied by -1
index_neg = [1, 2, 3, 4, 5]

# ====================================================================================================================
# Linear Regression
# ====================================================================================================================

# Keep track of performance
cv_results_df = pd.DataFrame()
test_results_df = pd.DataFrame()

lr_model_name = 'LR'

# =============================
# Original features
# =============================
# Create Linear Regression model and pipeline
lr_pipeline0 = create_pipeline(lr_model_name, LinearRegression(), num_cols_orig, cat_cols_orig)

# Perform cross-validation and store results in df

cv_results_df['lr_orig'], test_results_df['lr_orig'] = cv_results(lr_pipeline0, X_orig, y_orig, score_abbv_dict, index_neg)


# X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, train_size=0.8, test_size=0.2, random_state=15)
# cv=10
# return_estimator=True
# rr_scores = cross_validate(lr_pipeline0, X_train, y_train, scoring=list(score_abbv_dict.keys()), cv=cv, return_estimator=return_estimator)
# avg_cv_scores_df = average_cv_scores(rr_scores, score_abbv_dict, index_neg)
# cv_test_scores_df = model_scores_test_data(rr_scores['estimator'], X_test, y_test)



# =============================
# New features
# =============================
# Create Linear Regression model and pipeline
lr_pipeline1 = create_pipeline_bmi_smoker(lr_model_name, LinearRegression(), num_cols_pipeline, cat_cols_pipeline)

# Perform cross-validation and store results in df
cv_results_df['lr_new_feat'], test_results_df['lr_new_feat'] = cv_results(lr_pipeline1, X, y, score_abbv_dict, index_neg)

# ====================================================================================================================
# Ridge Regression
# ====================================================================================================================

rr_model_name = 'RR'

# =======================================================================================
# Original features
# =======================================================================================
# ==========================================================
# CV results, no hyperparameter tuning
# ==========================================================

# Create Ridge Regression model and pipeline
rr_pipeline0 = create_pipeline(rr_model_name, Ridge(alpha=1.0, random_state=15), num_cols_orig, cat_cols_orig)

# Perform cross-validation and store results in df
cv_results_df['rr_orig_cv'], test_results_df['rr_orig_cv'] = cv_results(rr_pipeline0, X_orig, y_orig, score_abbv_dict, index_neg)

# ==========================================================
# Hyperparameter tuning
# ==========================================================
# GridSearch parameters
refit = 'r2'
cv = 10
verbose=5
n_jobs=-1

# Create model and pipeline
rid_reg_gs0 = Ridge(alpha=1.0, random_state=15)
rr_pipeline_gs0 = create_pipeline(rr_model_name, rid_reg_gs0, num_cols_orig, cat_cols_orig)

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, train_size=0.8, test_size=0.2, random_state=15)

# Determine hyperparameters to be tuned
rid_reg_gs0.get_params()
rr_pipeline_gs0.get_params()
rr_parameters0 = {rr_model_name + '__alpha': range(0, 10, 1)}

# Perform GridSearch hyperparameter tuning
gs_obj0, gs_results0 = perform_grid_search(rr_pipeline_gs0, X_train, y_train, list(score_abbv_dict.keys()), rr_parameters0, 'orig data')

# Metrics to be extracted from gs_results and their abbreviations
# This is specific to ridge regression gs as the only parameter it includes is alpha
metric_dict =  {'param_RR__alpha':'alpha', 'mean_test_r2':'r2', 'rank_test_r2':'r2_rank',
                'mean_test_neg_root_mean_squared_error':'rmse', 'mean_test_neg_mean_absolute_error':'mae', 
                'mean_test_neg_mean_absolute_percentage_error':'mape',
                'mean_test_neg_median_absolute_error':'med_ae', 'mean_test_max_error':'me'}

# List of metrics that return negative, so will be multiplied by -1
negative_list = ['rmse', 'mae', 'mape', 'med_ae', 'me']

# Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
relevant_gs_results_df0 = gs_relevant_results_to_df(gs_results0, metric_dict, negative_list)

# Visualize change in r2 with each hyperparameter
plt.plot(relevant_gs_results_df0['alpha'], relevant_gs_results_df0['r2'], marker='o', markersize=4)
plt.ylabel('r2')
plt.xlabel('alpha')
plt.title('GS Results-Ridge (orig data)')
plt.grid()
plt.show()
# Looks like I captured the max r2, no need to dive deeper into hyperparameter tuning


# Access metrics of best estimator, this is based on 'refit' parameter which defaults to 'r2'
best_estimator_row0 = relevant_gs_results_df0.loc[relevant_gs_results_df0['r2_rank'] == 1]

# Hyperparameter of best estimator 
best_estimator_alpha0 = best_estimator_row0['alpha'].iloc[0]

# Format all performance metrics of best estimator, these are technically means of the cv results for that estimator
best_estimator_scores0 = best_estimator_row0.drop(['alpha', 'r2_rank'], axis=1).T.round(decimals=3)

# Using optimal model (best_estimator_) from GridSearch results, run model on test data to compare difference in metrics 
best_estimator0 = gs_obj0.best_estimator_
test_data_model_results0 = model_scores_test_data([best_estimator0], X_test, y_test)

# Keep track of results
cv_results_df['rr_orig_gs'] = best_estimator_scores0
test_results_df['rr_orig_gs'] = test_data_model_results0


# =======================================================================================
# New features
# =======================================================================================
# ==========================================================
# CV results, no hyperparameter tuning
# ==========================================================
# Create model and pipeline
rr_pipeline1 = create_pipeline(rr_model_name, Ridge(alpha=1.0, random_state=15), num_cols_pipeline, cat_cols_pipeline)

# Perform cross-validation and store results in df
cv_results_df['rr_new_feat_cv'], test_results_df['rr_new_feat_cv'] = cv_results(rr_pipeline1, X, y, score_abbv_dict, index_neg)

# ==========================================================
# Hyperparameter tuning
# ==========================================================
# GridSearch parameters
refit = 'r2'
cv = 10
verbose=5
n_jobs=-1

# Create model and pipeline
rid_reg_gs1 = Ridge(alpha=1.0, random_state=15)
rr_pipeline_gs1 = create_pipeline(rr_model_name, rid_reg_gs1, num_cols_pipeline, cat_cols_pipeline)

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# Determine hyperparameters to be tuned
rid_reg_gs1.get_params()
rr_pipeline_gs1.get_params()
rr_parameters1 = {rr_model_name + '__alpha': range(0, 10, 1)}

# Perform GridSearch hyperparameter tuning
gs_obj1, gs_results1 = perform_grid_search(rr_pipeline_gs1, X_train, y_train, list(score_abbv_dict.keys()), rr_parameters1, 'new feats')

# Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
relevant_gs_results_df1 = gs_relevant_results_to_df(gs_results1, metric_dict, negative_list)

# Visualize change in r2 with each hyperparameter
plt.plot(relevant_gs_results_df1['alpha'], relevant_gs_results_df1['r2'], marker='o', markersize=4)
plt.ylabel('r2')
plt.xlabel('alpha')
plt.title('GS Results-Ridge (new feats)')
plt.grid()
plt.show()
# Max r2 at alpha=0, decreases from there

# Access metrics of best estimator, this is based on 'refit' parameter which defaults to 'r2'
best_estimator_row1 = relevant_gs_results_df1.loc[relevant_gs_results_df1['r2_rank'] == 1]

# Hyperparameter of best estimator 
best_estimator_alpha1 = best_estimator_row1['alpha'].iloc[0]

# Format all performance metrics of best estimator, these are technically means of the cv results for that estimator
best_estimator_scores1 = best_estimator_row1.drop(['alpha', 'r2_rank'], axis=1).T.round(decimals=3)

# Using optimal model (best_estimator_) from GridSearch results, run model on test data to compare difference in metrics 
best_estimator1 = gs_obj1.best_estimator_
test_data_model_results1 = model_scores_test_data([best_estimator1], X_test, y_test)

# Keep track of results
cv_results_df['rr_new_feat_gs'] = best_estimator_scores1
test_results_df['rr_new_feat_gs'] = test_data_model_results1

# ====================================================================================================================
# Plot everything so far
# ====================================================================================================================
rmse_data = test_results_df.loc['rmse']
plt.plot(rmse_data.index,rmse_data, marker='o', markersize=4)
plt.xlabel('rmse')
plt.xticks(rotation = -25)
plt.ylabel('model')
plt.title('RMSE Summary')
plt.grid()
plt.show()




# ====================================================================================================================
# Lasso Regression
# ====================================================================================================================

lsr_model_name = 'LSR'

# =======================================================================================
# Original features
# =======================================================================================
# ==========================================================
# CV results, no hyperparameter tuning
# ==========================================================

# Create Ridge Regression model and pipeline
lsr_pipeline0 = create_pipeline(lsr_model_name, Lasso(alpha=1.0, random_state=15), num_cols_orig, cat_cols_orig)

# Perform cross-validation and store results in df
cv_results_df['lsr_orig_cv'], test_results_df['lsr_orig_cv'] = cv_results(lsr_pipeline0, X_orig, y_orig, score_abbv_dict, index_neg)

# ==========================================================
# Hyperparameter tuning
# ==========================================================
# GridSearch parameters
refit = 'r2'
cv = 10
verbose=5
n_jobs=-1

# Create model and pipeline
las_reg_gs0 = Lasso(alpha=1.0, random_state=15)
lsr_pipeline_gs0 = create_pipeline(lsr_model_name, las_reg_gs0, num_cols_orig, cat_cols_orig)

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, train_size=0.8, test_size=0.2, random_state=15)

# Determine hyperparameters to be tuned
las_reg_gs0.get_params()
lsr_pipeline_gs0.get_params()
lsr_parameters0 = {lsr_model_name + '__alpha': range(0, 10, 1)}

# Perform GridSearch hyperparameter tuning
lsr_gs_obj0, lsr_gs_results0 = perform_grid_search(lsr_pipeline_gs0, X_train, y_train, list(score_abbv_dict.keys()), lsr_parameters0, 'orig data')

# Metrics to be extracted from gs_results and their abbreviations
# This is specific to lasso regression gs as the only parameter it includes is alpha
metric_dict_lsr =  {'param_LSR__alpha':'alpha', 'mean_test_r2':'r2', 'rank_test_r2':'r2_rank',
                'mean_test_neg_root_mean_squared_error':'rmse', 'mean_test_neg_mean_absolute_error':'mae', 
                'mean_test_neg_mean_absolute_percentage_error':'mape',
                'mean_test_neg_median_absolute_error':'med_ae', 'mean_test_max_error':'me'}

# List of metrics that return negative, so will be multiplied by -1
negative_list = ['rmse', 'mae', 'mape', 'med_ae', 'me']

# Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
lsr_relevant_gs_results_df0 = gs_relevant_results_to_df(lsr_gs_results0, metric_dict_lsr, negative_list)

# Visualize change in r2 with each hyperparameter
plt.plot(lsr_relevant_gs_results_df0['alpha'], lsr_relevant_gs_results_df0['r2'], marker='o', markersize=4)
plt.ylabel('r2')
plt.xlabel('alpha')
plt.title('GS Results-Lasso (orig data)')
plt.grid()
plt.show()
# R2 increasing as alpha increases, doesn't seem to hit its max yet


# New range for hyperparameter alpha
lsr_parameters0 = {lsr_model_name + '__alpha': np.arange(44.0, 44.2, 0.05)}

# Perform GridSearch hyperparameter tuning
lsr_gs_obj0, lsr_gs_results0 = perform_grid_search(lsr_pipeline_gs0, X_train, y_train, list(score_abbv_dict.keys()), lsr_parameters0, 'orig data')

# Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
lsr_relevant_gs_results_df0 = gs_relevant_results_to_df(lsr_gs_results0, metric_dict_lsr, negative_list)

# Visualize change in r2 with each hyperparameter
plt.plot(lsr_relevant_gs_results_df0['alpha'], lsr_relevant_gs_results_df0['r2'], marker='o', markersize=4)
plt.ylabel('r2')
#plt.ylim(0.72726, 0.72728)
plt.xlabel('alpha')
plt.xticks(rotation = -25)
plt.title('GS Results-Lasso (orig data)')
plt.grid()
plt.show()
# Kept changing alph range, found two peaks of r2 around alpha of 50 and 110. The first was a higher r2, so continued to fine-tune
# (Made sure to check way further out to an alpha of 10,000. The r2 just keeps decreasing)
# Alpha of 44.10 gave the best r2 (0.727273)
# Alpha of 36.535 gave the best rmse (6231.735661)

# Access metrics of best estimator, this is based on 'refit' parameter which defaults to 'r2'
lsr_best_estimator_row0 = lsr_relevant_gs_results_df0.loc[lsr_relevant_gs_results_df0['r2_rank'] == 1]

# Hyperparameter of best estimator 
lsr_best_estimator_alpha0 = lsr_best_estimator_row0['alpha'].iloc[0]

# Format all performance metrics of best estimator, these are technically means of the cv results for that estimator
lsr_best_estimator_scores0 = lsr_best_estimator_row0.drop(['alpha', 'r2_rank'], axis=1).T.round(decimals=3)

# Using optimal model (best_estimator_) from GridSearch results, run model on test data to compare difference in metrics 
lsr_best_estimator0 = lsr_gs_obj0.best_estimator_
lsr_test_data_model_results0 = model_scores_test_data([lsr_best_estimator0], X_test, y_test)

# Keep track of results
cv_results_df['lsr_orig_gs'] = lsr_best_estimator_scores0
test_results_df['lsr_orig_gs'] = lsr_test_data_model_results0


# =======================================================================================
# New features
# =======================================================================================
# ==========================================================
# CV results, no hyperparameter tuning
# ==========================================================
# Create model and pipeline
lsr_pipeline1 = create_pipeline(lsr_model_name, Lasso(alpha=1.0, random_state=15), num_cols_pipeline, cat_cols_pipeline)

# Perform cross-validation and store results in df
cv_results_df['lsr_new_feat_cv'], test_results_df['lsr_new_feat_cv'] = cv_results(lsr_pipeline1, X, y, score_abbv_dict, index_neg)

# ==========================================================
# Hyperparameter tuning
# ==========================================================
# GridSearch parameters
refit = 'r2'
cv = 10
verbose=5
n_jobs=-1

# Create model and pipeline
las_reg_gs1 = Lasso(alpha=1.0, random_state=15)
lsr_pipeline_gs1 = create_pipeline(lsr_model_name, las_reg_gs1, num_cols_pipeline, cat_cols_pipeline)

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# Determine hyperparameters to be tuned
las_reg_gs1.get_params()
lsr_pipeline_gs1.get_params()
lsr_parameters1 = {lsr_model_name + '__alpha': np.arange(26, 26.5, 0.05)}

# Perform GridSearch hyperparameter tuning
lsr_gs_obj1, lsr_gs_results1 = perform_grid_search(lsr_pipeline_gs1, X_train, y_train, list(score_abbv_dict.keys()), lsr_parameters1, 'new feats')

# Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
lsr_relevant_gs_results_df1 = gs_relevant_results_to_df(lsr_gs_results1, metric_dict_lsr, negative_list)

# Visualize change in r2 with each hyperparameter
plt.plot(lsr_relevant_gs_results_df1['alpha'], lsr_relevant_gs_results_df1['r2'], marker='o', markersize=4)
plt.ylabel('r2')
plt.xlabel('alpha')
plt.title('GS Results-Lasso (new feats)')
plt.grid()
plt.show()
# Seems that r2 peaks at 7.56 at an alpha of 0 and drops from there (went out to alpha of 10,000 to make sure)

# Access metrics of best estimator, this is based on 'refit' parameter which defaults to 'r2'
lsr_best_estimator_row1 = lsr_relevant_gs_results_df1.loc[lsr_relevant_gs_results_df1['r2_rank'] == 1]

# Hyperparameter of best estimator 
lsr_best_estimator_alpha1 = lsr_best_estimator_row1['alpha'].iloc[0]

# Format all performance metrics of best estimator, these are technically means of the cv results for that estimator
lsr_best_estimator_scores1 = lsr_best_estimator_row1.drop(['alpha', 'r2_rank'], axis=1).T.round(decimals=3)

# Using optimal model (best_estimator_) from GridSearch results, run model on test data to compare difference in metrics 
lsr_best_estimator1 = lsr_gs_obj1.best_estimator_
lsr_test_data_model_results1 = model_scores_test_data([lsr_best_estimator1], X_test, y_test)

# Keep track of results
cv_results_df['lsr_new_feat_gs'] = lsr_best_estimator_scores1
test_results_df['lsr_new_feat_gs'] = lsr_test_data_model_results1

# ====================================================================================================================
# Plot everything so far
# ====================================================================================================================
r2_data = test_results_df.loc['r2']
plt.plot(r2_data.index, r2_data, marker='o', markersize=4)
plt.xlabel('r2')
plt.xticks(rotation = -25)
plt.ylabel('model')
plt.title('r2 Summary')
plt.grid()
plt.show()

# ====================================================================================================================
# ElasticNet?
# ====================================================================================================================


# ====================================================================================================================
# Random Forest
# ====================================================================================================================

