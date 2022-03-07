"""
In cost_modeling_2.py I demonstrated that new features perform better that original data in every regression model, so I won't analyze 
original data in further analysis.
Now I will also only work with scores: R2, MSE, RMSE, MAE
"""

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
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
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
# Visualization helper functions
# ====================================================================================================================



def create_obese_smoker_category(X_df):
    """
    Takes dataframe of X values (X_df), uses them to create Series with categories:
    'obese smokers', 'nonobese smokers', 'obese nonsmokers', 'nonobese nonsmokers'
    Returns a series of said categories which corresponds to 'X_df' parameter
    https://datagy.io/pandas-conditional-column/

    Parameters
    ----------
    X_df : DataFrame
        Contains X values to be used to create categories.

    Returns
    -------
    Series
        Series of said categories (specified above) which corresponds to 'X_df' parameter.

    """
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


class MultiplyTransformer2(BaseEstimator, TransformerMixin):
    """
    Same as MultiplyTransformer() but this one replaces drops 'bmi' column

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
        
        #OH Encoding for 'smoker' (technically don't need to as it only has two possible values, but will keep for future use)
        X_copy.drop(['smoker'], axis=1, inplace=True)
        X_copy['smoker_1'] = pd.DataFrame(self.OH_encoder.transform(X[['smoker']]), index=X.index, columns=['smoker_1'])
        
        # Multiply 'bmi' and 'smoker' columns
        X_copy['bmi*smoker'] = X_copy['bmi'] * X_copy['smoker_1']
        X_copy.drop(['bmi'], axis=1, inplace=True)

        #print('\n>>>>>transform() finished\n')
        return X_copy


# Uses MultiplyTransformer, which takes 'bmi' and 'smoker' and transforms both of them and creates bmi*smoker feature
def create_pipeline_bmi_smoker(model_name, model, num_cols, cat_cols): 
    # Use FeatureUnion preprocessing for 'bmi' and 'smoker' and creates 'bmi*smoker' feature
    union = FeatureUnion([('bmi_smoker', MultiplyTransformer2())])
    
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
# Model Scoring Functions
# ====================================================================================================================
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


def model_scores_test_data(estimators, X_test, y_test, scoring, round=3):
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
    scoring : List
        List of model performance score names, they need to map to the keys of 'score_fnx_dict' below.
    round : Integer, optional
        How many decimals places to round results. The default is 3.

    Returns
    -------
    return_df : DataFrame
        DataFrame containing the model performance scores on the test data.

    """
    
    # Dictionary mapping the name of a score to it's actual sklearn function
    score_fnx_dict = {'r2':r2_score,
                      'mse':mean_squared_error,
                      'rmse':mean_squared_error,
                      'mae':mean_absolute_error,
                      'mape':mean_absolute_percentage_error,
                      'med_ae':median_absolute_error,
                      'me':max_error}
    
    # Using a defaultdict allows you to create a dictionary where the values are lists, which you can 
    # access directly by their keys and append values to. This doesn't work on a normal dictionary.
    test_results_dict = defaultdict(list)
    
    for estimator in estimators:
        y_pred = estimator.predict(X_test)
        
        for score in scoring:
            #'score_fnx_dict[score]' represents the sklearn score function, which always takes (y_test, y_pred) as parameters
            if score=='rmse':
                # There is no 'rmse' function, so you have to take the sqrt of the mse function
                test_results_dict[score].append(np.sqrt(score_fnx_dict[score](y_test, y_pred)))
            else:
                test_results_dict[score].append(score_fnx_dict[score](y_test, y_pred))
        
        # test_results_dict['r2'].append(r2_score(y_test, y_pred))
        # test_results_dict['mse'].append(mean_squared_error(y_test, y_pred))
        # test_results_dict['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        # test_results_dict['mae'].append(mean_absolute_error(y_test, y_pred))
        
        # test_results_dict['mape'].append(mean_absolute_percentage_error(y_test, y_pred))
        # test_results_dict['med_ae'].append(median_absolute_error(y_test, y_pred))
        # test_results_dict['me'].append(max_error(y_test, y_pred))
    
    # Each value in test_results_dict is a list of scores, one for each estimator, this loop averages each score
    #scores = ['r2', 'mse', 'rmse', 'mae']#, 'mape', 'med_ae', 'me']
    test_results_dict2 = defaultdict(list)
    #for score in scores:
    for score in scoring:
        test_results_dict2[score] = np.round(np.mean(test_results_dict[score]), round)
    
    # Convert result dict to df
    return_df = pd.DataFrame.from_dict(dict(test_results_dict2), orient='index')    
    
    return return_df

# Extracts relevant metrics from GridSearch cv_results_ and returns as a dataframe
# metric_dict keys are the names of the metrics in GridSearch cv_results_ and 
# metric_dict values are the updated names of the metrics (usually shorter strings)
def gs_relevant_results_to_df(gs_results, metric_dict, negative_list):
    # Convert GridSearchCV results to a dataframe    
    gs_results_df = pd.DataFrame(gs_results)
    
    # Pulls only the relevant results from gs_results_df
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
    cv_test_scores_df = model_scores_test_data(cv_scores['estimator'], X_test, y_test, scoring=list(score_abbv_dict.values()))
    
    return avg_cv_scores_df, cv_test_scores_df


# Takes parameters and performs hyperparamter tuning using GridSearchCV. 
# Returns best hyperparameter, mean cv results of best estimator, and results when best estimator applied to remaining test data
def perform_grid_search(pipeline, X_train, y_train, scoring_list, param_grid, refit, cv=10, verbose=10, n_jobs=-1):    
    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring_list, 
                               refit=refit, n_jobs=n_jobs, cv=cv, verbose=verbose)
       
    # Hyperparameter tuning using GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Access GridSearch results
    gs_results = grid_search.cv_results_
    
    return grid_search, gs_results

# ====================================================================================================================
# Plot functions
# ====================================================================================================================
# cv_results_df, test_results_df are both DataFrames that contain model performance scores for each model
# This function plots the score according to 'plot_metric'
def plot_regression_model_score_df(cv_results_df, test_results_df, plot_metric, save_img=False, img_filename=None, save_dir=None):
    # Create figure, gridspec, list of axes/subplots mapped to gridspec location
    fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=2, num_cols=1, figsize=(7, 8))

    # Performance on cross-val
    plot_data = cv_results_df.loc[plot_metric]
    ax1 = ax_array_flat[0]
    ax1.plot(plot_data.index, plot_data, marker='o', markersize=4)
    ax1.set_xlabel('Model')
    plt.setp(ax1.get_xticklabels(), rotation=-25, horizontalalignment='left')
    ax1.set_ylabel(plot_metric)
    ax1.set_title('Model Performance During Cross-Validation')
    ax1.grid()

    # Performance on test data
    plot_data = test_results_df.loc[plot_metric]
    ax2 = ax_array_flat[1]
    ax2.plot(plot_data.index, plot_data, marker='o', markersize=4)
    ax2.set_xlabel('Model')
    plt.setp(ax2.get_xticklabels(), rotation=-25, horizontalalignment='left')
    ax2.set_ylabel(plot_metric)
    ax2.set_title('Model Performance on Test Data')
    ax2.grid()

    fig.suptitle('Model Performance', fontsize=22)
    fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
    
    if save_img:
        dh.save_image(img_filename, save_dir)

        
def plot_y_true_vs_pred(y, y_pred, title=None, ax=None, textbox_str=None, outer_text=None, save_img=False, img_filename=None, save_dir=None):   
    largest_num = max(max(y), max(y_pred))
    smallest_num = min(min(y), min(y_pred))
    
    show_plot = False
    if not ax: 
        ax = plt.gca()
        show_plot = True
    
    plot_limits = [smallest_num - (0.02*largest_num), largest_num + (0.02*largest_num)]
    ax.set_xlim(plot_limits)
    ax.set_ylim(plot_limits)
    
    ax.scatter(y, y_pred, s=10, alpha=0.7)    

    ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--', transform=ax.transAxes)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('True Values vs. Predicted Values')
    
    ax.set_ylabel('Predicted Values')
    ax.set_xlabel('True Values')
    
    if textbox_str:
        box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.9}
        ax.text(0.95, 0.95, textbox_str, bbox=box_style, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right') 
    
    if outer_text:
        box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.9}
        ax.text(1.05, 0.99, outer_text, bbox=box_style, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left') 
    
    if save_img:
        dh.save_image(img_filename, save_dir)
        
    if show_plot: plt.show()
    
def plot_y_true_vs_pred_for_cv_and_test_data(estimator, X_train, X_test, y_train, y_test, model_abbrev, title_fontsize=24, 
                                             cv_plot_text=None, test_plot_text=None, fig_title=None, outer_text=None,
                                             ax=None, save_img=False, img_filename=None, save_dir=None):
    
    # Create figure, gridspec, list of axes/subplots mapped to gridspec location
    fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=(9, 4))
    
    # Plot on left is model performance on CV data
    plot_y_true_vs_pred(y_train, estimator.predict(X_train), title='CV Data', ax=ax_array_flat[0], textbox_str=cv_plot_text)
    
    # Plot on right is model performance on Test data
    plot_y_true_vs_pred(y_test, estimator.predict(X_test), title='Test Data', ax=ax_array_flat[1], textbox_str=test_plot_text, outer_text=outer_text)
    
    # Figure title
    if fig_title:
        fig.suptitle(fig_title)
    else:
        fig.suptitle(f'{model_abbrev} Model Performance: CV Data vs. Test Data', fontsize=title_fontsize)
    fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
    
    if save_img:
        dh.save_image(img_filename, save_dir)
    plt.show()



# ====================================================================================================================
# Feature engineering from cost_lin_reg.py
# ====================================================================================================================
# Create formatted columns dictionary in dh module
dh.create_formatted_cols_dict(dataset.columns)
dh.update_formatted_cols('bmi', 'BMI')

# Separate target from predictors
y = dataset['charges']
X = dataset.drop(['charges'], axis=1)

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

# Remove ['bmi_>=_30'] feature. It is correlated to ['smoker*obese'] by definition and 
# it's coefficient in the model was zero by the time ['smoker*obese'] was added in initial statistical analysis
X.drop(['bmi_>=_30'], axis=1, inplace=True)

# I originally wanted to preprocess the data before adding the ['bmi*smoker'] feature. If I added the feature first,
# then tried to scale it, the zeros would be included in the scaling (not sure how to prevent that other than replacing with NaN then back to zero). 
# And the way sklearn cross-validation works is that the preprocessing has to be included in the pipeline, since
# each fold will have a different test/train split. So I created my own custom transformer and pipeline to 
# create the ['bmi*smoker'] feature.


# ====================================================================================================================
# Model Building
# ====================================================================================================================

# =======================================================================================
# Model Building Variables
# =======================================================================================
# Categorize columns before new features added, will not include 'bmi' or 'smoker' as they
# have their own special preprocessing
num_cols = ['age^2', 'children']
cat_cols = ['sex', 'region', 'smoker*obese']


# =============================
# Variables for cross_validate()
# =============================
# List of scores for cross_validate() function to return
scoring_list = ['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']

# Abbreviation for output of each score
score_abbvs = ['r2', 'mse', 'rmse', 'mae']
score_abbv_dict = dict(zip(scoring_list, score_abbvs))

# Indeces of scores to be multiplied by -1
index_neg = [1, 2, 3]

# =============================
# Variables for GridSearchCV()
# =============================

# Metrics to be extracted from gs_results and their abbreviations
# This is specific to ridge regression gs as the only parameter it includes is alpha
gs_metric_dict =  {'mean_test_r2':'r2', 
                   'rank_test_r2':'r2_rank', 
                   'mean_test_neg_mean_squared_error':'mse',
                   'rank_test_neg_mean_squared_error':'mse_rank',
                   'mean_test_neg_root_mean_squared_error':'rmse', 
                   'rank_test_neg_root_mean_squared_error':'rmse_rank',
                   'mean_test_neg_mean_absolute_error':'mae',
                   'rank_test_neg_mean_absolute_error':'mae_rank'}

# List of GridSearchCV() metrics that return negative, so will be multiplied by -1
gs_negative_list = ['mse', 'rmse', 'mae']

# =============================
# Keep track of model performance
# =============================

# DataFrame of model performances scores in CV
cv_results_df = pd.DataFrame()

# DataFrame of model performance scores when applied to test data
test_results_df = pd.DataFrame()


# ====================================================================================================================
# Linear Regression
# ====================================================================================================================

lr_model_name = 'LR'

# Create Linear Regression pipeline
lr_pipeline = create_pipeline_bmi_smoker(lr_model_name, LinearRegression(), num_cols, cat_cols)

# Perform cross-validation and store results in df
cv_results_df['lr_scores'], test_results_df['lr_scores'] = cv_results(lr_pipeline, X, y, score_abbv_dict, index_neg)

# ====================================================================================================================
# Ridge Regression
# ====================================================================================================================

# ==========================================================
# Define variables
# ==========================================================

# Model name to be used in pipeline, which also means it will be used to name pipeline hyperparameters
rr_model_name = 'RR'

# Model hyperparameter names
rr_params = ['alpha']

# Dictionary of ridge regression hyperparameter name (as returned by GridSearchCV()) and an associated
# string name
# (Should be {'param_RR__alpha':'alpha'})
rr_param_dict =  {'param_'+rr_model_name+'__'+param:param for param in rr_params}

# Metrics to be extracted from gs_results and their abbreviations
# Combines gs_metric_dict and rr_param_dict
rr_gs_results_dict = rr_param_dict.copy()
rr_gs_results_dict.update(gs_metric_dict)

# ==========================================================
# Hyperparameter tuning
# ==========================================================
# Create model and pipeline
rr_model = Ridge(random_state=15)
rr_pipeline = create_pipeline_bmi_smoker(rr_model_name, rr_model, num_cols, cat_cols)

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# Determine hyperparameters to be tuned
rr_model.get_params()
rr_pipeline.get_params()
rr_parameters = {rr_model_name + '__alpha': range(0, 10, 1)}

# Perform GridSearch hyperparameter tuning. **** Will tune to minimize MSE as that is most sensitive to outliers **** 
rr_gs_obj, rr_gs_results = perform_grid_search(rr_pipeline, X_train, y_train, list(score_abbv_dict.keys()), rr_parameters, refit='neg_mean_absolute_error')

# Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
rr_gs_results_df = gs_relevant_results_to_df(rr_gs_results, rr_gs_results_dict, gs_negative_list)

# Visualize change in mse with each hyperparameter
plot_metric = 'mae'
plt.plot(rr_gs_results_df['alpha'], rr_gs_results_df[plot_metric], marker='o', markersize=4)
plt.ylabel(plot_metric)
plt.xlabel('alpha')
plt.title('GS Results - Ridge')
plt.grid()
plt.show()
# Min mae as alpha = 0 when optimized to mae and 1 when optmized to mse

# Hyperparameter of best estimator 
rr_best_params = rr_gs_obj.best_params_

# Access metrics of best estimator, this is based on 'refit' parameter which I set to 'mae'
rr_best_estimator_row = rr_gs_results_df.loc[rr_gs_results_df['mse_rank'] == 1]

# Format all performance metrics of best estimator, these are technically means of the cv results for that estimator
rr_best_estimator_scores = rr_best_estimator_row.drop(['alpha', 'r2_rank', 'mse_rank', 'rmse_rank', 'mae_rank'], axis=1).T.round(decimals=3)

# Using optimal model (best_estimator_) from GridSearch results, run model on test data to compare difference in metrics 
rr_model_scores_test_data = model_scores_test_data([rr_gs_obj.best_estimator_], X_test, y_test, scoring=list(score_abbv_dict.values()))

# Keep track of results
cv_results_df['rr_scores'] = rr_best_estimator_scores
test_results_df['rr_scores'] = rr_model_scores_test_data


# ====================================================================================================================
# Lasso Regression
# ====================================================================================================================

# ==========================================================
# Define variables
# ==========================================================

# Model name to be used in pipeline, which also means it will be used to name pipeline hyperparameters
lsr_model_name = 'LSR'

# Model hyperparameter names
lsr_params = ['alpha']

# Dictionary of lasso regression hyperparameter name (as returned by GridSearchCV()) and an associated
# string name
lsr_param_dict =  {'param_'+lsr_model_name+'__'+param:param for param in lsr_params}

# Metrics to be extracted from gs_results and their abbreviations
# Combines gs_metric_dict and rr_param_dict
lsr_gs_results_dict = lsr_param_dict.copy()
lsr_gs_results_dict.update(gs_metric_dict)


# ==========================================================
# Hyperparameter tuning
# ==========================================================

# Create model and pipeline
lsr_model = Lasso(random_state=15)
lsr_pipeline = create_pipeline_bmi_smoker(lsr_model_name, lsr_model, num_cols, cat_cols)

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# Determine hyperparameters to be tuned
lsr_model.get_params()
lsr_pipeline.get_params()
lsr_parameters = {lsr_model_name + '__alpha': np.arange(0, 300, 10)}

# Perform GridSearch hyperparameter tuning. **** Will tune to minimize MSE as that is most sensitive to outliers **** 
lsr_gs_obj, lsr_gs_results = perform_grid_search(lsr_pipeline, X_train, y_train, list(score_abbv_dict.keys()), lsr_parameters, refit='neg_mean_absolute_error')

# Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
lsr_gs_results_df = gs_relevant_results_to_df(lsr_gs_results, lsr_gs_results_dict, gs_negative_list)

# Visualize change in scores with each hyperparameter
plot_metric = 'mae'
plt.plot(lsr_gs_results_df['alpha'], lsr_gs_results_df[plot_metric], marker='o', markersize=4)
plt.ylabel(plot_metric)
plt.xlabel('alpha')
plt.title('GS Results - Lasso')
plt.grid()
plt.show()
# Alpha of 0 yields the best results

# Hyperparameter of best estimator 
lsr_best_params1 = lsr_gs_obj.best_params_

# Access metrics of best estimator, this is based on 'refit' parameter which I set as 'mae'
lsr_best_estimator_row = lsr_gs_results_df.loc[lsr_gs_results_df['mse_rank'] == 1]

# Format all performance metrics of best estimator, these are technically means of the cv results for that estimator
lsr_best_estimator_scores = lsr_best_estimator_row.drop(['alpha', 'r2_rank', 'mse_rank', 'rmse_rank', 'mae_rank'], axis=1).T.round(decimals=3)

# Using optimal model (best_estimator_) from GridSearch results, run model on test data to compare difference in metrics 
lsr_model_scores_test_data = model_scores_test_data([lsr_gs_obj.best_estimator_], X_test, y_test, scoring=list(score_abbv_dict.values()))

# Keep track of results
cv_results_df['lsr_scores'] = lsr_best_estimator_scores
test_results_df['lsr_scores'] = lsr_model_scores_test_data


# ====================================================================================================================
# ElasticNet
# ====================================================================================================================

# ==========================================================
# Define variables
# ==========================================================

# Model name to be used in pipeline, which also means it will be used to name pipeline hyperparameters
en_model_name = 'EN'

# Model hyperparameter names
en_params = ['alpha', 'l1_ratio']

# Dictionary of ElasticNet hyperparameter names (as returned by GridSearchCV()) and an associated
# string name
en_param_dict =  {'param_'+en_model_name+'__'+param:param for param in en_params}

# Metrics to be extracted from gs_results and their abbreviations
# Combines gs_metric_dict and rr_param_dict
en_gs_results_dict = en_param_dict.copy()
en_gs_results_dict.update(gs_metric_dict)

# ==========================================================
# Hyperparameter tuning
# ==========================================================

# Create model and pipeline
en_model = ElasticNet(random_state=15)
en_pipeline = create_pipeline_bmi_smoker(en_model_name, en_model, num_cols, cat_cols)

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# Determine hyperparameters to be tuned
en_model.get_params()
en_pipeline.get_params()
en_parameters = {en_model_name + '__alpha': np.arange(0, 10, 1),
                 en_model_name + '__l1_ratio': np.arange(0, 1.1, 0.1)}

# Perform GridSearch hyperparameter tuning
en_gs_obj, en_gs_results = perform_grid_search(en_pipeline, X_train, y_train, list(score_abbv_dict.keys()), en_parameters, refit='neg_mean_absolute_error')

# Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
en_gs_results_df = gs_relevant_results_to_df(en_gs_results, en_gs_results_dict, gs_negative_list)

# Visualize change in scores with each hyperparameter
unique_l1_ratios = en_gs_results_df['l1_ratio'].unique()

plot_score = 'mae'
for l1_ratio in unique_l1_ratios:
    line_data = en_gs_results_df[en_gs_results_df['l1_ratio']==l1_ratio]
    plt.plot(line_data['alpha'], line_data[plot_score], marker='o', markersize=4, label=np.round(l1_ratio, 2))
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='l1 ratio')
plt.ylabel(plot_score)
plt.xlabel('alpha')
plt.title('GS Results - Elastic Net')
plt.grid()
plt.show()

# Peak R2 (~0.85) at alpha=0. Same for all l1 ratios. But l1 ratio of 0.9 does the best for the rest of the alphas as well
# MAE ~2500 at that alpha=0

# Hyperparameter of best estimator 
en_best_params = en_gs_obj.best_params_
# {'EN__alpha': 0, 'EN__l1_ratio': 0.0}

# Access metrics of best estimator, this is based on 'refit' parameter I set to 'mae'
en_best_estimator_row = en_gs_results_df.loc[(en_gs_results_df['alpha'] == 0) & (en_gs_results_df['l1_ratio'] == 0)]

# Format all performance metrics of best estimator, these are technically means of the cv results for that estimator
en_best_estimator_scores = en_best_estimator_row.drop(['alpha', 'r2_rank', 'mse_rank', 'rmse_rank', 'mae_rank'], axis=1).T.round(decimals=3)

# Using optimal model (best_estimator_) from GridSearch results, run model on test data to compare difference in metrics 
en_model_scores_test_data = model_scores_test_data([en_gs_obj.best_estimator_], X_test, y_test, scoring=list(score_abbv_dict.values()))

# Keep track of results
cv_results_df['en_scores'] = en_best_estimator_scores
test_results_df['en_scores'] = en_model_scores_test_data


# ====================================================================================================================
# Random Forest Regression
# ====================================================================================================================

# ==========================================================
# Define variables
# ==========================================================

# Model name to be used in pipeline, which also means it will be used to name pipeline hyperparameters
rf_model_name = 'RF'

# Model hyperparameter names
rf_params = ['n_estimators', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'bootstrap']

# Dictionary of Random Forest hyperparameter names (as returned by GridSearchCV()) and an associated
# string names
rf_param_dict =  {'param_'+rf_model_name+'__'+param:param for param in rf_params}

# Metrics to be extracted from gs_results and their abbreviations
# Combines gs_metric_dict and rr_param_dict
rf_gs_results_dict = rf_param_dict.copy()
rf_gs_results_dict.update(gs_metric_dict)

# ==========================================================
# Hyperparameter tuning
# ==========================================================

# Create model and pipeline
rf_model = RandomForestRegressor(random_state=15)
rf_pipeline = create_pipeline_bmi_smoker(rf_model_name, rf_model, num_cols, cat_cols)

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# Determine hyperparameters to be tuned
rf_model.get_params()
rf_pipeline.get_params()

# Set hyperparameter ranges for RandomizedSearchCV
n_estimators = np.arange(100, 2000, step=100)
max_features = ['auto', 'sqrt', 'log2']
max_depth = list(np.arange(10, 100, step=10)) + [None]
min_samples_split = np.arange(2, 10, step=2)
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

rf_parameters = {rf_model_name + '__n_estimators': n_estimators,
                 rf_model_name + '__max_features': max_features,
                 rf_model_name + '__max_depth': max_depth,
                 rf_model_name + '__min_samples_split': min_samples_split,
                 rf_model_name + '__min_samples_leaf': min_samples_leaf,
                 rf_model_name + '__bootstrap': bootstrap}

# First use RandomizedSearchCV() to explore a random subset of the hyperparameters, then use the results 
# to narrow the ranges for GridSearchCV()
# https://towardsdatascience.com/automatic-hyperparameter-tuning-with-sklearn-gridsearchcv-and-randomizedsearchcv-e94f53a518ee
rf_random_cv = RandomizedSearchCV(rf_pipeline, rf_parameters, n_iter=100, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=10)
rf_rs_obj = rf_random_cv.fit(X_train, y_train)
rf_rs_best_params = rf_random_cv.best_params_

# NEWEST results (mae optimized)
# {'RF__n_estimators': 1900,
#  'RF__min_samples_split': 2,
#  'RF__min_samples_leaf': 2,
#  'RF__max_features': 'log2',
#  'RF__max_depth': 30,
#  'RF__bootstrap': False}

# NEW results (mse optimized)
# {'RF__n_estimators': 1700,
#  'RF__min_samples_split': 6,
#  'RF__min_samples_leaf': 4,
#  'RF__max_features': 'log2',
#  'RF__max_depth': 10,
#  'RF__bootstrap': True}

# OLD results (r2 optimized)
# {'RF__n_estimators': 200,
#  'RF__min_samples_split': 6,
#  'RF__min_samples_leaf': 4,
#  'RF__max_features': 'log2',
#  'RF__max_depth': 90,
#  'RF__bootstrap': True}
rs_best_score = rf_random_cv.best_score_

# Use RandomizedSearchCV results to narrow ranges of hyperparameters for GridSearchCV()
rf_parameters2 = {rf_model_name + '__n_estimators': [1700, 1800, 1900, 2000, 2100],
                  rf_model_name + '__max_features': ['log2'],
                  rf_model_name + '__max_depth': [None, 20, 30, 40],
                  rf_model_name + '__min_samples_split': [2, 4, 6],
                  rf_model_name + '__min_samples_leaf': [2, 4, 6],
                  rf_model_name + '__bootstrap': [False]}


# Perform GridSearch hyperparameter tuning
rf_gs_obj, rf_gs_results = perform_grid_search(rf_pipeline, X_train, y_train, list(score_abbv_dict.keys()), rf_parameters2, refit='neg_mean_absolute_error')

# Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
rf_gs_results_df = gs_relevant_results_to_df(rf_gs_results, rf_gs_results_dict, gs_negative_list)

# Can't easily visualize this many combinations of hyperparameters

# Hyperparameter of best estimator 
rf_best_params = rf_gs_obj.best_params_

# NEW: MAE optimized
# {'RF__bootstrap': False,
#  'RF__max_depth': 20,
#  'RF__max_features': 'log2',
#  'RF__min_samples_leaf': 2,
#  'RF__min_samples_split': 2,
#  'RF__n_estimators': 1700}

# MSE optimized
# {'RF__bootstrap': True,
#  'RF__max_depth': 10,
#  'RF__max_features': 'log2',
#  'RF__min_samples_leaf': 6,
#  'RF__min_samples_split': 4,
#  'RF__n_estimators': 1900}

# Access metrics of best estimator, this is based on 'refit' parameter which I set to 'mae'
# There are multiple rows with mse rank is 1, so I have to narrow it down
rf_best_estimator_row = rf_gs_results_df.loc[(rf_gs_results_df['mse_rank'] == 1) & (rf_gs_results_df['min_samples_split'] == 2) & (rf_gs_results_df['max_depth'] == 20)]

# Format all performance metrics of best estimator, these are technically means of the cv results for that estimator
rf_best_estimator_scores = rf_best_estimator_row.drop(list(rf_param_dict.values()) + ['r2_rank', 'mse_rank', 'rmse_rank', 'mae_rank'], axis=1).T.round(decimals=3)

# Using optimal model (best_estimator_) from GridSearch results, run model on test data to compare difference in metrics 
rf_test_data_model_results = model_scores_test_data([rf_gs_obj.best_estimator_], X_test, y_test, scoring=list(score_abbv_dict.values()))

# Keep track of results
cv_results_df['rf_scores'] = rf_best_estimator_scores
test_results_df['rf_scores'] = rf_test_data_model_results


# ====================================================================================================================
# Huber Regression
# ====================================================================================================================

# ==========================================================
# Define variables
# ==========================================================

# Model name to be used in pipeline, which also means it will be used to name pipeline hyperparameters
hr_model_name = 'HR'

# Model hyperparameter names
hr_params = ['alpha', 'epsilon']

# Dictionary of Random Forest hyperparameter names (as returned by GridSearchCV()) and an associated
# string names
hr_param_dict =  {'param_'+hr_model_name+'__'+param:param for param in hr_params}

# Metrics to be extracted from gs_results and their abbreviations
# Combines gs_metric_dict and rr_param_dict
hr_gs_results_dict = hr_param_dict.copy()
hr_gs_results_dict.update(gs_metric_dict)

# ==========================================================
# Hyperparameter tuning
# ==========================================================

# Create model and pipeline
hr_model = HuberRegressor()
hr_pipeline = create_pipeline_bmi_smoker(hr_model_name, hr_model, num_cols, cat_cols)

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=15)

# Determine hyperparameters to be tuned
hr_model.get_params()
hr_pipeline.get_params()

# Set hyperparameter ranges for RandomizedSearchCV
hr_parameters = {hr_model_name + '__alpha': [0.0001],
                 hr_model_name + '__epsilon': np.arange(2, 5, step=0.1)}

# Perform GridSearch hyperparameter tuning
hr_gs_obj, hr_gs_results = perform_grid_search(hr_pipeline, X_train, y_train, list(score_abbv_dict.keys()), hr_parameters, refit='neg_mean_absolute_error')

# Extract revelant performance metrics from gs_results (based on metric_dict) and convert to df
hr_gs_results_df = gs_relevant_results_to_df(hr_gs_results, hr_gs_results_dict, gs_negative_list)

# Visualize change in scores with each hyperparameter
unique_alphas = hr_gs_results_df['alpha'].unique()

plot_score = 'mae'
for alpha in unique_alphas:
    line_data = hr_gs_results_df[hr_gs_results_df['alpha']==alpha]
    plt.plot(line_data['epsilon'], line_data[plot_score], marker='o', markersize=4, label=np.round(alpha, 2))
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='alpha')
plt.ylabel(plot_score)
plt.xlabel('epsilon')
plt.title('GS Results - Huber Regression')
plt.grid()
plt.show()

# Alpha of 0 always best. Best MAE at epsilon of 2. Best MSE at epsilon of 4.2

# Hyperparameter of best estimator 
hr_best_params = hr_gs_obj.best_params_
#{'HR__alpha': 0.0001, 'HR__epsilon': 2}

# Access metrics of best estimator, this is based on 'refit' parameter which I set to 'mae'
# There are multiple rows with mse rank is 1, so I have to narrow it down
hr_best_estimator_row = hr_gs_results_df.loc[hr_gs_results_df['mse_rank'] == 1]

# Format all performance metrics of best estimator, these are technically means of the cv results for that estimator
hr_best_estimator_scores = hr_best_estimator_row.drop(list(hr_param_dict.values()) + ['r2_rank', 'mse_rank', 'rmse_rank', 'mae_rank'], axis=1).T.round(decimals=3)

# Using optimal model (best_estimator_) from GridSearch results, run model on test data to compare difference in metrics 
hr_test_data_model_results = model_scores_test_data([hr_gs_obj.best_estimator_], X_test, y_test, scoring=list(score_abbv_dict.values()))

# Keep track of results
cv_results_df['hr_scores'] = hr_best_estimator_scores
test_results_df['hr_scores'] = hr_test_data_model_results


# ====================================================================================================================
# Plot everything so far
# ====================================================================================================================

# =======================================================================================
# Plot y vs. y_pred
# =======================================================================================

plot_metric = 'mae'
plot_regression_model_score_df(cv_results_df, test_results_df, plot_metric, save_img=False, img_filename='model_performance_mae', save_dir=ml_models_output_dir)

# LR, RR, LSR, and EN all perform exactly the same because the regularization model parameters ended up being tuned 
# to the point where the model functions like linear regression. Random Forest didn't perform as well, MAE was increased by ~$180
# For reference, MAE was ~4000 before feature engineering, now it's ~2100 on the test data
# Huber Regression was the first model to achieve an MAE below 2100 (2088), next lowest was Lasso with 2110

# =======================================================================================
# Model Fit to during CV and when applied to Test Data
# =======================================================================================

digits=2

model_abbrev = 'LR'
lr_pipeline.fit(X_train, y_train)
cv_mae_str = 'MAE: ' + str(round(cv_results_df['lr_scores'].loc['mae'], digits))
test_mae_str = 'MAE: ' + str(round(test_results_df['lr_scores'].loc['mae'], digits))
filename = 'performance_' + model_abbrev
outer_text = 'Hyperparameters:\n\nNone'
plot_y_true_vs_pred_for_cv_and_test_data(lr_pipeline, X_train, X_test, y_train, y_test, 
                                         model_abbrev, cv_plot_text=cv_mae_str, test_plot_text=test_mae_str, outer_text=outer_text,
                                         save_img=False, img_filename=filename, save_dir=ml_models_output_dir)

model_abbrev = 'RR'
gs_obj = rr_gs_obj
cv_mae_str = 'MAE: ' + str(round(cv_results_df['rr_scores'].loc['mae'], digits))
test_mae_str = 'MAE: ' + str(round(test_results_df['rr_scores'].loc['mae'], digits))
filename = 'performance_' + model_abbrev
outer_text = 'Hyperparameters:\n\nAlpha: 0.0'
plot_y_true_vs_pred_for_cv_and_test_data(gs_obj.best_estimator_, X_train, X_test, y_train, y_test, 
                                         model_abbrev, cv_plot_text=cv_mae_str, test_plot_text=test_mae_str, outer_text=outer_text,
                                         save_img=False, img_filename=filename, save_dir=ml_models_output_dir)

model_abbrev = 'LSR'
gs_obj = lsr_gs_obj
cv_mae_str = 'MAE: ' + str(round(cv_results_df['lsr_scores'].loc['mae'], digits))
test_mae_str = 'MAE: ' + str(round(test_results_df['lsr_scores'].loc['mae'], digits))
filename = 'performance_' + model_abbrev
outer_text = 'Hyperparameters:\n\nAlpha: 0.0'
plot_y_true_vs_pred_for_cv_and_test_data(gs_obj.best_estimator_, X_train, X_test, y_train, y_test, 
                                         model_abbrev, cv_plot_text=cv_mae_str, test_plot_text=test_mae_str, outer_text=outer_text,
                                         save_img=False, img_filename=filename, save_dir=ml_models_output_dir)

model_abbrev = 'EN'
gs_obj = en_gs_obj
cv_mae_str = 'MAE: ' + str(round(cv_results_df['en_scores'].loc['mae'], digits))
test_mae_str = 'MAE: ' + str(round(test_results_df['en_scores'].loc['mae'], digits))
filename = 'performance_' + model_abbrev
outer_text = 'Hyperparameters:\n\nAlpha: 0.0\nL1 Ratio: 0.0'
plot_y_true_vs_pred_for_cv_and_test_data(gs_obj.best_estimator_, X_train, X_test, y_train, y_test, 
                                         model_abbrev, cv_plot_text=cv_mae_str, test_plot_text=test_mae_str, outer_text=outer_text,
                                         save_img=False, img_filename=filename, save_dir=ml_models_output_dir)

model_abbrev = 'RF'
gs_obj = rf_gs_obj
cv_mae_str = 'MAE: ' + str(round(cv_results_df['rf_scores'].loc['mae'], digits))
test_mae_str = 'MAE: ' + str(round(test_results_df['rf_scores'].loc['mae'], digits))
filename = 'performance_' + model_abbrev
outer_text = 'Hyperparameters:\n\nBootstrap: False\nMax Depth: 20\nMax Features: log2\nMin Samples Leaf: 2\nMin Samples Split: 2\nN Estimators: 1700'
plot_y_true_vs_pred_for_cv_and_test_data(gs_obj.best_estimator_, X_train, X_test, y_train, y_test, 
                                         model_abbrev, cv_plot_text=cv_mae_str, test_plot_text=test_mae_str, outer_text=outer_text, 
                                         save_img=False, img_filename=filename, save_dir=ml_models_output_dir)

model_abbrev = 'HR'
gs_obj = hr_gs_obj
cv_mae_str = 'MAE: ' + str(round(cv_results_df['hr_scores'].loc['mae'], digits))
test_mae_str = 'MAE: ' + str(round(test_results_df['hr_scores'].loc['mae'], digits))
filename = 'performance_' + model_abbrev
outer_text = 'Hyperparameters:\n\nAlpha: 0.0001\nEpsilon: 2.0'
plot_y_true_vs_pred_for_cv_and_test_data(gs_obj.best_estimator_, X_train, X_test, y_train, y_test, 
                                         model_abbrev, cv_plot_text=cv_mae_str, test_plot_text=test_mae_str, outer_text=outer_text, 
                                         save_img=False, img_filename=filename, save_dir=ml_models_output_dir)


# # Optimized to MAE
# model_abbrev = 'HR2'
# hr_model2 = HuberRegressor(epsilon=3)
# hr_pipeline2 = create_pipeline_bmi_smoker(hr_model_name, hr_model2, num_cols, cat_cols)
# hr_pipeline2.fit(X_train, y_train)

# y_pred_cv = hr_pipeline2.predict(X_train)
# y_pred_test = hr_pipeline2.predict(X_test)

# mae_cv = mean_absolute_error(y_train, y_pred_cv)
# mae_test = mean_absolute_error(y_test, y_pred_test)
# cv_mae_str = 'MAE: ' + str(round(mae_cv, digits)) + '\nR2: ' + str(round(r2_score(y_train, y_pred_cv), digits))
# test_mae_str = 'MAE: ' + str(round(mae_test, digits)) + '\nR2: ' + str(round(r2_score(y_test, y_pred_test), digits))
# filename = 'performance_' + model_abbrev
# outer_text = 'Hyperparameters:\n\nAlpha: 0.0001\nEpsilon: 3.0'
# plot_y_true_vs_pred_for_cv_and_test_data(hr_pipeline2, X_train, X_test, y_train, y_test, 
#                                          model_abbrev, cv_plot_text=cv_mae_str, test_plot_text=test_mae_str, outer_text=outer_text, 
#                                          save_img=False, img_filename=filename, save_dir=ml_models_output_dir)





