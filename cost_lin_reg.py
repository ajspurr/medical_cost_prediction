import sys
import numpy as np
import pandas as pd
from os import chdir
import seaborn as sns 
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import PureWindowsPath, Path

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence as influence

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

# Create list of categorical variables with target and one without target
num_cols_w_target = numerical_cols.copy()
numerical_cols.remove('charges')

# Create list of categorical + ordinal variables for certain tasks
cat_ord_cols = categorical_cols.copy()
cat_ord_cols.append('children')

# Create formatted columns dictionary in dh module
dh.create_formatted_cols_dict(dataset.columns)
dh.add_edit_formatted_col('bmi', 'BMI')

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
  
# ====================================================================================================================
# Data preprocessing function without using pipeline
# ====================================================================================================================

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

# =======================================================================================
# Statsmodels functions
# =======================================================================================

def sm_lr_model_results_subgrouped(lr_model, X_data, y, y_pred, plot_title, combine_plots=True, 
                                   grouping=None, cmap=my_cmap, save_img=False, filename_unique=None):      
    """
    Create Scale-Location Plot (studentized residuals vs. predicted values) and true values vs. predicted values plot
    
    Credits:
    https://datavizpyr.com/add-legend-to-scatterplot-colored-by-a-variable-with-matplotlib-in-python/
    https://www.statology.org/matplotlib-scatterplot-legend/

    Parameters
    ----------
    lr_model : statsmodels.regression.linear_model.OLS
        statsmodels OLS model used.
    X_data : Pandas DataFrame
        Features and their values.
    y : array_like (1-D)
        Series of y-values (target) from original dataset.
    y_pred : array_like (1-D)
        Series of predicted y-values.
    plot_title : string
        String to be added to figure title (if combining plots).
    combine_plots : boolean, optional
        Whether or not to combine the two plots created. The default is True.
    grouping : array_like (1-D), optional
        If scatterplots are to be colored by a certain subcategory, this is the series with labels for each sample. The default is None.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to be used if grouping plots. The default is my_cmap.
    save_img : boolean, optional
        Whether or not to save the image (can only save if combining). The default is False.
    filename_unique : string, optional
        Unique string to be added to 'sm_lr_results_' as the saved image file name. The default is None.

    Returns
    -------
    het_metrics : Dictionary of dictionaries
        Dictionary of dictionaries containing the heteroscedasticity metrics for both Breusch-Pagan test and White test.

    """
    
    # Organize relevant data
    standardized_residuals = pd.DataFrame(lr_model.get_influence().resid_studentized_internal, columns=['stand_resid'])
    y_pred_series = pd.Series(y_pred, name='y_pred')
    y_series = pd.Series(y, name='y')
    
    # Changed join to 'inner' as the 'grouping' parameter may be larger than other variables like 'y_series'. 'grouping'
    # may have been created before removing certain samples, like outliers. Join='inner' maps the grouping index to the 
    # y_series index
    relevant_data = pd.concat([y_series, y_pred_series, standardized_residuals, grouping], axis=1, join='inner')
        
    # Quantify Heteroscedasticity using Breusch-Pagan test and White test
    bp_test = het_breuschpagan(lr_model.resid, lr_model.model.exog)
    white_test = het_white(lr_model.resid, lr_model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    bp_test_results = dict(zip(labels, bp_test))
    white_test_results = dict(zip(labels, white_test))
    bp_lm_p_value = '{:0.2e}'.format(bp_test_results['LM-Test p-value'])
    white_lm_p_value = '{:0.2e}'.format(white_test_results['LM-Test p-value'])
    
    # Convert the grouping variable 'grouping' to a pandas.Categorical object so I can encode each 
    # category to a number (grouping_as_cat.codes) and save the associated category names (grouping_as_cat.categories.tolist())
    if grouping is not None:
        grouping = relevant_data['grouping']
        grouping_as_cat = grouping.astype('category').cat
        grouping_as_codes = grouping_as_cat.codes
        grouping_categories = grouping_as_cat.categories.tolist()
    else:
        grouping_as_codes = None
        grouping_categories = None
        
    # =============================
    # Initialize plot formatting variables
    # =============================
    if combine_plots:
        # Create figure, gridspec, list of axes/subplots mapped to gridspec location
        fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=(12, 5))
    
    # Format text box with relevant metric of each plot
    box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.8}


    # =============================
    # Plot studentized residuals vs. predicted values
    # =============================
    if not combine_plots:
        scatter1 = plt.scatter(relevant_data['y_pred'], relevant_data['stand_resid'], c=grouping_as_codes, cmap=cmap, alpha=0.5)
        ax1 = plt.gca()
    else:
        ax1 = ax_array_flat[0]
        ax1.scatter(relevant_data['y_pred'], relevant_data['stand_resid'], c=grouping_as_codes, cmap=cmap, alpha=0.5)
    
    ax1.axhline(y=0, color='darkblue', linestyle='--')
    ax1.set_ylabel('Studentized Residuals')
    ax1.set_xlabel('Predicted Values')
    ax1.set_title('Scale-Location')
    textbox_text = f'BP: {bp_lm_p_value} \n White: {white_lm_p_value}' 
    ax1.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right')  
    if not combine_plots: 
        if grouping is not None:
            ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Subgroup', 
                       handles=scatter1.legend_elements()[0], labels=grouping_categories)  
        plt.show()
    
    # =============================
    # True Values vs. Predicted Values 
    # =============================
    if not combine_plots:
        scatter2 = plt.scatter(relevant_data['y'], relevant_data['y_pred'], c=grouping_as_codes, cmap=cmap, alpha=0.5)
        ax2 = plt.gca()
    else:
        ax2 = ax_array_flat[1]
        scatter2 = ax2.scatter(relevant_data['y'], relevant_data['y_pred'], c=grouping_as_codes, cmap=cmap, alpha=0.5)
      
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
    if grouping is not None:
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Subgroup', 
                   handles=scatter2.legend_elements()[0], labels=grouping_categories)  
    r2_str = r'$R^2$: %0.3f' %lr_model.rsquared
    r2_adj_str = r'Adj $R^2$: %0.3f' %lr_model.rsquared_adj
    textbox_text = f"{r2_str}\n{r2_adj_str}"
    ax2.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right') 
    if not combine_plots: plt.show()
    
    # =============================
    # Format and save figure
    # =============================
    if combine_plots:
        fig.suptitle('LR Model Performance (' + plot_title + ')', fontsize=24)
        fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
        if save_img:
            save_filename = 'sm_lr_results_' + filename_unique
            dh.save_image(save_filename, models_output_dir)
        plt.show()

    het_metrics = dict(zip(['BP', 'White'], [bp_test_results, white_test_results]))
    
    return het_metrics

# Combine statsmodels linear regression model creation, fitting, and returning results    
def fit_lr_model_results(fxn_X, fxn_y, plot_title, combine_plots=True, subgroup=False, ob_smoke_series=None, cmap=my_cmap, save_img=False, filename_unique=None):
    # Fit model
    fxn_lin_reg = sm.OLS(fxn_y, fxn_X).fit()
    
    # Predict target
    fxn_y_pred = fxn_lin_reg.predict(fxn_X) 
    
    if subgroup:
        if ob_smoke_series is None:
            # Create new category that combines both smoking and obesity (obese smoker, obese nonsmoker, etc.)
            ob_smoke_series = create_obese_smoker_category(fxn_X)
           
        # Plot results subgrouped, get heteroscedasticity metrics
        het_results = sm_lr_model_results_subgrouped(fxn_lin_reg, fxn_X, fxn_y, fxn_y_pred, plot_title, combine_plots=combine_plots,
                                                 grouping=ob_smoke_series, cmap=cmap, save_img=save_img, filename_unique=filename_unique)
    
    else:
        het_results = sm_lr_model_results_subgrouped(fxn_lin_reg, fxn_X, fxn_y, fxn_y_pred, plot_title, combine_plots=combine_plots,
                                                 cmap=cmap, save_img=save_img, filename_unique=filename_unique)
        
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
vif = dh.calulate_vif(dataset, numerical_cols).to_frame()

# All very close to 1, no multicollinearity. (Greater than 5-10 indicates multicollinearity)

# Rename VIF columns
vif.rename(columns={0:'VIF'}, inplace=True)

# Round to 2 decimal places
vif = np.round(vif, decimals=2)

# Convert VIF values to string to avoid render_mpl_table() removing trailing zeroes
vif['VIF'] = vif['VIF'].map('{:,.2f}'.format)

# Create table image
dh.render_mpl_table(vif, index_col_name='Feature')
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

# Fit and plot model
title_0 = 'Original'
model_name_0 = 'original'
file_name_0 = '0_' + model_name_0
sm_lin_reg_0, sm_y_pred_0, het_results_0 = fit_lr_model_results(sm_processed_X, y,  plot_title=title_0, subgroup=False, save_img=False, filename_unique=file_name_0)

# Organize model performance metrics
summary_df_0 = sm_results_to_df(sm_lin_reg_0.summary())
coeff_0 = pd.Series(summary_df_0['coef'], name=model_name_0)
sm_lr_results_0 = pd.Series(dh.evaluate_model_sm(y, sm_y_pred_0, sm_lin_reg_0, het_results_0), name=model_name_0)

# ==========================================================
# Based on EDA, created dichotomous feature 'bmi_>=_30'
# ==========================================================
# Create new feature
new_X_1 = X.copy()
new_X_1['bmi_>=_30'] = new_X_1['bmi'] >= 30
bmi_dict = {False:'no', True:'yes'}
new_X_1['bmi_>=_30'] = new_X_1['bmi_>=_30'].map(bmi_dict)

# Add the new feature to the columns lists (necessary for preprocessing)
categorical_cols.append('bmi_>=_30')
cat_ord_cols.append('bmi_>=_30')
dh.add_edit_formatted_col('bmi_>=_30', 'BMI >= 30')

# Preprocess with new feature
new_X_1 = manual_preprocess_sm(new_X_1)

# Plot model without subgrouping
title_1 = 'w [bmi>=30] feature'
model_name_1 = '[bmi_>=_30]'
file_name_1_0 = '1_bmi_30_feature'
sm_lin_reg_1_0, sm_y_pred_1_0, het_results_1_0 = fit_lr_model_results(new_X_1, y, title_1, save_img=False, filename_unique=file_name_1_0)

# Plot model with subgrouping
file_name_1 = '1_bmi_30_feature_grouped'
sm_lin_reg_1, sm_y_pred_1, het_results_1 = fit_lr_model_results(new_X_1, y, title_1, subgroup=True, save_img=False, filename_unique=file_name_1)

# Organize model performance metrics
summary_df_1 = sm_results_to_df(sm_lin_reg_1.summary())
coeff_1 = pd.Series(summary_df_1['coef'], name=model_name_1)
sm_lr_results_1 = pd.Series(dh.evaluate_model_sm(y, sm_y_pred_1, sm_lin_reg_1, het_results_1), name=model_name_1)

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
sm_lin_reg_2, sm_y_pred_2, het_results_2 = fit_lr_model_results(new_X_2, y, title_2, subgroup=True, save_img=False, filename_unique=file_name_2)

# Organize model performance metrics
summary_df_2 = sm_results_to_df(sm_lin_reg_2.summary())
coeff_2 = pd.Series(summary_df_2['coef'], name=model_name_2)
sm_lr_results_2 = pd.Series(dh.evaluate_model_sm(y, sm_y_pred_2, sm_lin_reg_2, het_results_2), name=model_name_2)

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
sm_lin_reg_3, sm_y_pred_3, het_results_3 = fit_lr_model_results(new_X_3, y, title_3, subgroup=True, save_img=False, filename_unique=file_name_3)

# Organize model performance metrics
summary_df_3 = sm_results_to_df(sm_lin_reg_3.summary())
coeff_3 = pd.Series(summary_df_3['coef'], name=model_name_3)
sm_lr_results_3 = pd.Series(dh.evaluate_model_sm(y, sm_y_pred_3, sm_lin_reg_3, het_results_3), name=model_name_3)

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
sm_lin_reg_4, sm_y_pred_4, het_results_4 = fit_lr_model_results(new_X_4, y, title_4, subgroup=True, save_img=False, filename_unique=file_name_4)

# Organize model performance metrics
summary_df_4 = sm_results_to_df(sm_lin_reg_4.summary())
coeff_4 = pd.Series(summary_df_4['coef'], name=model_name_4)
sm_lr_results_4 = pd.Series(dh.evaluate_model_sm(y, sm_y_pred_4, sm_lin_reg_4, het_results_4), name=model_name_4)

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

# All features
coeff_df_new = coeff_df.apply(pd.to_numeric)

# Replace NaN with 0
coeff_df_new = coeff_df_new.replace(np.nan, 0)

# Separate new and old features
num_orig_features = 9
orig_features_df = coeff_df_new.iloc[0:num_orig_features]
new_features_df = coeff_df_new.iloc[num_orig_features:len(coeff_df_new.index)]

# Separate smoker variable out from orig_features_df as its scale is much larger
smoker_df = orig_features_df.loc['smoker_yes'].to_frame().T
orig_features_no_smoker = orig_features_df.drop(['smoker_yes'], axis=0)

# Plot coefficients
dh.plot_coefficient_df(smoker_df, orig_features_no_smoker, new_features_df)


# =============================
# Drop a few of the mostly-static coefficients
# =============================
# Drop variables that don't change much: children, sex_male, all regions, const.
drop_var = ['region_northwest', 'region_southeast', 'region_southwest', 'const']
coeff_df_new = coeff_df_new.drop(drop_var, axis=0)

# Separate new and old features
num_orig_feat_left = 5
orig_features_df = coeff_df_new.iloc[0:num_orig_feat_left]
new_features_df = coeff_df_new.iloc[num_orig_feat_left:len(coeff_df_new.index)]

# Separate smoker variable out from orig_features_df as its scale is much larger
smoker_df = orig_features_df.loc['smoker_yes'].to_frame().T
orig_features_no_smoker = orig_features_df.drop(['smoker_yes'], axis=0)

# Plot coefficients
dh.plot_coefficient_df(smoker_df, orig_features_no_smoker, new_features_df)

# =======================================================================================
# Compare performance before and after new features
# =======================================================================================

# Convert values to numeric
sm_results_df = sm_results_df.apply(pd.to_numeric)

# Separate out metrics by scale of their values
all__error_mets = sm_results_df.loc[['max_e', 'rmse', 'mae', 'med_abs_e']]
max_e_df = sm_results_df.loc['max_e'].to_frame().T
error_metrics = sm_results_df.loc[['rmse', 'mae', 'med_abs_e']]
r_metrics = sm_results_df.loc[['r2', 'r2_adj']]
het_stats = sm_results_df.loc[['bp_lm_p', 'white_lm_p']]

# Plot model performance metrics
dh.plot_model_metrics_df(max_e_df, error_metrics, r_metrics, het_stats)



# =======================================================================================
# Remove old features
# =======================================================================================
# Remove variables which have standardized coefficients close to zero AND were used to create one of the 
# new features I created: bmi, age, bmi>=30. By definition, they are correlated to the new features 
# [bmi*smoker], [smoker*obese], and [age^2] and should be removed for that reason anyway.

# Create new category that combines both smoking and obesity (obese smoker, obese nonsmoker, etc.)
# I do it before calling fit_lr_model_results() as I'm about to remove 'bmi_>=_30_yes' feature
ob_smoke_series = create_obese_smoker_category(new_X_4)

# Remove old variables
remove_var = ['age', 'bmi', 'bmi_>=_30_yes']
new_X_4_2 = new_X_4.drop(remove_var, axis=1)

# Make sure  model still works
title_4_2 = 'w/o old features'
model_name_4_2 = 'rem_old_var'
file_name_4_2 = '4_2_rem_old_features'
sm_lin_reg_4_2, sm_y_pred_4_2, het_results_4_2 = fit_lr_model_results(new_X_4_2, y, title_4_2, subgroup=True, 
                                                                      ob_smoke_series=ob_smoke_series, save_img=False, 
                                                                      filename_unique=file_name_4_2)

# Organize model performance metrics
summary_df_4_2 = sm_results_to_df(sm_lin_reg_4_2.summary())
coeff_4_2 = pd.Series(summary_df_4_2['coef'], name=model_name_4_2)
sm_lr_results_4_2 = pd.Series(dh.evaluate_model_sm(y, sm_y_pred_4_2, sm_lin_reg_4_2, het_results_4_2), name=model_name_4_2)

# Keep track of model performance for comparison later
coeff_df = pd.concat([coeff_df, coeff_4_2], axis=1)
sm_results_df = pd.concat([sm_results_df, sm_lr_results_4_2], axis=1)

# ==========================================================
# Compare coefficients before and after removing old features
# ==========================================================

# All features
coeff_df_new = coeff_df.apply(pd.to_numeric)

# Replace NaN with 0
coeff_df_new = coeff_df_new.replace(np.nan, 0)

# Separate new and old features
num_orig_features = 9
orig_features_df = coeff_df_new.iloc[0:num_orig_features]
new_features_df = coeff_df_new.iloc[num_orig_features:len(coeff_df_new.index)]

# Separate smoker variable out from orig_features_df as its scale is much larger
smoker_df = orig_features_df.loc['smoker_yes'].to_frame().T
orig_features_no_smoker = orig_features_df.drop(['smoker_yes'], axis=0)

# Plot coefficients
dh.plot_coefficient_df(smoker_df, orig_features_no_smoker, new_features_df)

# =============================
# Drop coefficients of variables I just removed as they will all be zero anyway
# =============================
# Drop coefficients of variables I just removed as they will all be zero anyway
drop_var = ['age', 'bmi', 'bmi_>=_30_yes']
coeff_df_new = coeff_df_new.drop(drop_var, axis=0)

# Separate new and old features
num_orig_feat_left = 7
orig_features_df = coeff_df_new.iloc[0:num_orig_feat_left]
new_features_df = coeff_df_new.iloc[num_orig_feat_left:len(coeff_df_new.index)]

# Separate smoker variable out from orig_features_df as its scale is much larger
smoker_df = orig_features_df.loc['smoker_yes'].to_frame().T
orig_features_no_smoker = orig_features_df.drop(['smoker_yes'], axis=0)

# Plot coefficients
dh.plot_coefficient_df(smoker_df, orig_features_no_smoker, new_features_df)

# =============================
# Now that I dropped old features, I can just remove the constant, and the scales will work out well
# =============================
# Drop variables that don't change much: children, sex_male, all regions, const.
drop_var = ['const']
coeff_df_new = coeff_df_new.drop(drop_var, axis=0)

# Separate new and old features
num_orig_feat_left = 6
orig_features_df = coeff_df_new.iloc[0:num_orig_feat_left]
new_features_df = coeff_df_new.iloc[num_orig_feat_left:len(coeff_df_new.index)]

# Separate smoker variable out from orig_features_df as its scale is much larger
smoker_df = orig_features_df.loc['smoker_yes'].to_frame().T
orig_features_no_smoker = orig_features_df.drop(['smoker_yes'], axis=0)

# Plot coefficients
dh.plot_coefficient_df(smoker_df, orig_features_no_smoker, new_features_df, save_img=False, filename='coeff_vert_6_rem_old_var', save_dir=models_output_dir)

# =======================================================================================
# Compare performance before and after removing old features
# =======================================================================================

# Convert values to numeric
sm_results_df = sm_results_df.apply(pd.to_numeric)

# Separate out metrics by scale of their values
all__error_mets = sm_results_df.loc[['max_e', 'rmse', 'mae', 'med_abs_e']]
max_e_df = sm_results_df.loc['max_e'].to_frame().T
error_metrics = sm_results_df.loc[['rmse', 'mae', 'med_abs_e']]
r_metrics = sm_results_df.loc[['r2', 'r2_adj']]
het_stats = sm_results_df.loc[['bp_lm_p', 'white_lm_p']]

# Plot model performance metrics
dh.plot_model_metrics_df(max_e_df, error_metrics, r_metrics, het_stats, save_img=False, filename='performance_no_outliers', save_dir=models_output_dir)

# =======================================================================================
# Outlier Analysis
# =======================================================================================
# https://towardsdatascience.com/linear-regression-model-with-python-481c89f0f05b
# Observation has a high influence if the Cook's distance is greater than 4/(N-k-1)
# N = number of observations, k = number of predictors, yellow horizontal line in the plot

# ==========================================================
# Identify Outliers
# ==========================================================
#inf = influence(sm_lin_reg_4)
inf = influence(sm_lin_reg_4_2)
(cooks, d) = inf.cooks_distance
cooks_cutoff = 4 / (len(cooks) - (new_X_4_2.shape[1] - 1) - 1) # 0.00301

outlier_df = new_X_4_2.copy()
outlier_df['cooks'] = cooks
outlier_df['outlier'] = outlier_df['cooks'] > cooks_cutoff
outlier_dict = {False:'no', True:'yes'}
outlier_df['outlier'] = outlier_df['outlier'].map(outlier_dict)

num_outliers = outlier_df[outlier_df['outlier'] == 'yes'].shape[0] # 90 -> 85 after removing old variables
perc_outliers = num_outliers / outlier_df.shape[0] # 0.0672 -> 0.0635 after removing old variables
outlier_df['true_values'] = y
outlier_df['y_pred'] = sm_y_pred_4_2
outlier_df['stud_resid'] = sm_lin_reg_4_2.get_influence().resid_studentized_internal

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
sns.lmplot(x='age', y='charges', hue="outlier", data=orig_data_w_outlier, ci=None, fit_reg=False, legend=False) # LM plot just makes it easier to color by outlier
plt.title("Age vs. Charges")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title="Cook's Outlier")
#dh.save_image('outliers_age_v_charges', models_output_dir)

# Nonsmoker age vs. charges
nonsmoker_outlier_df = orig_data_w_outlier[orig_data_w_outlier['smoker']=='no']
sns.lmplot(x='age', y='charges', hue="outlier", data=nonsmoker_outlier_df, ci=None, fit_reg=False, legend=False) 
plt.title("Age vs. Charges in nonsmokers")
#dh.save_image('outliers_age_v_charges_nonsmoker', models_output_dir)

# Obese smoker age vs. charges
ob_smoker_outlier_df = orig_data_w_outlier[(orig_data_w_outlier['smoker']=='yes') & (orig_data_w_outlier['bmi_>=_30']=='yes')]
sns.lmplot(x='age', y='charges', hue="outlier", data=ob_smoker_outlier_df, ci=None, fit_reg=False, legend=False)
plt.title("Age vs. Charges in obese smokers")
#dh.save_image('outliers_age_v_charges_ob_smoker', models_output_dir)

# Nonobese smoker age vs. charges
nonob_smoker_outlier_df = orig_data_w_outlier[(orig_data_w_outlier['smoker']=='yes') & (orig_data_w_outlier['bmi_>=_30']=='no')]
sns.lmplot(x='age', y='charges', hue="outlier", data=nonob_smoker_outlier_df, ci=None, fit_reg=False, legend=False)
plt.title("Age vs. Charges in obese smokers")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title="Cook's Outlier")
#dh.save_image('outliers_age_v_charges_nonob_smoker', models_output_dir)

# In the next two graphs, the outliers are scattered around, there is no obvious grouping
# Smoker bmi vs. charges
smoker_outlier_df = orig_data_w_outlier[orig_data_w_outlier['smoker'] >= 'yes']
sns.lmplot(x='bmi', y='charges', hue="outlier", data=smoker_outlier_df)#, ci=None, fit_reg=False)
plt.plot()

# Children vs. charges
sns.lmplot(x='children', y='charges', hue="outlier", data=orig_data_w_outlier)#, ci=None, fit_reg=False)
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
#dh.save_image('perc_outlier_subcat', models_output_dir)

# Subcategory of 4 children has 15% outliers whereas basically all other subcategories range between 5-8%
# You can also see in 'Categorical Variable Relationships with Target' figure that samples with 4 kids 
# have a different distribution than the rest
# However, that only represents 4 outliers out of 85, so unsurprisingly, further exploration didn't lead anywhere


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
merge_newX_y = pd.concat([new_X_4_2, y, outlier_df['outlier']], axis=1)
no_outliers_df = merge_newX_y[merge_newX_y['outlier']=='no']

# Reset the index of the no_outliers_df DataFrame to allow for easier combination
# with other series in future functions
# Don't need this anymore as I changed the pd.concat in sm_lr_model_results_subgrouped() to use an inner join
# no_outliers_df.reset_index(drop=True, inplace=True)

# Separate target from predictors
no_outliers_y = no_outliers_df['charges']
no_outliers_X = no_outliers_df.drop(['charges', 'outlier'], axis=1)

# Plot model
title_5 = 'removed outliers'
model_name_5 = 'no_out'
file_name_5 = '5_no_outliers'
sm_lin_reg_5, sm_y_pred_5, het_results_5 = fit_lr_model_results(no_outliers_X, no_outliers_y, title_5, 
                                                                subgroup=True, ob_smoke_series=ob_smoke_series, 
                                                                save_img=False, filename_unique=file_name_5)

# Organize model performance metrics
summary_df_5 = sm_results_to_df(sm_lin_reg_5.summary())
coeff_5 = pd.Series(summary_df_5['coef'], name=model_name_5)
sm_lr_results_5 = pd.Series(dh.evaluate_model_sm(no_outliers_y, sm_y_pred_5, sm_lin_reg_5, het_results_5), name=model_name_5)

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
cooks_cutoff_5 = 4 / (len(cooks_5) - (no_outliers_X.shape[1] - 1) - 1) # 0.00322

outlier_df_5 = no_outliers_X.copy()
outlier_df_5['cooks'] = cooks_5
outlier_df_5['outlier'] = outlier_df_5['cooks'] > cooks_cutoff_5
outlier_dict = {False:'no', True:'yes'}
outlier_df_5['outlier'] = outlier_df_5['outlier'].map(outlier_dict)

num_outliers_5 = outlier_df_5[outlier_df_5['outlier'] == 'yes'].shape[0] # 29 -> 34 after removing old variables (still 119 total)
perc_outliers_5 = num_outliers_5 / outlier_df_5.shape[0] # 0.0232 -> 0.0271 after removing old variables
outlier_df_5['true_values'] = y
outlier_df_5['y_pred'] = sm_y_pred_5
outlier_df_5['stud_resid'] = sm_lin_reg_5.get_influence().resid_studentized_internal

# Visualize Cook's Distances
plt.title("Cook's Distance Plot (#2)")
plt.stem(range(len(cooks_5)), cooks_5, markerfmt=",")
plt.plot([0, len(cooks_5)], [cooks_cutoff_5, cooks_cutoff_5], color='darkblue', linestyle='--', label='4 / (N-k-1)')
plt.xlabel("Observation")
plt.ylabel("Cook's Distance")
plt.legend(title="Cook's Distance Cutoff", loc="upper left")
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

# Reset the index of the no_outliers_df DataFrame to allow for easier combination
# with other series in future functions
#no_outliers_df_2.reset_index(drop=True, inplace=True)

# Separate target from predictors
no_outliers_y_2 = no_outliers_df_2['charges']
no_outliers_X_2 = no_outliers_df_2.drop(['charges', 'outlier'], axis=1)

# Plot model
title_6 = 'removed outliers x2'
model_name_6 = 'no_out_2'
file_name_6 = '6_no_outliers_2'
sm_lin_reg_6, sm_y_pred_6, het_results_6 = fit_lr_model_results(no_outliers_X_2, no_outliers_y_2, title_6, 
                                                                ob_smoke_series=ob_smoke_series, subgroup=True, 
                                                                save_img=False, filename_unique=file_name_6)

# Organize model performance metrics
summary_df_6 = sm_results_to_df(sm_lin_reg_6.summary())
coeff_6 = pd.Series(summary_df_6['coef'], name=model_name_6)
sm_lr_results_6 = pd.Series(dh.evaluate_model_sm(no_outliers_y_2, sm_y_pred_6, sm_lin_reg_6, het_results_6), name=model_name_6)

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
#dh.save_image('influence_plot_1', models_output_dir)
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
#dh.save_image('influence_plot_2', models_output_dir)
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

# # Q-Q plot (default is normal dist)
# def my_qq(data, my_data_str='Residuals', dist_obj=stats.norm, fit_params=None, dist_str='Normal Dist', 
#           ax=None, y=1, save_img=False, img_filename=None): 
    
#     if not fit_params:
#         # Fit my data to dist_obj and get fit parameters
#         fit_params = dist_obj.fit(data)
    
#     # Specify scipy distribution shape, location, and scale based on the parameters calculated from fit()
#     loc = fit_params[-2]
#     scale = fit_params[-1]
#     shape_params = fit_params[:-2]
    
#     # Q-Q Plot
#     qqplot(data, line='45', fit=False, dist=dist_obj, loc=loc, scale=scale, distargs=shape_params, ax=ax)
    
#     if not ax:
#         ax = plt.gca()

#     ax.set_xlabel('Theoretical Quantiles')
#     ax.set_ylabel('Sample Quantiles')
#     ax.set_title(f'Q-Q Plot {my_data_str} vs. {dist_str}', y=y)
    
#     if save_img:
#         dh.save_image(img_filename, models_output_dir)

#     if not ax:
#         plt.show()

# # Plots a scipy distribution vs. histogram of my_data
# def hist_vs_dist_plot(my_data, my_data_str='Residuals', dist_obj=stats.norm, fit_params=None, dist_str='Normal Dist', 
#                       bins=200, ax=None, textbox_str=None, save_img=False, img_filename=None):    
    
#     if not fit_params:
#         # Fit my data to dist_obj and get fit parameters
#         fit_params = dist_obj.fit(my_data)
    
#     # Specify scipy distribution shape, location, and scale based on the parameters calculated from fit()
#     loc = fit_params[-2]
#     scale = fit_params[-1]
#     shape_params = fit_params[:-2]
    
#     # Specify scipy distribution shape, location, and scale based on the parameters calculated from fit() above
#     rv = dist_obj(*shape_params, loc, scale)
    
#     # Use the distribution to create x values for the plot
#     # ppf() is the inverse of cdf(). So if cdf(10) = 0.1, then ppf(0.1)=10
#     # ppf(0.1) is the x-value at which 10% of the values are less than or equal to it
#     x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
    
#     if not ax:
#         ax = plt.gca()
    
#     # Plot distribution on top of histogram of charges in order to compare
#     ax.hist(my_data, bins=bins, density=True, histtype='stepfilled', alpha=0.9, label=my_data_str)
#     ax.plot(x, rv.pdf(x), 'r-', lw=2.5, alpha=1, label=dist_str)
#     ax.set_title(f'{my_data_str} vs. {dist_str}', y=1.05)
#     ax.set_xlabel(f'{my_data_str}')
    
#     if textbox_str:
#         # Add normality test interpretation text
#         box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.8}
#         ax.text(1.05, 0.99, textbox_str, bbox=box_style, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left') 
    
    
#     ax.legend()
    
#     if save_img:
#         dh.save_image(img_filename, models_output_dir)

# # Plot both qq and hist vs. dist plots in same figure
# def plot_qq_hist_dist_combined(my_data, my_data_str='Residuals', dist_obj=stats.norm, dist_str='Normal Dist', 
#                                bins=50, textbox_str=None, fig_title=None, title_fontsize = 24, figsize=(10, 5), save_img=False, img_filename=None):
    
#     # Create figure, gridspec, list of axes/subplots mapped to gridspec location
#     fig, gs, ax_array_flat = dh.initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=figsize)

#     # Fit my data to dist_obj and get fit parameters
#     fit_params = dist_obj.fit(my_data)
    
#     # Plot Q-Q, add to figure
#     my_qq(my_data, my_data_str=my_data_str, dist_obj=dist_obj, fit_params=fit_params, 
#           ax=ax_array_flat[0], y=1.05) # Increase title space to match hist_vs_dist_plot()
    
#     # Plot hist vs. dist, add to figure
#     hist_vs_dist_plot(my_data, my_data_str=my_data_str, dist_obj=dist_obj, fit_params=fit_params, 
#                       dist_str=dist_str, bins=bins, ax=ax_array_flat[1], textbox_str=textbox_str)
    
#     # Figure title
#     if fig_title:
#         fig.suptitle(fig_title, fontsize=title_fontsize)
#     else:
#         fig.suptitle(f'{my_data_str} vs. {dist_str}', fontsize=title_fontsize)
#     fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
    
#     if save_img:
#         dh.save_image(img_filename, models_output_dir)
#     plt.show()



# ==========================================================
# Test for normality before removing Cook's outliers
# ==========================================================

resid4 = sm_lin_reg_4_2.resid_pearson
#resid4 = sm_lin_reg_4.resid
#resid4 = sm_lin_reg_4.get_influence().resid_studentized_internal

# =============================
# Statistical test for normality
# =============================
# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/

# Shapiro-Wilk test for normality
# not useful for big samples(>5000), since it tends to reject normality too often. Not an issue here
sw_stat4, sw_pval4 = stats.shapiro(resid4)

# D’Agostino’s K-squared test
dk_stat4, dk_pval4 = stats.normaltest(resid4)

# Anderson-Darling Normality Test
ad_stat4, ad_critvals4, ad_siglevels4 = stats.anderson(resid4, dist='norm')

# Chi-Square Normality Test
cs_stat4, cs_pval4 = stats.chisquare(resid4)

# Jarque–Bera test for Normality
js_stat4, js_pval4 = stats.jarque_bera(resid4)

# Kolmogorov-Smirnov test for Normality
ks_stat4, ks_pval4 = stats.kstest(resid4, 'norm')
# According to KS-table, for alpha of 0.05 and with n > 40, we want the test-statistic to be 
# less than 1.36/sqrt(n). This will give us 95% confidence that our data comes from the 
# given distribution
# https://oak.ucc.nau.edu/rh83/Statistics/ks1/
ks_stat_cutoff = 1.36 / np.sqrt(len(dataset['charges'])) # = (1.36 / sqrt(1338)) = 0.0372

# Lilliefors Test for Normality 
# Same as Kolmogorov-Smirnov?
lt_stat4, lt_pval4 = lilliefors(resid4, dist='norm')

# Function combining above normaly tests and interpreting results
normal_results4, normal_interpret4, nml_interpret_txt4 = dh.normality_tests(resid4)

# Q-Q plot and Residual Histogram vs. Normal
qqhist_filename_1 = 'qqhist1_orig'
dh.plot_qq_hist_dist_combined(resid4, fig_title='Residual Distribution', textbox_str=nml_interpret_txt4, 
                           save_img=False, img_filename=qqhist_filename_1, save_dir=models_output_dir)

# Plot y and predicted y histograms
plt.hist(y, bins=50, density=True, label='charges', alpha=0.5)
plt.hist(sm_y_pred_4, bins=50, density=True, label='pred charges', alpha=0.5)
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
dh.plot_qq_hist_dist_combined(resid5, fig_title='Residual Dist After Outlier Removal',  textbox_str=nml_interpret_txt5, 
                           save_img=False, img_filename=qqhist_filename_3, save_dir=models_output_dir)


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
dh.plot_qq_hist_dist_combined(resid6, fig_title='Residual Dist After Outlier Removal x2', textbox_str=nml_interpret_txt6,
                           save_img=False, img_filename=qqhist_filename_4, save_dir=models_output_dir)

# ==========================================================
# I tried applying nonlinear transformations to variables rather than outlier removal
# But independent variables are already close to normal or uniform distributions, so this didn't change much
# ==========================================================

# Plot distribution of nontransformed y
sns.distplot(dataset['charges'], bins=50)
# plt.hist(y, bins=50, density=True, alpha=0.8)
# sns.kdeplot(x=y, shade=True)

# Plot nontransformed y and predicted y histograms
plt.hist(y, bins=50, density=True, label='charges transformed', alpha=0.5)
plt.hist(sm_y_pred_4, bins=50, density=True, label='pred charges', alpha=0.5)
plt.legend()

# =============================
# Box-Cox transformation of y
# =============================
# New dataset
new_X_7 = new_X_4_2.copy()

# Boxcox 'charges'
y_bc, lambd = stats.boxcox(y)

# Plot distribution of 'charges' boxcox transformed
sns.distplot(y_bc, bins=50)
plt.title('Charges Box-Cox Transformed', fontsize=20, y=1.04)
plt.xlabel('charges')
#dh.save_image('charges_boxcox', models_output_dir, dpi=300, bbox_inches='tight', pad_inches=0.1)

# Plot model
title_7 = 'box-cox charges'
model_name_7 = 'normalized [charges]'
file_name_7 = '7_bc_charges'
sm_lin_reg_7, sm_y_pred_7, het_results_7 = fit_lr_model_results(new_X_7, y_bc, title_7, subgroup=True, 
                                                                ob_smoke_series=ob_smoke_series,
                                                                save_img=False, filename_unique=file_name_7)

# Function combining normality tests and interpreting results
normal_results7, normal_interpret7, nml_interpret_txt7 = dh.normality_tests(sm_lin_reg_7.resid_pearson)

# Q-Q plot and Residual Histogram vs. Normal
qqhist_filename_2 = 'qqhist2_boxcox_y'
dh.plot_qq_hist_dist_combined(sm_lin_reg_7.resid_pearson, fig_title='Residual Dist After Normalizing Target', 
                           textbox_str=nml_interpret_txt7, save_img=False, img_filename=qqhist_filename_2, 
                           save_dir=models_output_dir)



# =============================
# Other transformation of y
# =============================

# Log, sqrt, cube root transform
y_log = np.log(y)
y_sqrt = np.sqrt(y)
y_cbrt = np.cbrt(y)

# Plot distribution of 'charges' transformed
sns.distplot(y_log, bins=50)
plt.title('Charges Log Transformed', fontsize=20, y=1.04)
plt.xlabel('charges')

sns.distplot(y_sqrt, bins=50)
plt.title('Charges Square Root Transformed', fontsize=20, y=1.04)
plt.xlabel('charges')

sns.distplot(y_cbrt, bins=50)
plt.title('Charges Cube Root Transformed', fontsize=20, y=1.04)
plt.xlabel('charges')

# Plot models
title_8 = 'log charges'
model_name_8 = 'log [charges]'
file_name_8 = '8_log_charges'
sm_lin_reg_8, sm_y_pred_8, het_results_8 = fit_lr_model_results(new_X_7, y_log, title_8, subgroup=True, 
                                                                ob_smoke_series=ob_smoke_series,
                                                                save_img=False, filename_unique=file_name_8)

title_9 = 'sqrt charges'
model_name_9 = 'sqrt [charges]'
file_name_9 = '9_sqrt_charges'
sm_lin_reg_9, sm_y_pred_9, het_results_9 = fit_lr_model_results(new_X_7, y_sqrt, title_9, subgroup=True, 
                                                                ob_smoke_series=ob_smoke_series,
                                                                save_img=False, filename_unique=file_name_9)

title_91 = 'cbrt charges'
model_name_91 = 'cbrt [charges]'
file_name_91 = '91_cbrt_charges'
sm_lin_reg_91, sm_y_pred_91, het_results_91 = fit_lr_model_results(new_X_7, y_cbrt, title_91, subgroup=True, 
                                                                ob_smoke_series=ob_smoke_series,
                                                                save_img=False, filename_unique=file_name_91)



# Function combining normality tests and interpreting results
normal_results8, normal_interpret8, nml_interpret_txt8 = dh.normality_tests(sm_lin_reg_8.resid_pearson)
normal_results9, normal_interpret9, nml_interpret_txt9 = dh.normality_tests(sm_lin_reg_9.resid_pearson)
normal_results91, normal_interpret91, nml_interpret_txt91 = dh.normality_tests(sm_lin_reg_91.resid_pearson)

# Q-Q plot and Residual Histogram vs. Normal
qqhist_filename_3 = 'qqhist3_log_y'
dh.plot_qq_hist_dist_combined(sm_lin_reg_8.resid_pearson, fig_title='Residual Dist After Log-Transform Target', 
                           textbox_str=nml_interpret_txt8, save_img=False, img_filename=qqhist_filename_3, 
                           save_dir=models_output_dir)

qqhist_filename_4 = 'qqhist4_sqrt_y'
dh.plot_qq_hist_dist_combined(sm_lin_reg_9.resid_pearson, fig_title='Residual Dist After Sqrt-Transform Target', 
                           textbox_str=nml_interpret_txt9, save_img=False, img_filename=qqhist_filename_4, 
                           save_dir=models_output_dir)

qqhist_filename_5 = 'qqhist5_cbrt_y'
dh.plot_qq_hist_dist_combined(sm_lin_reg_91.resid_pearson, fig_title='Residual Dist After Cbrt-Transform Target', 
                           textbox_str=nml_interpret_txt91, save_img=False, img_filename=qqhist_filename_5, 
                           save_dir=models_output_dir)