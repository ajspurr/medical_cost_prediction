import sys
import numpy as np
import pandas as pd
from os import chdir
import seaborn as sns 
import scipy.stats as ss
import matplotlib.pyplot as plt
from pathlib import PureWindowsPath, Path

# Read in data
project_dir = PureWindowsPath(r"D:\GitHubProjects\medical_cost_prediction\\")
chdir(project_dir)
dataset = pd.read_csv('./input/insurance.csv')
eda_output_dir = Path(project_dir, Path('./output/eda'))

# Import my data science helper functions (relative dir based on project_dir)
my_module_dir = str(Path.resolve(Path('../my_ds_modules')))
sys.path.insert(0, my_module_dir)
import ds_helper as dh

# ====================================================================================================================
# EXPLORATORY DATA ANALYSIS 
# ====================================================================================================================
print("\nDATASET SHAPE:")
print(dataset.shape)
print("\nCOLUMN INFO:")
print(dataset.info())
pd.set_option("display.max_columns", len(dataset.columns))

print("\nBASIC INFORMATION NUMERICAL VARIABLES:")
print(dataset.describe())
print("\nDATA SAMPLE:")
print(dataset.head())
pd.reset_option("display.max_columns")
 
# =============================
# Explore target 
# =============================
# size includes NaN values, count does not
print("\nTARGET SUMMARY:")
print(dataset['charges'].agg(['size', 'count', 'nunique', 'unique']))
# Count of each unique value
print("\nVALUE COUNTS:")
print(dataset['charges'].value_counts())
# Total null values
print("\nTOTAL NULL VALUES:")
print(dataset['charges'].isnull().sum())

# =============================
# Explore features
# =============================
feature_summary = pd.DataFrame()
feature_summary['Data Type'] = dataset.dtypes
feature_summary['Unique Values'] = dataset.nunique()
feature_summary['Missing Values'] = dataset.isnull().sum()
feature_summary['Percent Missing'] = round((feature_summary['Missing Values'] / len(dataset.index)) * 100, 2)

print("\nDATASET SHAPE:")
print(dataset.shape)
print('\nFEATURE SUMMARY')
print(feature_summary)

# No missing values

# =============================
# Create images summarizing dataset
# =============================
# Image versions of dataset.shape
dh.df_shape_to_img(dataset, h_spacing_between_numbers=0.45)
#dh.save_image('data_overview', eda_output_dir, dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

# Image versions of feature_summary
# Row indeces normally not includes, rather that rewrite the function, I made them the first column
feature_summary.insert(0, 'Feature', feature_summary.index)

ax = dh.render_mpl_table(feature_summary, header_columns=0, col_width=2.9, header_color='#2693d7')
ax.set_title('Feature Summary:', fontdict={'fontsize':26}, loc='left', weight='bold', pad=20)
#dh.save_image('feature_summary', eda_output_dir, dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

# =============================
# Organize features
# =============================
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

# Create list of categorical + ordinal variables for certain data visualizations
cat_ord_cols = categorical_cols.copy()
cat_ord_cols.append('children')

# Create list of continuous variables with target and one without target
cont_cols = numerical_cols.copy()
cont_cols.remove('children')
cont_cols_w_target = cont_cols.copy()
cont_cols_w_target.append('charges')


# =======================================================================================
# Feature engineering
# =======================================================================================

# Based on EDA below, BMI has an impact on charges. I will create a new categorical feature for BMI.
# I had originally used the cutoff of average BMI (30.7), which is extremely close to the 
# cutoff for clinical obesity, which is 30. I will use 30 as it has more clinical significance. 

dataset['bmi_>=_30'] = dataset['bmi'] >= 30
bmi_dict = {False:'no', True:'yes'}
dataset['bmi_>=_30'] = dataset['bmi_>=_30'].map(bmi_dict)

# Add the new feature to the columns lists
categorical_cols.append('bmi_>=_30')
cat_ord_cols.append('bmi_>=_30')

# ====================================================================================================================
# Visualize data
# ====================================================================================================================

# =======================================================================================
# Functions and global variable creation
# =======================================================================================

# Standardize image saving parameters
def save_image(filename, dir=eda_output_dir, dpi=300, bbox_inches='tight'):
    plt.savefig(dir/filename, dpi=dpi, bbox_inches=bbox_inches)
    print("\nSaved image to '" + str(dir/filename) +"'\n")

# Create dictionary of formatted column names  to be used for
# figure labels (title() capitalizes every word in a string)
formatted_cols = {}
for col in dataset.columns:
    formatted_cols[col] = col.replace('_', ' ').title()
formatted_cols['bmi'] = 'BMI'
formatted_cols['bmi_>=_30'] = 'BMI >= 30'

# Function returning the formatted version of column name
def format_col(col_name):
    return formatted_cols[col_name]

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


# =======================================================================================
# Categorical variables
# =======================================================================================

# Categorical data bar charts, total count of each category
for col in cat_ord_cols:
    sns.barplot(x=dataset[col].value_counts().index, y=dataset[col].value_counts())
    plt.title(format_col(col) + ' Count')
    plt.ylabel('Count')
    plt.xlabel(format_col(col))
    #save_filename = 'counts_' + col
    #save_image(output_dir, save_filename, bbox_inches='tight')
    plt.show()
    
   
# =============================
# Combine into one figure
# =============================
# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=5, figsize=(12, 4))

# Loop through categorical variables, plotting each in the figure
i = 0
for col in cat_ord_cols:
    axis = ax_array_flat[i]
    sns.barplot(x=dataset[col].value_counts().index, y=dataset[col].value_counts(), ax=axis)
    axis.set_title(format_col(col) + ' Count')
    axis.set_xlabel(format_col(col))
    
    # Rotate x-axis tick labels so they don't overlap
    if col == 'region':
        plt.setp(axis.get_xticklabels(), rotation=30, horizontalalignment='right')
    
    # Only want to label the y-axis on the first subplot of each row
    if i != 0:
        axis.set_ylabel('')
    else:
        axis.set_ylabel('Count')
    i += 1

# Finalize figure formatting and export
fig.suptitle('Categorical Variable Counts', fontsize=18)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'combined_cat_counts'
#save_image(save_filename)
plt.show()


# =============================
# Relationship between categorical data and target
# =============================

# Boxplots
for col in cat_ord_cols:
    sns.boxplot(data=dataset, x=col, y='charges')
    plt.title(format_col(col) + ' vs. Charges')
    plt.show()

# Distributions
for col in cat_ord_cols:
    alpha_increment = 1 / len(dataset[col].unique())
    alpha = 1
    for category in dataset[col].unique():
        sns.kdeplot(data=dataset[dataset[col]==category], x='charges', shade=True, alpha=alpha, label=category)
        alpha = alpha - alpha_increment
    plt.title('Distribution of Charges by ' + format_col(col))
    plt.legend()   
    plt.show()


# =============================
# Combine categorical variable relationships with target into one figure
# =============================
# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=2, num_cols=5, figsize=(18, 8))

# Loop through categorical variables, plotting each in the figure
i = 0
for col in cat_ord_cols:
    # Boxplot
    axis1 = ax_array_flat[i]
    sns.boxplot(data=dataset, x='charges', y=col, orient='h', ax=axis1)
    axis1.set_title(format_col(col) + ' vs. Charges')
    axis1.set_ylabel(format_col(col))
    axis1.set_xlabel('Charges')
    
    # Change y-label axis tick rotation to save space
    if col == 'region':
        plt.setp(axis1.get_yticklabels(), rotation=65, verticalalignment='center')
    
    # Distributions
    axis2 = ax_array_flat[i+len(cat_ord_cols)]
    alpha_increment = 1 / len(dataset[col].unique())
    alpha = 1
    
    categories = dataset[col].unique()
    if dataset[col].dtype == 'int64':
        categories = np.sort(categories)
    
    for category in categories:
        sns.kdeplot(data=dataset[dataset[col]==category], x='charges', shade=True, alpha=alpha, label=category, ax=axis2)
        alpha = alpha - alpha_increment
    axis2.set_title('Distribution of Charges by ' + format_col(col), y=1.04)
    axis2.set_xlabel('Charges')
    axis2.legend() 
    
    # Change y-label axis tick rotation to save space
    if col == 'children':
        plt.setp(axis2.get_yticklabels(), rotation=65, verticalalignment='center')
        print(axis2.get_yticklabels())
            
    # Only want to label the y-axis on the first subplot of bottom row
    if i != 0:
        axis2.set_ylabel('')

    i += 1

# Finalize figure formatting and export
fig.suptitle('Categorical Variable Relationships with Target', fontsize=26)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'cat_variables_vs_target'
#save_image(save_filename)
plt.show()

# =============================
# Violin plots to better visualize distribution of charges subgrouped by all categorical variables
# =============================
# List of all categorical variables which are dichotomous (violin plot can only have two hues values)
cat_col_2_val = ['sex', 'smoker', 'bmi_>=_30']

for cat_var1 in cat_ord_cols:
    for cat_var2 in cat_col_2_val:
        if cat_var1 != cat_var2:
            sns.violinplot(x=cat_var1, y='charges', data=dataset, split=True, hue=cat_var2)
            plt.show()
            
# Combine all violin plots info one figure
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=3, num_cols=4, figsize=(16, 8))
i = 0
for cat_var1 in cat_ord_cols:
    for cat_var2 in cat_col_2_val:
        if cat_var1 != cat_var2:
            axis = ax_array_flat[i]
            sns.violinplot(x=cat_var1, y='charges', data=dataset, split=True, hue=cat_var2, ax=axis)
            print(i)
            i+=1

fig.suptitle('Categorical Variable Violin Plots', fontsize=26)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'violin_cat_var'
#save_image(save_filename)
plt.show()            

# =============================
# Further exploration bimodal distribution of smokers
# =============================
smokers_data = dataset[dataset['smoker']=='yes']

# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=2, figsize=(10, 5))

# Distribution of Charges in Smokers
axis1 = ax_array_flat[0]
sns.kdeplot(data=smokers_data, x='charges', shade=True, ax=axis1)
axis1.set_title('Distribution of Charges in Smokers', fontsize=16, y=1.04)
axis1.set_xlabel('Charges')

# Distribution of Charges in Smokers by BMI
axis2 = ax_array_flat[1]
sns.kdeplot(data=smokers_data[smokers_data['bmi_>=_30'] == 'no'], x='charges', 
            shade=True, alpha=1, label='BMI < 30', ax=axis2)
sns.kdeplot(data=smokers_data[smokers_data['bmi_>=_30'] == 'yes'], x='charges', 
            shade=True, alpha=0.5, label='BMI >= 30', ax=axis2)
axis2.legend() 
axis2.set_title('Distribution of Charges in Smokers (by BMI)', fontsize=16, y=1.04)
axis2.set_xlabel('Charges')
axis2.set_ylabel('')

# Finalize figure formatting and export
#fig.suptitle('Exploration Bimodal Distribution of Charges in Smokers', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'smoker_dist_by_bmi'
#save_image(save_filename)
plt.show()


# =======================================================================================
# Numerical variables
# =======================================================================================

# =============================
# Plot target (charges) on its own
# =============================
sns.distplot(dataset['charges'])
plt.title('Charges Histogram', fontsize=20, y=1.04)
#save_filename = 'hist_charges'
#save_image(save_filename)
plt.show()

# =============================
# Numerical data histograms
# =============================
for col in numerical_cols:
    #sns.distplot used to plot the histogram and fit line, but it's been deprecated to displot or histplot which don't 
    sns.distplot(dataset[col])
    plt.title(format_col(col) + ' Histogram')
    #save_filename = 'hist_' + col
    #save_image(output_dir, save_filename)
    plt.show()

# =============================
# Numerical data relationships with target 
# =============================
# lmplots
pearsons = dataset.corr(method='pearson').round(2)
spearmans = dataset.corr(method='spearman').round(2) 
box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.9}
for col in numerical_cols:
    fgrid = sns.lmplot(x=col, y="charges", data=dataset)
    ax = fgrid.axes[0,0]
    plt.title(format_col(col) + ' vs. Charges')
    if col=='children':
        textbox_text = "Spearmans's ρ = %0.2f" %spearmans[col].loc['charges']
    else:
        textbox_text = "Pearson's r = %0.2f" %pearsons[col].loc['charges']
    plt.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax.transAxes, 
             verticalalignment='top', horizontalalignment='right')
    #save_filename = f'lmplot_{col}_vs_charges'
    #save_image(save_filename)

# regplots, because you can't add lmplots to gridspec
pearsons = dataset.corr(method='pearson').round(2)
spearmans = dataset.corr(method='spearman').round(2) 
box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.9}
for col in numerical_cols:
    fgrid = sns.regplot(x=col, y="charges", data=dataset)
    ax = fgrid.axes
    plt.title(format_col(col) + ' vs. Charges')
    if col=='children':
        textbox_text = "Spearmans's ρ = %0.2f" %spearmans[col].loc['charges']
    else:
        textbox_text = "Pearson's r = %0.2f" %pearsons[col].loc['charges']
    plt.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax.transAxes, 
             verticalalignment='top', horizontalalignment='right')
    
    #save_filename = f'regplot_{col}_vs_charges'
    #save_image(save_filename)
    plt.show()

# jointplots
for col in numerical_cols:
    p = sns.jointplot(x=col, y="charges", data = dataset, kind='reg')
    p.fig.suptitle(format_col(col) + ' vs. Charges', y=1.03)
    p.set_axis_labels(format_col(col), 'Charges')
    #plt.title(format_col(col) + ' vs. Charges')
    #plt.legend()
    #save_filename = 'hist_by_stroke-' + col
    #save_image(save_filename)    
    plt.show()

# =============================
# Combine numerical variable graphs into one figure
# =============================

# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=2, num_cols=3, figsize=(16, 8))

# Calculate correlation coefficients
pearsons = dataset.corr(method='pearson').round(2)
spearmans = dataset.corr(method='spearman').round(2)

# Format text box
box_style = {'facecolor':'white', 'boxstyle':'round', 'alpha':0.9}

# Loop through categorical variables, plotting each in the figure
i = 0
for col in numerical_cols:
    # Numerical variable distributions
    axis1 = ax_array_flat[i]
    sns.distplot(dataset[col], ax=axis1)
    axis1.set_title(format_col(col) + ' Histogram')
    
    # Numerical variable relationship to target
    axis2 = ax_array_flat[i+3]
    
    fgrid = sns.regplot(x=col, y="charges", data=dataset, ax=axis2)
    ax = fgrid.axes
    axis2.set_title(format_col(col) + ' vs. Charges')
    if col=='children':
        textbox_text = "Spearmans's ρ = %0.2f" %spearmans[col].loc['charges']
    else:
        textbox_text = "Pearson's r = %0.2f" %pearsons[col].loc['charges']
    axis2.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax.transAxes, 
             verticalalignment='top', horizontalalignment='right')
    
    # Only want to label the y-axis on the first subplot of each row
    if i == 0:
        axis2.set_ylabel('Charges')
    else:
        axis1.set_ylabel('')
        axis2.set_ylabel('')
    
    i+=1

# Finalize figure formatting and export
fig.suptitle('Numerical Variable Exploration', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'num_var_combined_2'
#save_image(save_filename)
plt.show()


# ==========================================================
# Further explore numerical variables and smoking
# ==========================================================

# =============================================
# Age vs. Charges
# =============================================

# Age vs. Charges, grouped by smoking status
sns.jointplot(x='age', y="charges", data = dataset, hue='smoker')
plt.show()
sns.jointplot(x='age', y="charges", data = dataset, kind='kde', hue='smoker')
plt.show()

# There is obvious grouping of charges by smoking status, will separate out both groups
smokers_data = dataset[dataset['smoker']=='yes'].copy()
nonsmokers_data = dataset[dataset['smoker']=='no'].copy()

# Calculate pearsons in smokers and nonsmokers, obese and nonobese
pearson_smokers = smokers_data.corr(method='pearson').round(2)
pearson_smokers_age_charge = pearson_smokers['age'].loc['charges']
pearson_nonsmokers = nonsmokers_data.corr(method='pearson').round(2)
pearson_nonsmokers_age_charge = pearson_nonsmokers['age'].loc['charges']

# Within smokers group, calculate pearsons in obese and nonobese individuals
obese_df = smokers_data[smokers_data['bmi_>=_30']=='yes'].copy()
nonobese_df = smokers_data[smokers_data['bmi_>=_30']=='no'].copy()
pearson_obese = obese_df.corr(method='pearson').round(2)
pearson_obese_age_charge = pearson_obese['age'].loc['charges']
pearson_nonobese = nonobese_df.corr(method='pearson').round(2)
pearson_nonobese_age_charge = pearson_nonobese['age'].loc['charges']


sns.lmplot(x='age', y='charges', hue="smoker", data=dataset, legend=False, line_kws={'color': 'green'})
ax = plt.gca()
ax.legend(title='Smoker', loc='upper right')
leg = ax.get_legend()
labels = leg.get_texts()
for legend_label in labels:
    if legend_label.get_text() == 'no':
        legend_label.set_text("No (Pearson's %0.2f)" %pearson_nonsmokers_age_charge)
    else:
        legend_label.set_text("Yes (Pearson's %0.2f)" %pearson_smokers_age_charge)
plt.title("Age vs. Charges, grouped by smoking status")
#save_filename = 'age_vs_charges_grp_smoking_status'
#save_image(save_filename)
plt.show()


sns.lmplot(x='age', y='charges', data=smokers_data)
plt.title("Age vs. Charges in smokers")
#save_filename = 'age_vs_charges_smokers'
#save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()

sns.lmplot(x='age', y='charges', data=nonsmokers_data)
plt.title("Age vs. Charges in nonsmokers")
#save_filename = 'age_vs_charges_nonsmokers'
#save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()

# =============================
# Explore smokers
# =============================
# Age vs. Charges in smokers, grouped by BMI>=30
sns.lmplot(x='age', y='charges', hue="bmi_>=_30", data=smokers_data)
plt.title("Age vs. Charges in smokers, grouped by BMI")
plt.show()

# Smokers group very well by BMI. Do not group well by sex, region, or # children (I left that code out)

# Will explore different BMI cutoffs
# ===============
# BMI 30
# ===============
obese_df = smokers_data[smokers_data['bmi_>=_30']=='yes'].copy()
nonobese_df = smokers_data[smokers_data['bmi_>=_30']=='no'].copy()

pearson_obese = obese_df.corr(method='pearson').round(3)
pearson_age_charge_ob = pearson_obese['age'].loc['charges']

pearson_nonobese = nonobese_df.corr(method='pearson').round(3)
pearson_age_charge_nonob = pearson_nonobese['age'].loc['charges']

# Pearsons for age and charges in smokers: 0.67 for obese and 0.69 for nonobese

# Replot with pearsons labels
g = sns.lmplot(x='age', y='charges', hue="bmi_>=_30", data=smokers_data, legend=False)
ax = g.axes[0,0]
ax.legend(title='BMI >= 30', loc='upper right')
leg = ax.get_legend()
labels = leg.get_texts()
for legend_label in labels:
    if legend_label.get_text() == 'no':
        legend_label.set_text("No (Pearson's %0.3f)" %pearson_age_charge_nonob)
    else:
        legend_label.set_text("Yes (Pearson's %0.3f)" %pearson_age_charge_ob)
plt.title("Age vs. Charges in smokers, grouped by BMI (30)")
#save_filename = 'age_vs_charges_smokers_grp_bmi30'
#save_image(save_filename)
plt.show()

# ===============
# BMI 29
# ===============
test_smokers_df = smokers_data.copy()
test_smokers_df['bmi_>=_29'] = test_smokers_df['bmi'] >= 29
bmi_dict = {False:'no', True:'yes'}
test_smokers_df['bmi_>=_29'] = test_smokers_df['bmi_>=_29'].map(bmi_dict)

obese_df = test_smokers_df[test_smokers_df['bmi_>=_29']=='yes'].copy()
nonobese_df = test_smokers_df[test_smokers_df['bmi_>=_29']=='no'].copy()

pearson_obese = obese_df.corr(method='pearson').round(2)
pearson_age_charge_ob = pearson_obese['age'].loc['charges']

pearson_nonobese = nonobese_df.corr(method='pearson').round(2)
pearson_age_charge_nonob = pearson_nonobese['age'].loc['charges']

# Pearsons for age and charges in smokers: 0.52 for obese and 0.66 for nonobese (with obese cutoff of BMI 29)

g = sns.lmplot(x='age', y='charges', hue="bmi_>=_29", data=test_smokers_df, legend=False)
ax = g.axes[0,0]
ax.legend(title='BMI >= 29', loc='upper right')
leg = ax.get_legend()
labels = leg.get_texts()
for legend_label in labels:
    if legend_label.get_text() == 'no':
        legend_label.set_text("No (Pearson's %0.2f)" %pearson_age_charge_nonob)
    else:
        legend_label.set_text("Yes (Pearson's %0.2f)" %pearson_age_charge_ob)
plt.title("Age vs. Charges in smokers, grouped by BMI (29)")
#save_filename = 'age_vs_charges_smokers_grp_bmi29'
#save_image(save_filename)
plt.show()


# ===============
# BMI 31
# ===============
test_smokers_df['bmi_>=_31'] = test_smokers_df['bmi'] >= 31
test_smokers_df['bmi_>=_31'] = test_smokers_df['bmi_>=_31'].map(bmi_dict)

obese_df = test_smokers_df[test_smokers_df['bmi_>=_31']=='yes'].copy()
nonobese_df = test_smokers_df[test_smokers_df['bmi_>=_31']=='no'].copy()

pearson_obese = obese_df.corr(method='pearson').round(2)
pearson_age_charge_ob = pearson_obese['age'].loc['charges']

pearson_nonobese = nonobese_df.corr(method='pearson').round(2)
pearson_age_charge_nonob = pearson_nonobese['age'].loc['charges']

# Pearsons for age and charges in smokers: 0.72 for obese and 0.46 for nonobese (with obese cutoff of BMI 31)

g = sns.lmplot(x='age', y='charges', hue="bmi_>=_31", data=test_smokers_df, legend=False)
ax = g.axes[0,0]
ax.legend(title='BMI >= 31', loc='upper right')
leg = ax.get_legend()
labels = leg.get_texts()
for legend_label in labels:
    if legend_label.get_text() == 'no':
        legend_label.set_text("No (Pearson's %0.2f)" %pearson_age_charge_nonob)
    else:
        legend_label.set_text("Yes (Pearson's %0.2f)" %pearson_age_charge_ob)
plt.title("Age vs. Charges in smokers, grouped by BMI(31)")
#save_filename = 'age_vs_charges_smokers_grp_bmi31'
#save_image(save_filename)
plt.show()

# =============================
# Explore nonsmokers
# =============================
pearson_nonsmokers = nonsmokers_data.corr(method='pearson')['age'].loc['charges'].round(3)
g = sns.lmplot(x='age', y='charges', data=nonsmokers_data, line_kws={'color':'cyan'})
#ax = g.axes[0,0]
ax = plt.gca()
textbox_text = "Pearson's r = %0.3f" %pearson_nonsmokers
plt.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax.transAxes, 
         verticalalignment='top', horizontalalignment='right')
plt.title("Age vs. Charges in nonsmokers")
#save_filename = 'age_vs_charges_nonsmokers'
#save_image(save_filename)
plt.show()

# Nonsmokers do not group well by BMI, sex, region, or # children (I left that code out)

# =============================
# Test squaring the age as the shape looks almost parabolic
# =============================
new_age_data = dataset.copy()
new_age_data['age^2'] = np.power(new_age_data['age'], 2)

# Divide new dataset into relevant groups
new_age_smokers_data = new_age_data[dataset['smoker']=='yes']
new_age_nonsmokers_data = new_age_data[dataset['smoker']=='no']
new_obese_smoker_data = new_age_smokers_data[new_age_smokers_data['bmi_>=_30']=='yes']
new_nonobese_smoker_data = new_age_smokers_data[new_age_smokers_data['bmi_>=_30']=='no']

# Within smokers group, calculate pearsons in obese and nonobese individuals
new_pearson_obese_smoker = new_obese_smoker_data.corr(method='pearson').round(3)
new_pearson_obese_smoker_age_charge = new_pearson_obese_smoker['age^2'].loc['charges']
new_pearson_nonobese_smoker = new_nonobese_smoker_data.corr(method='pearson').round(3)
new_pearson_nonobese_smoker_age_charge = new_pearson_nonobese_smoker['age^2'].loc['charges']

# Calculate nonsmokers pearsons
new_pearson_nonsmokers = new_age_nonsmokers_data.corr(method='pearson').round(3)
new_pearson_nonsmokers_age_charge = new_pearson_nonsmokers['age^2'].loc['charges']

# Plot nonsmoker lmplot again
sns.lmplot(x='age^2', y='charges', data=new_age_nonsmokers_data, line_kws={'color':'cyan'})
ax = plt.gca()
textbox_text = "Pearson's r = %0.3f" %new_pearson_nonsmokers_age_charge
plt.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax.transAxes, 
         verticalalignment='top', horizontalalignment='right')
plt.title("Age^2 vs. Charges in nonsmokers")
save_filename = 'age_sq_vs_charges_nonsmokers'
save_image(save_filename)
plt.show()

# Plot smoker lmplot again
g = sns.lmplot(x='age^2', y='charges', hue="bmi_>=_30", data=new_age_smokers_data, legend=False)
ax = g.axes[0,0]
ax.legend(title='BMI >= 30', loc='upper right')
leg = ax.get_legend()
labels = leg.get_texts()
for legend_label in labels:
    if legend_label.get_text() == 'no':
        legend_label.set_text("No (Pearson's %0.3f)" %new_pearson_nonobese_smoker_age_charge)
    else:
        legend_label.set_text("Yes (Pearson's %0.3f)" %new_pearson_obese_smoker_age_charge)
plt.title("Age vs. Charges in smokers, grouped by BMI (30)")
save_filename = 'age_sq_vs_charges_smokers_grp_bmi30'
save_image(save_filename)
plt.show()



# =============================================
# Explore BMI vs. Charges
# =============================================
pearson_bmi_charges_smokers = smokers_data.corr(method='pearson')['bmi'].loc['charges'].round(2)
pearson_bmi_charges_nonsmokers = nonsmokers_data.corr(method='pearson')['bmi'].loc['charges'].round(2)

g = sns.lmplot(x='bmi', y='charges', hue="smoker", data=dataset, legend=False)
ax = g.axes[0,0]
ax.legend(title='Smoker', loc='upper right')
leg = ax.get_legend()
labels = leg.get_texts()
for legend_label in labels:
    if legend_label.get_text() == 'no':
        legend_label.set_text("No (Pearson's %0.2f)" %pearson_bmi_charges_nonsmokers)
    else:
        legend_label.set_text("Yes (Pearson's %0.2f)" %pearson_bmi_charges_smokers)
plt.title("BMI vs. Charges grouped by smoking status")
#save_filename = 'bmi_vs_charges_grp_smoking'
#save_image(save_filename)
plt.show()

# Other subgroupings don't yield anything helpful

# =============================================
# Explore Children vs. Charges
# =============================================
# Children vs. Charges with no obvious subgrouping
sns.lmplot(x='children', y='charges', hue="smoker", data=dataset)
plt.show()

sns.lmplot(x='children', y='charges', hue="bmi_>=_30", data=dataset)
plt.show()


# =======================================================================================
# Correlation between variables
# =======================================================================================
# ==========================================================
# Correlation between numerical variables
# ==========================================================
# =============================
# Pairplots and PaidGrids to visualize relationships between numerical variables
# =============================
# Use pairplot to get a sense of relationships between numerical variables
sns.pairplot(dataset)
sns.pairplot(dataset, hue="smoker")
sns.pairplot(dataset, hue="bmi_>=_30")
sns.pairplot(dataset, hue="region")

pp = sns.pairplot(dataset, hue="sex")
pp.fig.suptitle("Relationship Between Numerical Variables", y=1.03, fontsize=24)
#save_filename = 'relationship_num_var_by_sex'
#save_image(save_filename, bbox_inches='tight')
plt.show()

# Tried PairGrid, but diagonal graph kde plots don't display properly due to inappropriate y-scale
# Create an instance of the PairGrid class
grid = sns.PairGrid(data=dataset, hue='smoker')
grid = grid.map_upper(plt.scatter)
#grid = grid.map_diag(sns.kdeplot, alpha=0.5)
grid = grid.map_diag(plt.hist, alpha=0.5)
grid = grid.map_lower(sns.kdeplot)
grid = grid.add_legend()
plt.show()

# =============================
# Correlation heatmaps
# =============================
pearsons_df = dataset[num_cols_w_target].corr(method='pearson')
spearmans_df = dataset[num_cols_w_target].corr(method='spearman')

sns.heatmap(pearsons_df, annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1)
plt.title('Correlation Numerical Variables (Pearson)')
#save_filename = 'corr_num_var_pearson'
#save_image(save_filename)  
plt.show()

sns.heatmap(spearmans_df, annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1)
plt.title('Correlation Numerical Variables (Spearman)')
#save_filename = 'corr_num_var_spearman'
#save_image(save_filename)  
plt.show()

# np.trui sets all the values above a certain diagonal to 0, so we don't have redundant boxes
matrix = np.triu(pearsons_df) 
sns.heatmap(pearsons_df, annot=True, linewidth=.8, mask=matrix, cmap="Blues", vmin=0, vmax=1)
plt.show()

matrix = np.triu(spearmans_df) 
sns.heatmap(spearmans_df, annot=True, linewidth=.8, mask=matrix, cmap="Blues", vmin=0, vmax=1)
plt.show()

# Spearman with improved correlation for children vs. charges (0.07 to 0.13) and in age vs. charges (0.3 to 0.53)

# ==========================================================
# Association between categorical variables
# ==========================================================
# Credit to: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

# Calculate Cramér’s V (based on a nominal variation of Pearson’s Chi-Square Test) between two categorical featuers 'x' and 'y'
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

# New dataframe to store results for each combination of categorical variables
cramers_df = pd.DataFrame(columns=cat_ord_cols, index=cat_ord_cols)

# Loop through each paring of categorical variables, calculating the Cramer's V for each and storing in dataframe
for col in cramers_df.columns:
    for row in cramers_df.index:
        cramers_df.loc[[row], [col]] = cramers_v(dataset[row], dataset[col])

# Values default to 'object' dtype, will convert to numeric
cramers_df = cramers_df.apply(pd.to_numeric)

# Output results as heatmap
sns.heatmap(cramers_df, annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1)
plt.title("Association Between Categorical Variables (Cramér's V)")
#save_filename = 'association_cat_variables'
#save_image(save_filename)  
plt.show()

# =============================
# Further exploration association categorical variables
# =============================
# Plot catplots of categorical variables with cramers > 'cramers_cutoff'
# Loop through cramers_df diagonally to skip redundant pairings
cramers_cutoff = -1
for col in range(len(cramers_df.columns)-1):
    for row in range(col+1, len(cramers_df.columns)):
        cramers_value = cramers_df.iloc[[row], [col]].iat[0,0].round(3)
        if cramers_value > cramers_cutoff:
            column_name = cramers_df.columns[col]
            row_name = cramers_df.index[row]
            sns.catplot(data=dataset, x=column_name, hue=row_name, kind="count", legend=False)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title=row_name)
            plt.title(format_col(column_name) + ' vs. ' + format_col(row_name) + " (Cramer's=" + str(cramers_value) + ')')
            # if column_name=='sex' and row_name=='region':
            #     save_filename = 'compare_' + column_name + '_vs_' + row_name
            #     save_image(output_dir, save_filename)
            plt.show()

# ==========================================================
# Correlation between continuous and categorical variables
# ==========================================================
# Credit to: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

# Calculate correlation ratio between a categorical feature ('categories') and numeric feature ('measurements')
def correlation_ratio(categories, measurements):
    merged_df = categories.to_frame().merge(measurements, left_index=True, right_index=True)
    fcat, _ = pd.factorize(categories)
    merged_df['fcat'] = fcat
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = merged_df[merged_df['fcat']==i][measurements.name]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

# New dataframe to store results for each combination of numerical and categorical variables
corr_ratio_df = pd.DataFrame(columns=num_cols_w_target, index=categorical_cols)

# Loop through each paring of numerical and categorical variables, calculating the correlation ratio for each and storing in dataframe
for col in corr_ratio_df.columns:
    for row in corr_ratio_df.index:
        corr_ratio_df.loc[[row], [col]] = correlation_ratio(dataset[row], dataset[col])

# Values default to 'object' dtype, will convert to numeric
corr_ratio_df = corr_ratio_df.apply(pd.to_numeric)

# Output results as heatmap
sns.heatmap(corr_ratio_df, annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1)
plt.title("Correlation Ratio Between Numerical and Categorical Variables")
#save_filename = 'corr_ratio_cat_num_variables'
#save_image(save_filename)  
plt.show()

# =============================
# Further exploration correlation continuous and categorical variables
# =============================
# Plot boxplots of  continuous and categorical variables with correlation ratio > 'corr_ratio_cutoff'
corr_ratio_cutoff = 0.5
for col in corr_ratio_df.columns:
    for row in corr_ratio_df.index:
        corr_value = corr_ratio_df.loc[[row], [col]].iat[0,0].round(2)
        if corr_value > corr_ratio_cutoff:
            sns.boxplot(data=dataset, x=row, y=col)
            plt.title(format_col(col) + ' vs. ' + format_col(row) + ' (Corr Ratio=' + str(corr_value) + ')')
            #save_filename = 'relationship_' + col + '_' + row
            #save_image(output_dir, save_filename)  
            plt.show()












