import pandas as pd
import numpy as np
import scipy.stats as ss
from os import chdir
from pathlib import PureWindowsPath, Path
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Read in data
project_dir = PureWindowsPath(r"D:\GitHubProjects\medical_cost_prediction\\")
chdir(project_dir)
dataset = pd.read_csv('./input/insurance.csv')
output_dir = Path(project_dir, Path('./output/eda'))

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
feature_summary['dtype'] = dataset.dtypes
feature_summary['unique_values'] = dataset.nunique()
feature_summary['missing_values'] = dataset.isnull().sum()
feature_summary['percent_missing'] = round((feature_summary['missing_values'] / len(dataset.index)) * 100, 2)

print("\nDATASET SHAPE:")
print(dataset.shape)
print('\nFEATURE SUMMARY')
print(feature_summary)

# No missing values

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

# Create list of continuous variables for certain data visualizations
cont_cols = numerical_cols.copy()
cont_cols.remove('children')

# ==========================================================
# Feature engineering
# ==========================================================

# Based on EDA below, BMI has an impact on charges. I will create a new categorical feature for BMI.
# I had originally used the cutoff of average BMI (30.7), which is extremely close to the 
# cutoff for clinical obesity, which is 30. I will use 30 as it has more clinical significance. 

dataset['bmi_>=_30'] = dataset['bmi'] >= 30
bmi_dict = {False:'no', True:'yes'}
dataset['bmi_>=_30'] = dataset['bmi_>=_30'].map(bmi_dict)

# =======================================================================================
# Visualize data
# =======================================================================================

# ==========================================================
# Functions and global variable creation
# ==========================================================

# Standardize image saving parameters
def save_image(dir, filename, dpi=300, bbox_inches='tight'):
    plt.savefig(dir/filename, dpi=dpi, bbox_inches=bbox_inches)

# Create dictionary of formatted column names  to be used for
# figure labels (title() capitalizes every word in a string)
formatted_cols = {}
for col in dataset.columns:
    formatted_cols[col] = col.replace('_', ' ').title()
formatted_cols['bmi'] = 'BMI'

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


# ==========================================================
# Categorical variables
# ==========================================================

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
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=4, figsize=(10, 4))

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
    i += 1

# Finalize figure formatting and export
fig.suptitle('Categorical Variable Counts', fontsize=16)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'combined_cat_counts'
#save_image(output_dir, save_filename, bbox_inches='tight')
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
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=2, num_cols=4, figsize=(16, 8))

# Loop through categorical variables, plotting each in the figure
i = 0
for col in cat_ord_cols:
    # Boxplot
    axis1 = ax_array_flat[i]
    sns.boxplot(data=dataset, x='charges', y=col, orient='h', ax=axis1)
    axis1.set_title(format_col(col) + ' vs. Charges')
    axis1.set_ylabel(format_col(col))
    axis1.set_xlabel('Charges')
    
    # Distributions
    axis2 = ax_array_flat[i+4]
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
            
    # Only want to label the y-axis on the first subplot of each row
    if i != 0:
        axis2.set_ylabel('')

    i += 1

# Finalize figure formatting and export
fig.suptitle('Categorical Variable Relationships with Target', fontsize=26)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
save_filename = 'cat_variables_vs_target'
save_image(output_dir, save_filename, bbox_inches='tight')
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

# After exploring multiple variables, found that BMI could explain the bimodal distribution

# Distribution of Charges in Smokers by BMI
mean_bmi = smokers_data['bmi'].mean()
axis2 = ax_array_flat[1]
sns.kdeplot(data=smokers_data[smokers_data['bmi'] < mean_bmi], x='charges', 
            shade=True, alpha=1, label='BMI < avg (30.7)', ax=axis2)
sns.kdeplot(data=smokers_data[smokers_data['bmi'] > mean_bmi], x='charges', 
            shade=True, alpha=0.5, label='BMI > avg', ax=axis2)
axis2.legend() 
axis2.set_title('Distribution of Charges in Smokers (by BMI)', fontsize=16, y=1.04)
axis2.set_xlabel('Charges')
axis2.set_ylabel('')

# Finalize figure formatting and export
#fig.suptitle('Exploration Bimodal Distribution of Charges in Smokers', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
#save_filename = 'smoker_dist_by_bmi'
#save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()


# ==========================================================
# Numerical variables
# ==========================================================

# Plot target (charges) on its own
sns.distplot(dataset['charges'])
plt.title('Charges Histogram', fontsize=20, y=1.04)
save_filename = 'hist_charges'
save_image(output_dir, save_filename)
plt.show()

# Numerical data histograms
for col in numerical_cols:
    #sns.distplot used to plot the histogram and fit line, but it's been deprecated to displot or histplot which don't 
    sns.distplot(dataset[col])
    plt.title(format_col(col) + ' Histogram')
    #save_filename = 'hist_' + col
    #save_image(output_dir, save_filename)
    plt.show()

# Numerical data relationships with target (lmplots)
pearsons = dataset.corr(method='pearson').round(2)
spearmans = dataset.corr(method='spearman').round(2) 
box_style = {'facecolor':'white', 'boxstyle':'round'}
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

# Numerical data relationships with target (regplots, because you can't add lmplots to gridspec)
pearsons = dataset.corr(method='pearson').round(2)
spearmans = dataset.corr(method='spearman').round(2) 
box_style = {'facecolor':'white', 'boxstyle':'round'}
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
    plt.show()

# Numerical data relationships with target (joint plots)
for col in numerical_cols:
    p = sns.jointplot(x=col, y="charges", data = dataset, kind='reg')
    p.fig.suptitle(format_col(col) + ' vs. Charges', y=1.03)
    p.set_axis_labels(format_col(col), 'Charges')
    #plt.title(format_col(col) + ' vs. Charges')
    #plt.legend()
    #save_filename = 'hist_by_stroke-' + col
    #save_image(output_dir, save_filename)    
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
box_style = {'facecolor':'white', 'boxstyle':'round'}

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
save_filename = 'num_var_combined'
save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()



# ==========================================================
# Further explore numerical variables and smoking
# ==========================================================

# Age vs. Charges, grouped by smoking status
sns.jointplot(x='age', y="charges", data = dataset, hue='smoker')
plt.show()
sns.jointplot(x='age', y="charges", data = dataset, kind='kde', hue='smoker')
plt.show()

sns.lmplot(x='age', y='charges', hue="smoker", data=dataset)
plt.title("Age vs. Charges, grouped by smoking status")
#save_filename = 'age_vs_charges_grp_smoking_status'
#save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()

# There is obvious grouping of charges by smoking status, will separate out both groups
smokers_data = dataset[dataset['smoker']=='yes'].copy()
nonsmokers_data = dataset[dataset['smoker']=='no'].copy()

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

pearson_obese = obese_df.corr(method='pearson').round(2)
pearson_age_charge_ob = pearson_obese['age'].loc['charges']

pearson_nonobese = nonobese_df.corr(method='pearson').round(2)
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
        legend_label.set_text("No (Pearson's %0.2f)" %pearson_age_charge_nonob)
    else:
        legend_label.set_text("Yes (Pearson's %0.2f)" %pearson_age_charge_ob)
plt.title("Age vs. Charges in smokers, grouped by BMI (30)")
save_filename = 'age_vs_charges_smokers_grp_bmi30'
save_image(output_dir, save_filename, bbox_inches='tight')
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
save_filename = 'age_vs_charges_smokers_grp_bmi29'
save_image(output_dir, save_filename, bbox_inches='tight')
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
save_filename = 'age_vs_charges_smokers_grp_bmi31'
save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()

# =============================
# Explore nonsmokers
# =============================
pearson_nonsmokers = nonsmokers_data.corr(method='pearson')['age'].loc['charges'].round(2)
g = sns.lmplot(x='age', y='charges', data=nonsmokers_data, line_kws={'color':'cyan'})
ax = g.axes[0,0]
textbox_text = "Pearson's r = %0.2f" %pearson_nonsmokers
plt.text(0.95, 0.92, textbox_text, bbox=box_style, transform=ax.transAxes, 
         verticalalignment='top', horizontalalignment='right')

plt.title("Age vs. Charges in nonsmokers")
save_filename = 'age_vs_charges_nonsmokers'
save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()

# Nonsmokers do not group well by BMI, sex, region, or # children (I left that code out)


# =============================
# Explore BMI vs. Charges
# =============================
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
#save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()

# Other subgroupings don't yield anything helpful


# =============================
# Explore Children vs. Charges
# =============================
# Children vs. Charges with no obvious subgrouping
sns.jointplot(x='children', y="charges", data = dataset, hue='smoker')
plt.show()
sns.jointplot(x='children', y="charges", data = dataset, kind='kde', hue='smoker')
plt.show()
sns.lmplot(x='children', y='charges', hue="smoker", data=dataset)
plt.show()

sns.lmplot(x='children', y='charges', hue="bmi_>=_30", data=dataset)
plt.show()




# =======================================================================================
# Correlation between variables
# =======================================================================================

# ==========================================================
# Correlation between continuous variables
# ==========================================================

# Use pairplot to get a sense of relationship between numerical variables
sns.pairplot(dataset)
sns.pairplot(dataset, hue="smoker")

# Find correlation between variables
# np.trui sets all the values above a certain diagonal to 0, so we don't have redundant boxes
matrix = np.triu(dataset[numerical_cols].corr()) 
sns.heatmap(dataset[numerical_cols].corr(), annot=True, linewidth=.8, mask=matrix, cmap="rocket", vmin=0, vmax=1)
plt.show()

# You can also make a correlation matrix that includes the diagonal so that the color spectrum better 
# represents the more extreme values
sns.heatmap(dataset[numerical_cols].corr(), annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1)
plt.title('Correlation Between Continuous Variables')
#save_filename = 'correlation_cont_variables'
#save_image(output_dir, save_filename)  
plt.show()

# Age has the highest correlation with other continuous variables

# =============================
# Further exploration correlation continuous variables
# =============================
# Since age has the highest correlation with other variables, will plot those relationships
# Scatterplots and lineplots showing relationship between age and other continuous variables
sns.scatterplot(data=dataset, x='age', y='bmi')
plt.show()

sns.lineplot(data=dataset, x='age', y='bmi')
plt.title('Relationship Between Age and BMI')
#save_filename = 'correlation_age_bmi'
#save_image(output_dir, save_filename)  
plt.show()

sns.scatterplot(data=dataset, x='age', y='avg_glucose_level')
plt.show()

sns.lineplot(data=dataset, x='age', y='avg_glucose_level')
plt.title('Relationship Between Age and Avg Glucose Level')
#save_filename = 'correlation_age_avg_glucose_level'
#save_image(output_dir, save_filename)  
plt.show()

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
cramers_df = pd.DataFrame(columns=cat_cols_w_target, index=cat_cols_w_target)

# Loop through each paring of categorical variables, calculating the Cramer's V for each and storing in dataframe
for col in cramers_df.columns:
    for row in cramers_df.index:
        cramers_df.loc[[row], [col]] = cramers_v(dataset[row], dataset[col])

# Values default to 'object' dtype, will convert to numeric
cramers_df = cramers_df.apply(pd.to_numeric)

# Output results as heatmap
sns.heatmap(cramers_df, annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1)
plt.title("Association Between Categorical Variables (Cramér's V)")
#save_filename = 'correlation_cat_variables'
#save_image(output_dir, save_filename)  
plt.show()

# =============================
# Further exploration association categorical variables
# =============================
# Plot catplots of categorical variables with correlation ratio > 0.29
# Loop through cramers_df diagonally to skip redundant pairings
for col in range(len(cramers_df.columns)-1):
    for row in range(col+1, 8):
        cramers_value = cramers_df.iloc[[row], [col]].iat[0,0].round(2)
        if cramers_value > 0.29:
            column_name = cramers_df.columns[col]
            row_name = cramers_df.index[row]
            sns.catplot(data=dataset, x=column_name, hue=row_name, kind="count", legend=False)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title=row_name)
            plt.title(format_col(column_name) + ' vs. ' + format_col(row_name) + " (Cramer's=" + str(cramers_value) + ')')
            #save_filename = 'compare_' + column_name + '_vs_' + row_name
            #save_image(output_dir, save_filename)
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

# Grab numerical data from original dataset so that I can impute values(correlation_ratio() cannot handle unknown values)
num_df = dataset[numerical_cols]
num_df = pd.DataFrame(SimpleImputer().fit_transform(num_df), columns=numerical_cols, index=dataset.index)

# New dataframe to store results for each combination of numerical and categorical variables
corr_ratio_df = pd.DataFrame(columns=num_df.columns, index=cat_cols_w_target)

# Loop through each paring of numerical and categorical variables, calculating the correlation ratio for each and storing in dataframe
for col in corr_ratio_df.columns:
    for row in corr_ratio_df.index:
        corr_ratio_df.loc[[row], [col]] = correlation_ratio(dataset[row], num_df[col])

# Values default to 'object' dtype, will convert to numeric
corr_ratio_df = corr_ratio_df.apply(pd.to_numeric)

# Output results as heatmap
sns.heatmap(corr_ratio_df, annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1)
plt.title("Correlation Ratio Between Numerical and Categorical Variables")
#save_filename = 'correlation_cat_num_variables'
#save_image(output_dir, save_filename)  
plt.show()

# =============================
# Further exploration correlation continuous and categorical variables
# =============================
# Plot boxplots of  continuous and categorical variables with correlation ratio > 0.3
for col in corr_ratio_df.columns:
    for row in corr_ratio_df.index:
        corr_value = corr_ratio_df.loc[[row], [col]].iat[0,0].round(2)
        if corr_value > 0.3:
            sns.boxplot(data=dataset, x=row, y=col)
            plt.title(format_col(col) + ' vs. ' + format_col(row) + ' (Corr Ratio=' + str(corr_value) + ')')
            #save_filename = 'relationship_' + col + '_' + row
            #save_image(output_dir, save_filename)  
            plt.show()


# =============================
# Combine correlation graphs into one figure
# =============================

# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=3, figsize=(16, 6))

# Correlation between continuous variables
axis=ax_array_flat[0]
sns.heatmap(dataset[numerical_cols].corr(), annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1, cbar=False, ax=axis)
axis.set_title('Correlation Between Continuous Variables')

# Association between categorical variables
axis=ax_array_flat[1]
sns.heatmap(cramers_df, annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1, cbar=False, ax=axis)
axis.set_title("Association Between Categorical Variables (Cramér's V)")

# Correlation between continuous and categorical variables
axis=ax_array_flat[2]
sns.heatmap(corr_ratio_df, annot=True, linewidth=.8, cmap="Blues", vmin=0, vmax=1, ax=axis)
axis.set_title("Correlation Ratio Between Numerical and Categorical Variables")

# Finalize figure formatting and export
fig.suptitle('Feature Correlation', fontsize=24, y=1.08) # y=1.08 increases space below figure title
#fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
save_filename = 'combined_corr'
save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()


# Include three plots of variables with correlation > 0.5
# Create figure, gridspec, list of axes/subplots mapped to gridspec location
fig, gs, ax_array_flat = initialize_fig_gs_ax(num_rows=1, num_cols=3, figsize=(16, 6))

# Ever_married vs. age
axis=ax_array_flat[0]
sns.boxplot(data=dataset, x='ever_married', y='age', ax=axis)
axis.set_ylabel('Age')
axis.set_xlabel('Ever Married')
#axis.xaxis.get_label().set_fontsize(12)
axis.set_title("Ever Married vs. Age (Corr ratio=0.68)")

# Work_type vs. age
axis=ax_array_flat[1]
sns.boxplot(data=dataset, x='work_type', y='age', ax=axis)
axis.set_ylabel('Age')
axis.set_xlabel('Work Type')
#axis.xaxis.get_label().set_fontsize(12)
axis.set_title("Work Type vs. Age (Corr ratio=0.68)")

# Work_type vs. ever_married
axis=ax_array_flat[2]
sns.countplot(data=dataset, x='ever_married', hue='work_type', ax=axis)#, legend=False)
#plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0, title='Work Type')#, loc='upper left')
axis.legend(title='Work Type')
axis.set_ylabel('Count')
axis.set_xlabel('Ever Married')
axis.set_title("Ever Married vs. Work Type (Cramer's=0.57)")

# Finalize figure formatting and export
fig.suptitle('Feature Correlation Details', fontsize=24)
fig.tight_layout(h_pad=2) # Increase spacing between plots to minimize text overlap
save_filename = 'combined_corr_details'
save_image(output_dir, save_filename, bbox_inches='tight')
plt.show()


# ==========================================================
# Cumulative Risk of stroke by age
# ==========================================================
stroke_rates = []

dataset['age'].min()
# Found that min age is 0.08. For the sake of the loop counter needing be an int, will say min_age = 1 
# as it calculates the risk of stroke at any age below the current age (so it won't ignore the age=0.08)
min_age = 1

dataset['age'].max()
# Found that max age in the dataset is 82.0, will call it 82 (an int)
max_age = 82

# Looping through each age to calculate risk of having stroke by the time someone reaches that age
for i in range(min_age, max_age):
    # Current age calculating risk for
    age = i
    
    # In this dataset, number of strokes in anyone current age or younger
    num_strokes = dataset[dataset['age'] <= i]['stroke'].sum()
    
    # Total number of people in this dataset current age or younger
    num_people = len(dataset[dataset['age'] <= i])
    
    # Add the stroke rate to the list
    stroke_rates.append(num_strokes / num_people)

# Create line plot, the x-axis is technically the index of the value, but this is actually the age given the way the loop works
sns.lineplot(data=stroke_rates)
plt.xlabel('Age')
plt.ylabel('Cumulative Stroke Risk')
plt.title('Cumulative Stroke Risk vs. Age')
#save_filename = 'cumulative_stroke_risk_vs_age'
#save_image(output_dir, save_filename)  
plt.show()

















