# Medical Cost Prediction (in progress)

In this analysis, I explore the Kaggle [Medical Cost Dataset](https://www.kaggle.com/mirichoi0218/insurance). I'll go through the major steps in Machine Learning to build and evaluate regression models to predict total cost of medical care based on demographic data.

Theoretically, a model like this could be used by insurance companies to predict the total medical cost of an individual, which they could base their premiums on. According to the Kaggle poster, it comes from the book 'Machine Learning with R' by Brett Lantz and is in the public domain. I could not find more information on the origin of the dataset, but based on my EDA and its behavior in a linear model, it is almost certainly artificial data. Nevertheless, the process I go through is still valid and can be applied to real-world data. 

## Programming Language and Resource Details
**Python Version:** 3.8.12

**Packages:** pandas, numpy, sklearn, statsmodels, scipy, matplotlib, seaborn

**Resources:** Reference links embedded in appropriate sections

# EDA
Full code: 
- [cost_eda.py](/cost_eda.py) <br> All figures: [medical_cost_prediction/output/eda](/output/eda)
- Uses module I wrote and uploaded: [ds_helper.py](https://github.com/ajspurr/my_ds_modules/blob/main/ds_helper.py)

<p align="center"><img src="/output/eda/data_overview.png" width="600"/></p>
<p align="center"><img src="/output/eda/feature_summary.png" width="900"/></p>

Most of the features are self-explanatory, and the categories of each categorical variable can be seen below. But I will clarify a few here (per the Kaggle poster):
- children: number of children covered by health insurance / number of dependents
- charges: individual medical costs billed by health insurance

### Explore Target (charges)
The distribution is positively skewed and bimodal, although the second mode isn't huge. 

<p align="center"><img src="/output/eda/hist_charges.png" width="400"/></p>

I compared this distribution to 99 continuous distributions in the SciPy package using statistical goodness of fit (GOF) tests: Kolmogorov-Smirnov, Cramer-von Mises, and Anderson-Darling (when applicable). With a Kolmogorov-Smirnov, the null hypothesis (the data follow a specified distribution) is rejected if the test statistic is greater than the critical value obtained from a K-S table. According to [this table](https://oak.ucc.nau.edu/rh83/Statistics/ks1/) the critical value for n>40 is 1.36/sqrt(n), which in this case is 0.0372. Even though this distribution is bimodal, it did pass the Kolmogorov-Smirnov for an inverted gamma distribution, which produced a critical value of 0.0370. 

<p align="center"><img src="/output/eda/test_dist/ks_sorted_qqhist0_invgamma.png" width="800"/></p>

This still doesn't look like a great fit. In fact, distributions like Johnson SB and gamma looked like a better fit on their [Q-Q plots](/output/eda/test_dist). Based on the EDA below, I found that I could separate out the different modes using the features 'smoker' and newly-created 'BMI >= 30'. Those distributions look closer to normal/unimodal and have less skew. 

### Explore Categorical Variables
Summary of categorical variables. 'BMI >= 30' was added retroactively after finding its importance in the original EDA (in relation to smoking status).
<p align="center"><img src="/output/eda/combined_cat_counts.png" width="900"/></p>

<p align="center"><img src="/output/eda/cat_variables_vs_target.png" width="900"/></p>

Violin plots to visualize relationship of all categorical variables to dichotomous categorical variables.
<p align="center"><img src="/output/eda/violin_cat_var.png" width="900"/></p>

<br><br>
Origin story of feature 'BMI >= 30'. I had noticed a bimodal distribution of charges in smokers. So I attempted to subgroup by other categorical variables to no avail. After noticing that there was a clear clustering of datapoints around BMI=30 in scatterplot (further down), I found that BMI explained the bimodal distribution  very well. I will further explore the relationships between the numerical variables and target variable in the 'Assumptions of Multiple Linear Regression' section below. 
<br><br>
<p align="center"><img src="/output/eda/smoker_dist_by_bmi.png" width="700"/></p>

<p align="center"><img src="/output/eda/num_var_combined.png" width="900"/></p>

## Relationship Between Numerical Variables
I created multiple graphs like the one below, each subgrouping by a different categorical variable. No obvious relationships were seen between numerical variables, with or without subgrouping (other than those noted above).
<p align="center"><img src="/output/eda/relationship_num_var_by_sex.png" width="600"/></p>

Heatmaps of both Pearson's and Spearman's correlation coefficients displayed below. Notably, using Spearman increased correlation between children and charges (0.07 to 0.13) and between age and charges (0.3 to 0.53). It decreased correlation between BMI and charges (0.2 to 1.2). 

Spearman's correlation is rank-order, therefore the relationship between the variables doesn't need to be linear, but needs to be monotonic (both variables increasing or both decreasing, but not necessarily consistently, or at the same rate as each other). This makes it better than Pearson's for ordinal data. It is also better for data with non-Gaussian distributions and data with outliers. 

Thus, the increase in correlation between children and charges is likely due to the fact that the variable 'children' consists of ordinal data. The increase in correlation between age and charges makes sense if you consider the points above the solid, dense line at the bottom of the scatter plot (shown below) to be outliers. The decrease in correlation between BMI and charges may be due to the fact that BMI looks almost perfectly normally distributed, so it's better suited for Pearson's.

(Credit to Annie Guo's [article](https://anyi-guo.medium.com/correlation-pearson-vs-spearman-c15e581c12ce) on Pearson's vs. Spearman's correlation)

<p align="center">
  <img src="/output/eda/corr_num_var_pearson.png" width="400"/>
  <img src="/output/eda/corr_num_var_spearman.png" width="400"/>
</p>

<p align="center"><img src="/output/eda/scatter_num_var_vs_charges.png" width="900"/></p>


## Relationship Between Categorical Variables
I'm including ordinal variable 'children' in this analysis. After researching how to measure association between ordinal and categorical variables, I found that it is not a straightforward task. There are multiple complicated methods, including calculating 'Freeman's Theta' which I was unable to find the formula for. So I will treat 'children' as a categorical variable for this part of the analysis. The associations were very weak, so I included a plot comparing two variables that had zero association according to Cramér's V. It does, indeed, look like there is no association. In researching the source of this medical cost data, it is likely synthetic, which would explain the almost perfect distribution of observations between different categories.

(Credit to Shaked Zychlinski for explaining categorical correlation in [his article](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9))

<p align="center">
  <img src="/output/eda/association_cat_variables.png" width="500"/>
  <img src="/output/eda/compare_sex_vs_region.png" width="400"/>
</p>

## Relationship Between Numerical and Categorical Variables
I used Correlation Ratio to measure the association betwee numerical and categorical variables (again, credit to Shaked Zychlinski). The only noteworthy correlation is between smoking status and charges, which we already discussed above. The correlation between the new variable 'BMI >= 30' and the BMI variable makes sense, and increases my confidence that Correlatio Ratio works!

<p align="center"><img src="/output/eda/corr_ratio_cat_num_variables.png" width="600"/></p>

# Model Building: Multiple Linear Regression

In [LinearRegression.md](/LinearRegression.md), I go through each assumption of Multiple Linear Regression in great detail, tracking model performance with each change I make and plotting relevant relationships within the data. It is a more of a classical statistics approach than a conventional machine learning approach. Below I summarize how I tested for each assumption and the process I used if the assumption was not true.

### Assumption #1: No Multicollinearity Between Predictor Variables
- I used Variance Inflation Factor (VIF) and found no evidence of multicollinearity.
### Assumption #2: Linear Relationship Between Predictors and Target Variable
- I went through each numerical variable, plotted its relationship with the target, subgrouped by multiple categories, etc., all to find linear relationships, transform non-linear relationships, and take into account parts of the  data where no relationship is present. 
- Through this process, I created new features: [bmi>=30], [bmi\*smoker], [smoker\*obese], and [age^2], and removed their original features: bmi and age. I ended up removing [bmi>=30] as well, since it was used to create [smoker\*obese].

**Original model performance:**
<p align="center"><img src="/output/models/sm_lr_results_0_original.png" width="600"/></p>

The Scale-Location plot (residuals vs. predicted target values) on the left is to visualize heteroscedasticity. 'BP' and 'White' represent tests for heteroscedasticity. Values < 0.05 indicate presence of heteroscedasticity. The plot on the right is to visualize model performance. Data points near the diagonal line represent perfect predictions. 

**Model performance after adding the final new feature (age^2):**
<p align="center"><img src="/output/models/sm_lr_results_4_age_sq_feature.png" width="700"/></p>

**Summary of model performance after adding each new feature:**
<p align="center"><img src="/output/models/performance_new_feat.png" width="600"/></p>

### Assumption #3: Multivariate normality (residuals of the model are normally distributed)
- While I understand this is not as important in machine learning prediction models as it is in statistical inference models, I worked through this process for the sake of learning. 
- I performed multiple normality tests on the residuals of my models throughout the process: Shapiro-Wilk, D'Agostino's K-squared, Chi-Square, Jarque–Bera, Kolmogorov-Smirnov, Lilliefors, and Anderson-Darling. 
- The residuals were not normal before or after I added new features. 
  - I attempted to fix this by transforming the non-normal target using Box-Cox, log transform, square-root transform, and cube-root transform. They all caused the residual distribution to appear more normal in Q-Q plots, but none affected the normality tests, and all of them significantly worsened model performance. 
  - My second attempt to fix non-normal residuals was outlier removal. I used Cook's distance to identify outliers and after extensive exploration found no pattern to the outliers nor any relationship to any of the numerical or categorical variables, other than they were related to the visual outliers in the Age vs. Charges plots. 
  - While I don't think this process or outcome is realistic, as this data is artificial, after removing outliers (8.9% of the data) I achieved a perfect model with an adjusted R-squared of 1. 

**Residual distribution in model with all new features added:**
<p align="center"><img src="/output/models/qqhist1_orig.png" width="600"/></p>

**One of the visualizations of Cook's outliers in the relationship between Age and Charges (this plot is in smokers only):**
<p align="center"><img src="/output/models/outliers_age_v_charges_nonob_smoker.png" width="350"/></p>

**Residual distribution after all of Cook's outliers were removed:**
<p align="center"><img src="/output/models/qqhist4_outlier_2.png" width="700"/></p>

**Model performance after all Cook's outliers removed:**
<p align="center"><img src="/output/models/sm_lr_results_6_no_outliers_2.png" width="700"/></p>

### Assumption #4: Homoscedasticity
  - I used the Breusch-Pagan Test and White Test for Heteroscedasticity throughout the process, visualizing it in Scale-Location plots (predicted values vs. studentized residuals)
  - Heteroscedasticity was present in the initial model, but homoscedasticity was achieved after the second new feature was created. 
### Assumption #5: Observations are independent, i.e. no autocorrelation 
- Not relevant as this is not time series data.

