# Medical Cost Prediction (in progress)

In this analysis, I explore the Kaggle [Medical Cost Dataset](https://www.kaggle.com/mirichoi0218/insurance). I'll go through the major steps in Machine Learning to build and evaluate regression models to predict total cost of medical care based on demographic data.

# EDA
<p align="center"><img src="/output/eda/feature_summary.png" width="600"/></p>

<p align="center"><img src="/output/eda/hist_charges.png" width="400"/></p>

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
I'm including ordinal variable 'children' in this analysis. After researching how to measure association between ordinal and categorical variables, I found that it is not a straightforward task. There are multiple complicated methods, including calculating 'Freeman's Theta' which I was unable to find the formula for. So I will treat 'children' as a categorical variable for this part of the analysis. The associations were very weak, so I included a plot comparing two variables that had zero association according to Cramér's V. It does, indeed, look like there is no association. In researching the source of this medical cost data, it may be synthetic, which would explain the almost perfect distribution of observations between different categories.

(Credit to Shaked Zychlinski for explaining categorical correlation in [his article](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9))

<p align="center">
  <img src="/output/eda/association_cat_variables.png" width="500"/>
  <img src="/output/eda/compare_sex_vs_region.png" width="400"/>
</p>

## Relationship Between Numerical and Categorical Variables
I used Correlation Ratio to measure the association betwee numerical and categorical variables (again, credit to Shaked Zychlinski). The only noteworthy correlation is between smoking status and charges, which we already discussed above. The correlation between the new variable 'BMI >= 30' and the BMI variable makes sense, and increases my confidence that Correlatio Ratio works!

<p align="center"><img src="/output/eda/corr_ratio_cat_num_variables.png" width="600"/></p>

# Model Building
## Linear Regression
### Assumptions of Multiple Linear Regression
1. Linear relationship between each predictor variable and target variable
2. No multicollinearity between predictor variables
3. Observations are independent, i.e. no autocorrelation (not relevant as this is not time series data)
4. Homoscedasticity
5. Multivariate normality - **residuals** of the model are normally distributed

To test for these assumptions, I used package 'statsmodels' (as opposed to 'sklearn') to build a multiple linear regression model, as it offers a more robust set of statistical tests. I'd like to visualize the progression of the accuracy of the model as I correct any deviation from the above assumptions. So, I'll start by fitting all the data (no test/train split) to a multiple linear regression model: 

<p align="center"><img src="/output/models/sm_lr_results_original.png" width="700"/></p>

There are clear groupings of predicted values, which (surprise, surprise) relate to BMI and smoking status. Subgrouped plot below. In the plot on the left, 'BP' represents the p-value for the Breusch-Pagan Lagrange Multiplier Test for Heteroscedasticity and 'White' represents the p-value of White’s Lagrange Multiplier Test for Heteroscedasticity. These will be discussed in the 'Homoscedasticity' section below. Values less than 0.05 indicate presence of heteroscedasticity.

<p align="center"><img src="/output/models/sm_lr_results_orig_subgrouped.png" width="900"/></p>

### Linear Relationship Between Predictors and Target Variable
#### BMI vs. Charges
The linear relationship between BMI and charges is weak. But if you subgroup by smoking status, you can see that smokers' BMI have a strong linear relationship with charges (Pearson's of 0.8) while nonsmokers' BMI have basically no linear relationship with charges. As such, I will enginner a new feature: **[smoker\*bmi]**. This will remove the bmi of the nonsmokers, thus removing the data that does not have a linear relationship to the target. 
<p align="center">
  <img src="/output/eda/lmplot_bmi_vs_charges.png" width="400"/>
  <img src="/output/eda/bmi_vs_charges_grp_smoking.png" width="400"/>
</p>

As can be seen below, adding the new feature greatly reduced heteroscedasticity and improved R-squared.

<p align="center"><img src="/output/models/sm_lr_results_smoke_bmi_feature.png" width="900"/></p>

#### Age vs. Charges
'Age vs. Charges' plot looks like three distinct groups. I tried subgrouping by all the categorical variables and found that smoking status explained the groups quite well (plot on left). After isolating the nonsmoker data (middle plot), subgrouping by any categorical variable didn't account for the noise above the dense line of data points, but even without subgrouping I got a Pearson's r of 0.63, which is double the Pearson's without subgrouping by smoking status (0.30). Looking at only data from smokers (plot on right), I tried subgrouping by all categorical variables and found that BMI separated the groups very well. The Pearson's r is included for each fit line.

<p align="center">
  <img src="/output/eda/age_vs_charges_grp_smoking_status.png" width="310"/>
  <img src="/output/eda/age_vs_charges_nonsmokers.png" width="310"/>
  <img src="/output/eda/age_vs_charges_smokers_grp_bmi30.png" width="310"/>
</p>

For the sake of further exploration, I tried adjusting the BMI cutoff to 29 and 31 to see if the data splits better. The average Pearson's r for both cutoffs was 0.59 compared to 0.68 for BMI cutoff of 30 (images in /output/eda).

##### New Feature: [smoker\*obese]

In order to incorporate this relationship between obesity, smoking status, and age into the model, I tried creating multiple features. The one that worked fantastically was **[smoker\*obese]**. I originally tried **[smoker\*obese\*age]**, assuming that you needed the 'age' variable to actually make the prediction. However, if you look at the age vs. charges plot above you'll see that the 3 lines have very shallow slopes. So 'age' itself isn't very predicitive but the difference between the three groups (nonsmokers, obese smokers, and nonobese smokers) is very predictive. With this new variable, which isolates obese smokers, the model can give it a coefficient that basically adds a constant value to that group which is equal to the average difference in charges between the 'obese smokers' and 'nonobese smokers' lines in the age vs. charges plot. 

I also tried to add a variable incorporating the nonobese smokers, as it has its own line as well. This didn't change anything because the model already adds ~15,000 to the charges if you're a smoker (remember, there is a 'smoker' variable that has its own constant) then with this new **[smoker\*obese]** feature, it adds another ~20,000 for obese smokers.

<p align="center"><img src="/output/models/sm_lr_results_smoke_ob_feature.png" width="900"/></p>

As you can see, this greatly reduced the residuals of the predictions, although the outliers remain. 

##### New Feature: [age^2]

There is a clear curvilinear relationship between predicted charges and residuals. You can see this relationship a bit in the age vs. charges plots as well. So I added a new feature **[age^2]**. Visually, the regression line in the age^2 vs. charges plots seems to fit better, however, this new feature doesn't seem to affect the R-squared values. This may be due to the outliers and/or the shallowness of the slopes of the reg lines. 

<p align="center">
  <img src="/output/eda/age_sq_vs_charges_nonsmokers.png" width="340"/>
  <img src="/output/eda/age_sq_vs_charges_smokers_grp_bmi30.png" width="340"/>
</p>

It did slightly improve the R-squared of the model. The heteroscedasticity metrics didn't change much. However, you can visually appreciate the better fit on the model below. It seems that the outliers are the main reason the line isn't an almost perfect fit. 

<p align="center"><img src="/output/models/sm_lr_results_age_sq_feature.png" width="900"/></p>

#### Children vs. Charges
No new insights were gained by subgrouping this relationship.

### Changes in Multiple Regression Feature Coefficients with Each New Feature

<p align="center"><img src="/output/models/coeff_new_feat_vert_3.png" width="500"/></p>

Several of the features did not have much fluctuation in their coefficients. I took most of them out. I left two in (and separated the features into 3 graphs) in order to appreciate the scale of the change of the other feature coefficients. When [bmi >= 30] feature was added, the 'bmi' feature's coefficient decreased significantly. When the [bmi\*smoker] feature was added, [bmi]'s coefficient continued to decrease. When the [smoker\*obese] feature as added, the [smoker_yes] feature decreased dramatically (note the scales). In addition, the new features [bmi >= 30] and [bmi\*smoker] decreased as well, with [bmi >= 30]'s coefficient reaching close to 0! This means [smoker\*obese] was a much better predicitor of charges.. Lastly, when [age^2] was added, the [age] feature coefficient decreased to about 0.

### Summary of Model Performance with Each New Feature

<p align="center"><img src="/output/models/performance_new_feat.png" width="800"/></p>

RMSE penalizes large errors. MAE does not penalize large errors (there are several here). R-squared represents the percent of the variation of the target that is explained by it's relationship with the features. R-squared is a relative measure whereas RMSE and MAE are absolute measures. One drawback of R-squared is that by the nature of its calculation, it improves every time you add a new variable to the model. Adjusted R-squared corrects for this. ([Source](https://towardsdatascience.com/evaluation-metrics-model-selection-in-linear-regression-73c7573208be))

### Outlier Detection
Cooks distance using statsmodels. 

<p align="center"><img src="/output/models/cooks_dist_plot.png" width="600"/></p>

Cook's Distance outliers maps to the outliers in age vs. charges plots.

<p align="center">
  <img src="/output/models/outliers_age_v_charges_nonsmoker.png" width="290"/>
  <img src="/output/models/outliers_age_v_charges_ob_smoker.png" width="290"/>
  <img src="/output/models/outliers_age_v_charges_nonob_smoker.png" width="350"/> 
</p>

No subcategory has a significantly large percentage of outliers. Subcategory '4 children' has the highest percentage, 15%, compared to percentage of outliers in the entire dataset, which is about 6.7%. But This is such a tiny subcategory that it only accounts for 4 outliers out of 90. 

<p align="center"><img src="/output/models/perc_outlier_subcat.png" width="900"/></p>

Outlier data subcategory composition not very different than rest of data. 
<p align="center"><img src="/output/models/perc_subcat_by_outlier.png" width="900"/></p>


### Homoscedasticity
Breusch-Pagan test (the default) detects linear forms of heteroscedasticity. White's test detects non-linear forms. ([source](https://www3.nd.edu/~rwilliam/stats2/l25.pdf))

## Potential Future Exploration
- 
