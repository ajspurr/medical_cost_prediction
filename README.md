# Medical Cost Prediction (in progress)

In this analysis, I explore the Kaggle [Medical Cost Dataset](https://www.kaggle.com/mirichoi0218/insurance). I'll go through the major steps in Machine Learning to build and evaluate regression models to predict total cost of medical care based on demographic data.

Theoretically, a model like this could be used by insurance companies to predict the total medical cost of an individual, which they could base their premiums on. According to the Kaggle poster, it comes from the book 'Machine Learning with R' by Brett Lantz and is in the public domain. I could not find more information on the origin of the dataset, but based on my EDA and its behavior in a linear model, it is almost certainly artificial data. Nevertheless, the process I go through is still valid and can be applied to real-world data. 


# EDA

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

In [LinearRegression.md](/LinearRegression.md), I go through each assumption of Multiple Linear Regression in great detail, tracking model performance with each change I make and plotting relevant relationships within the data. Below I summarize how I tested each assumption and the process I used if the assumption was not true.

### Assumption #1: No Multicollinearity Between Predictor Variables
- I used Variance Inflation Factor (VIF) and found no evidence of multicollinearity.
### Assumption #2: Linear Relationship Between Predictors and Target Variable
- I went through each numerical variable, plotted its relationship with the target, subgrouped by multiple categories, etc., all to find linear relationships, transform non-linear relationships, and take into account parts of the  data where no relationship is present. 
- Through this process, I created new features: [bmi>=30], [bmi\*smoker], [smoker\*obese], and [age^2], and removed their original features: bmi and age. I ended up removing [bmi>=30] as well, since it was used to create [smoker\*obese].

Original model performance:
<p align="center"><img src="/output/models/sm_lr_results_0_original.png" width="600"/></p>

Scale-Location plot (residuals vs. predicted target values) on the left is to visualize heteroscedasticity. 'BP' and 'White' represent tests for heteroscedasticity. Values < 0.05 indicate presence of heteroscedasticity. The plot on the right is to visualize model performance. Data points near the diagonal line represent perfect predictions. 

Model performance after adding the final new feature: 
<p align="center"><img src="/output/models/sm_lr_results_4_age_sq_feature.png" width="700"/></p>

Summary of model performance after adding each new feature:
<p align="center"><img src="/output/models/performance_new_feat.png" width="600"/></p>

### Assumption #3: Multivariate normality (residuals of the model are normally distributed)
- While I understand this is not as important in machine learning prediction models as it is in statistical inference models, I worked through this process for the sake of learning. 
- I performed multiple normality tests on the residuals of my models throughout the process: Shapiro-Wilk, D'Agostino's K-squared, Chi-Square, Jarque–Bera, Kolmogorov-Smirnov, Lilliefors, and Anderson-Darling. 
- The residuals were not normal before or after I added new features. 
  - I attempted to fix this by transforming the non-normal target using Box-Cox, log transform, square-root transform, and cube-root transform. They all caused the residual distribution to appear more normal in Q-Q plots, but none affected the normality tests, and all of them significantly worsened model performance. 
  - My second attempt to fix non-normal residuals was outlier removal. I used Cook's distance to identify outliers and after extensive exploration found no pattern to the outliers nor any relationship to any of the numerical or categorical variables, other than they were related to the visual outliers in the Age vs. Charges plots. 
  - While I don't think this process or outcome is realistic, as this data is artificial, after removing outliers (8.9% of the data) I achieved a perfect model with an adjusted R-squared of 1. 

Residual distribution in model with all new features added:
<p align="center"><img src="/output/models/qqhist1_orig.png" width="600"/></p>

Visualization of Cook's outliers in the relationship between Age and Charges:
<p align="center"><img src="/output/models/outliers_age_v_charges_nonob_smoker.png" width="350"/></p>

Residual distribution after all Cook's outliers removed:
<p align="center"><img src="/output/models/qqhist4_outlier_2.png" width="700"/></p>

Model performance after all Cook's outliers removed:
<p align="center"><img src="/output/models/sm_lr_results_6_no_outliers_2.png" width="700"/></p>

### Assumption #4: Homoscedasticity
  - I used the Breusch-Pagan Test and White Test for Heteroscedasticity throughout the process, visualizing it in Scale-Location plots (predicted values vs. studentized residuals)
  - Heteroscedasticity was present in the initial model, but homoscedasticity was achieved after the second new feature was created. 
### Observations are independent, i.e. no autocorrelation 
- Not relevant as this is not time series data.


To test for these assumptions, I used package 'statsmodels' (as opposed to 'sklearn') to build a multiple linear regression model, as it offers a more robust set of statistical tests. I'd like to visualize the progression of the accuracy of the model as I correct any deviation from the above assumptions. So, I'll start by fitting all the data (no test/train split) to a multiple linear regression model: 

<p align="center"><img src="/output/models/sm_lr_results_0_original.png" width="700"/></p>

<p align="center"><img src="/output/models/sm_lr_results_1_bmi_30_feature.png" width="700"/></p>

There are clear groupings of predicted values, which (surprise, surprise) relate to BMI and smoking status. Subgrouped plot below. In the plot on the left, 'BP' represents the p-value for the Breusch-Pagan Lagrange Multiplier Test for Heteroscedasticity and 'White' represents the p-value of White’s Lagrange Multiplier Test for Heteroscedasticity. These will be discussed in the 'Homoscedasticity' section below. Values less than 0.05 indicate presence of heteroscedasticity.

<p align="center"><img src="/output/models/sm_lr_results_1_bmi_30_feature_grouped.png" width="900"/></p>

## Assumption #1: No Multicollinearity Between Predictor Variables
VIF table below shows that multicollinearity between numerical variables is not present. A value of 1 indicates that there is no correlation with any other predictor variables. A value between 1 and 5 indicates mild correlation, generally not enough to require attention. A value between 5 and 10 indicates moderate correlation. A value of 10 or greather indicates severe correlation (a.k.a. multicollinearity), in which case the coefficient estimates and p-values in the regression output are likely unreliable. Even if none of the variable pairs are highly correlated (as has already been shown in the 'Relationship Between Numerical Variables' section above), multicollinearity can still be present as a given variable can be explained by two or more other variables.
(References: [1](https://www.statology.org/how-to-calculate-vif-in-python/), [2](https://quantifyinghealth.com/correlation-collinearity-multicollinearity/))

<p align="center"><img src="/output/models/vif_table.png" width="300"/></p>

## Assumption #2: Linear Relationship Between Predictors and Target Variable
### BMI vs. Charges
The linear relationship between BMI and charges is weak. But if you subgroup by smoking status, you can see that smokers' BMI have a strong linear relationship with charges (Pearson's of 0.8) while nonsmokers' BMI have basically no linear relationship with charges. As such, I will enginner a new feature: **[smoker\*bmi]**. This will remove the bmi of the nonsmokers, thus removing the data that does not have a linear relationship to the target. 
<p align="center">
  <img src="/output/eda/lmplot_bmi_vs_charges.png" width="400"/>
  <img src="/output/eda/bmi_vs_charges_grp_smoking.png" width="400"/>
</p>

As can be seen below, adding the new feature greatly reduced heteroscedasticity and improved R-squared.

<p align="center"><img src="/output/models/sm_lr_results_2_smoke_bmi_feature.png" width="900"/></p>

### Age vs. Charges
'Age vs. Charges' plot looks like three distinct groups. I tried subgrouping by all the categorical variables and found that smoking status explained the groups quite well (plot on left). After isolating the nonsmoker data (middle plot), subgrouping by any categorical variable didn't account for the noise above the dense line of data points, but even without subgrouping I got a Pearson's r of 0.63, which is double the Pearson's without subgrouping by smoking status (0.30). Looking at only data from smokers (plot on right), I tried subgrouping by all categorical variables and found that BMI separated the groups very well. The Pearson's r is included for each fit line.

<p align="center">
  <img src="/output/eda/age_vs_charges_grp_smoking_status.png" width="310"/>
  <img src="/output/eda/age_vs_charges_nonsmokers.png" width="310"/>
  <img src="/output/eda/age_vs_charges_smokers_grp_bmi30.png" width="310"/>
</p>

For the sake of further exploration, I tried adjusting the BMI cutoff to 29 and 31 to see if the data splits better. The average Pearson's r for both cutoffs was 0.59 compared to 0.68 for BMI cutoff of 30 (images in /output/eda).

#### New Feature: [smoker\*obese]

In order to incorporate this relationship between obesity, smoking status, and age into the model, I tried creating multiple features. The one that worked fantastically was **[smoker\*obese]**. I originally tried **[smoker\*obese\*age]**, assuming that you needed the 'age' variable to actually make the prediction. However, if you look at the age vs. charges plot above you'll see that the 3 lines have very shallow slopes. So 'age' itself isn't very predicitive but the difference between the three groups (nonsmokers, obese smokers, and nonobese smokers) is very predictive. With this new variable, which isolates obese smokers, the model can give it a coefficient that basically adds a constant value to that group which is equal to the average difference in charges between the 'obese smokers' and 'nonobese smokers' lines in the age vs. charges plot. 

I also tried to add a variable incorporating the nonobese smokers, as it has its own line as well. This didn't change anything because the model already adds ~15,000 to the charges if you're a smoker (remember, there is a 'smoker' variable that has its own constant) then with this new **[smoker\*obese]** feature, it adds another ~20,000 for obese smokers.

<p align="center"><img src="/output/models/sm_lr_results_3_smoke_ob_feature.png" width="900"/></p>

As you can see, this greatly reduced the residuals of the predictions, although the outliers remain. 

#### New Feature: [age^2]

There is a clear curvilinear relationship between predicted charges and residuals. You can see this relationship a bit in the age vs. charges plots as well. So I added a new feature **[age^2]**. Visually, the regression line in the age^2 vs. charges plots seems to fit better, however, this new feature doesn't seem to affect the R-squared values. This may be due to the outliers and/or the shallowness of the slopes of the reg lines. 

<p align="center">
  <img src="/output/eda/age_sq_vs_charges_nonsmokers.png" width="340"/>
  <img src="/output/eda/age_sq_vs_charges_smokers_grp_bmi30.png" width="340"/>
</p>

It did slightly improve the R-squared of the model. The heteroscedasticity metrics didn't change much. However, you can visually appreciate the better fit on the model below. It seems that the outliers are the main reason the line isn't an almost perfect fit. 

<p align="center"><img src="/output/models/sm_lr_results_4_age_sq_feature.png" width="900"/></p>

### Children vs. Charges
No new insights were gained by subgrouping this relationship.

### Changes in Multiple Regression Feature Coefficients with Each New Feature

<p align="center"><img src="/output/models/coeff_new_feat_vert_3.png" width="500"/></p>

Several of the features did not have much fluctuation in their coefficients so I removed them from the plots. I left two in (and separated the features into 3 graphs) in order to appreciate the scale of the change of the other feature coefficients. When [bmi >= 30] feature was added, the [bmi] feature's coefficient decreased significantly. When the [bmi\*smoker] feature was added, [bmi]'s coefficient continued to decrease. When the [smoker\*obese] feature as added, the [smoker_yes] feature decreased dramatically (note the scales). In addition, the new features [bmi >= 30] and [bmi\*smoker] decreased as well, with [bmi >= 30]'s coefficient reaching close to 0! This means [smoker\*obese] was a much better predicitor of charges. Lastly, when [age^2] was added, the [age] feature coefficient decreased to about 0.

If I hadn't standardized the numerical varaibles, I would say that you can't necessarily compare the scale of one variable's coefficient to another variable's coefficient as the scale/units of the variable itself may be vastly different. As an example, let's say our dataset included serum sodium levels, which are usually between 135-145. Let's say we create a multiple regression model and it assigns a coefficient of 1 to serum sodium and 2 to bmi (range 25-35 for this example). In this case, serum sodium will generally add $135-145 to the predicted medical cost and bmi will only add $50-70 to the predicted medical cost, even with a coefficient twice as large. However, I did standardize the numerical varaibles before creating the model, so the model coefficients are actually standardized coefficients, which signify the mean change of the dependent variable given a one standard deviation shift in the independent variable. As such, the model coefficients can be compared to each other ([ref](https://statisticsbyjim.com/regression/identifying-important-independent-variables/)).

### Summary of Model Performance with Each New Feature

<p align="center"><img src="/output/models/performance_new_feat.png" width="800"/></p>

RMSE penalizes large errors the most. MAE does not penalize large errors as much. Median absolute error penalizes large errors the least. R-squared represents the percent of the variation of the target that is explained by it's relationship with the features. R-squared is a relative measure whereas RMSE and MAE are absolute measures. One drawback of R-squared is that by the nature of its calculation, it improves every time you add a new variable to the model. Adjusted R-squared corrects for this. ([Reference](https://towardsdatascience.com/evaluation-metrics-model-selection-in-linear-regression-73c7573208be))

I did not include the F-statistic as its p-value was 0.00 for every model. The null hypothesis of the F-test is that a model with no independent variables (intercept-only) fits the data as well as my model. The alternative hypothesis states that my model fits the data better than an intercept-only model. If the F-test p-value is less than the significance level (let's say 0.05), then the sample data provide sufficient evidence to conclude that my regression model fits the data better than the intercept-only model. ([Reference](https://statisticsbyjim.com/regression/interpret-f-test-overall-significance-regression/))

### A quick note on variable selection in multiple linear regression
I did some research on how to determine which variables are most important in multiple linear regression. In the world of statistics, this turns out to be very complicated ([ref](https://onlinelibrary.wiley.com/doi/full/10.1002/bimj.201700067)). It looks like standardized coefficients and 'change in R-squared for last variable added to the model' are potential sources of helpful information, but individual coefficient p-values are not ([ref](https://statisticsbyjim.com/regression/identifying-important-independent-variables/)). As I plan to focus more on machine learning models for prediction, rather than statistical models for inference, I won't explore this any further for now. However, I will remove variables which have standardized coefficients close to zero AND were used to create one of the new features I created: bmi, age, bmi>=30. By definition, they are correlated to the new features [bmi\*smoker], [smoker\*obese], and [age^2] and should be removed for that reason anyway. The graph below shows an updated version of the coefficients with these variables removed. I included all remaining variables this time as it wasn't as cluttered. You can see that were wasn't much change. I didn't replot the model performance metrics as they were essentially unchanged.

<p align="center"><img src="/output/models/coeff_vert_6_rem_old_var.png" width="500"/></p>

## Assumption #3: Multivariate normality (residuals of the model are normally distributed)
It seems this assumption is generally less relevant for machine learning than it is for classical statistics. The p-values of your model coefficients depend on this assumption. So if you are using your model to make inferences about the data, this assumption needs to be explored. However, if you are primarily concerned with your predictions (as is normally true for machine learning), this assumption is not important. Furthermore, according to [this source](https://www.decisiondata.blog/understanding-linear-regression-6db487377bac), even if you are trying to make inferences, if your dataset is "large enough and is not too far from normality then, by the Central Limit Theorem, our assumption of normality is not that important, and any inference from the model will still be valid." For the sake of learning, I will explore this assumption. 

As shown below, the residuals in my model do not look normally distributed. The normality tests that were performed (in the order they are displayed) are: Shapiro-Wilk, D'Agostino's K-squared, Chi-Square, Jarque–Bera, Kolmogorov-Smirnov, Lilliefors, and Anderson-Darling. Surprisingly, it passed the Chi-Square Test for normality. While pass/fail isn't technically the correct verbage, it represents whether or not the p-value was greater than a significance level of 0.05. Or in the case of Anderson-Darling, whether or not the test statistic was greater than the critical value associatied with a significance level of 0.05. 


<p align="center"><img src="/output/models/qqhist1_orig.png" width="800"/></p>

### What to do with non-normal residuals ([1](https://www.statology.org/multiple-linear-regression-assumptions/), [2](https://towardsdatascience.com/is-normal-distribution-necessary-in-regression-how-to-track-and-fix-it-494105bc50dd))
Two potential causes include:
- Dependent or independent variables are too non-normal 
- Existence of a few outliers/extreme values which disrupt the model prediction

So what you can do is:
- Perform transformations on dependent or independent variables
- Explore and remove outliers

#### Box-Cox Transformation of dependent variable 'charges'
'Charges' certainly looks more normally distributed after performing a Box-Cox transformation.

<p align="center">
  <img src="/output/eda/hist_charges.png" width="400"/>
  <img src="/output/models/charges_boxcox.png" width="425"/>
</p>

Now the residuals look closer to normal distribution as well, but they still only pass the Chi-Square Test for normality. 

<p align="center"><img src="/output/models/qqhist2_boxcox_y.png" width="800"/></p>

However, this skewed the model predictions and significantly decreased model performance. 

<p align="center"><img src="/output/models/sm_lr_results_7_bc_charges.png" width="800"/></p>

If you look at the last Standardized Residuals vs. Predicted Values plot, in the [section where I added the age^2 feature](#new-feature-age2), you can see the vast majority of predictions lie just under the 0-residual line with several outliers high above the line, clearly skewing the results. Once again, this is not real-world data and that pattern was likely created on purpose. I don't think transforming the dependent or indepdent variables will improve the model, although I performed three transformations to the dependent variable for fun, which did not help: log, sqrt, and cube root. Investigating the outliers will likely yield better results. 

#### Outlier Detection
Cooks distance using statsmodels. 

<p align="center"><img src="/output/models/cooks_dist_plot.png" width="600"/></p>

Visualization of Cook's outliers on Studentized Residual vs. Predicted Value plot. Nothing surprising.

<p align="center"><img src="/output/models/outliers_pred_vs_resid.png" width="600"/></p>

Do the model outliers represent outliers in the 'charges' distribution? If that were the case, the histograms below would show the outliers with their own unimodal distribution isolated in the higher end of the 'charges' range. Instead, the outliers look like the same bimodal distribution of the nonoutliers. Yes, it is shifted to the right, but there is still plenty of overlap.

<p align="center"><img src="/output/models/outliers_dist_charges.png" width="600"/></p>

Can the model outliers be explained by outliers in the relationship between 'charges' and any of the continuous variables? If you look at the age vs. charges plot, you may intitially think the answer is 'no'. But if you subgroup it by obesity and smoker status as I did above, you can see that each scatter plot has a more dense area with a linear relationship, and sparse data points above the line that represent the model outliers. 

<p align="center"><img src="/output/models/outliers_age_v_charges.png" width="600"/></p>

<p align="center">
  <img src="/output/models/outliers_age_v_charges_nonsmoker.png" width="290"/>
  <img src="/output/models/outliers_age_v_charges_ob_smoker.png" width="290"/>
  <img src="/output/models/outliers_age_v_charges_nonob_smoker.png" width="350"/> 
</p>

No subcategory has a significantly large percentage of outliers. Subcategory '4 children' has the highest percentage, 16%, compared to percentage of outliers in the entire dataset, which is about 6.3%. But This is such a tiny subcategory that it only accounts for 4 outliers out of 85. 

<p align="center"><img src="/output/models/perc_outlier_subcat.png" width="900"/></p>

Outlier data subcategory composition not very different than rest of data. 

<p align="center"><img src="/output/models/perc_subcat_by_outlier.png" width="900"/></p>

So far there is nothing I found that could categorize or explain the outliers other than the fact that they map well to the Charges vs. Age plot. If this were real-world data, at this point I would investigate how samples were selected and how the data was collected and input into the dataset. This would allow me to identify any systematic biases in sample selection or potential for data collection or data-entry mistakes. But with this artificial data, I will just remove the outliers so I can compare the model performance before and after. As you can seem below, this did not improve the distribution of the residuals much (in fact, this time it failed the Chi-Square Test for normality), but it grealy improved model performance. 

<p align="center"><img src="/output/models/qqhist3_outlier_1.png" width="800"/></p>

<p align="center"><img src="/output/models/sm_lr_results_5_no_outliers.png" width="900"/></p>

I plotted the changes in model performance metrics with this model. Again, the results aren't surprising. The max error (max_e) decreases dramatically. R-squared/adjusted increases dramatically. Still no calculated heteroscedasticity. Interestingly, not only did the the rmse, mae, and median asbolute errors decrease, but the difference between them decreased as well. This is a testament to the fact that they have varying sensitivities to large residuals, which have now been removed. I didn't replot the changes in model coefficients as they didn't change much. Most notably, the 'age' coefficient increased from ~-250 to ~0. 

<p align="center"><img src="/output/models/performance_no_outliers.png" width="800"/></p>

Since there are still a few outliers, I decided to perform Cook's test again. Since it measures the degree to which your predicted values change when a given datapoint is removed, it's possible that calculating it again will identify more datapoints as outliers, which is what happened. 

<p align="center">
  <img src="/output/models/cooks_dist_plot_2.png" width="350"/>
  <img src="/output/models/outliers_pred_vs_resid_2.png" width="350"/>
</p>

New model results below. The residuals look much more normal although they did not pass any tests for normality (interestingly, when I originally kept in the old features, this step did pass three tests for normality: D'Agostino's K-squared, Chi-Square, and Jarque–Bera). I also achieved a perfect model! Of course, this is after removing 119 outliers, which represent 8.9% of the data. In addition, this dataset is from a textbook so I'm sure it has been generated artificially to demonstrate the points I have been making. Real world data would not behave this perfectly, but the process I went through is still applicable.  

<p align="center"><img src="/output/models/qqhist4_outlier_2.png" width="800"/></p>

<p align="center"><img src="/output/models/sm_lr_results_6_no_outliers_2.png" width="800"/></p>

I did not include updated coefficients or model performance plots, but I can summarize here. The coefficients remained almost exactly the same. This can be explained with the influence plots below. As you can see, the datapoints with the highest Cook's distances (represented by the biggest circles) have the least leverage. And leverage "refers to the extent to which the coefficients in the regression model would change if a particular observation was removed from the dataset" ([source](https://www.statology.org/residuals-vs-leverage-plot/)). The model performance can be summarized with an R-squared of 1.0. Interestingly, as you can see above, the BP and White's Test p-values decreased significantly, which would normally indicate heteroscedasticity. I could do a deep dive into the formula behind those tests, but I would assume with a perfect model and residuals on the order of 10^-11, that the metrics aren't useful.

<p align="center">
  <img src="/output/models/influence_plot_1.png" width="350"/>
  <img src="/output/models/influence_plot_2.png" width="350"/>
</p>

## Assumption #4: Homoscedasticity
To reiterate, in the plots above, 'BP' represents the p-value for the Breusch-Pagan Lagrange Multiplier Test for Heteroscedasticity and 'White' represents the p-value of White’s Lagrange Multiplier Test for Heteroscedasticity. Values less than 0.05 indicate presence of heteroscedasticity. The Breusch-Pagan test (the default) detects linear forms of heteroscedasticity, while White's test detects non-linear forms ([source](https://www3.nd.edu/~rwilliam/stats2/l25.pdf)). If you look at the bottom-right plot in the "[Summary of Model Performance with Each New Feature](#summary-of-model-performance-with-each-new-feature)" section, you will see that once the [bmi\*smoker] feature is added, homoscedasticity is achieved. It remains after outliers are removed as well. 

# Potential Future Exploration
- Remove uneccesary features, for now I will address this in the machine learning centric part of the project
  - [Adjusted R-squared](https://www.decisiondata.blog/understanding-linear-regression-6db487377bac)
  - [Variable selection paper](https://onlinelibrary.wiley.com/doi/full/10.1002/bimj.201700067)
  - [Identifying the Most Important Independent Variables in Regression Models](https://statisticsbyjim.com/regression/identifying-important-independent-variables/)
