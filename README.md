# Medical Cost Prediction (in progress)

In this analysis, I explore the Kaggle [Medical Cost Dataset](https://www.kaggle.com/mirichoi0218/insurance). I'll go through the major steps in Machine Learning to build and evaluate regression models to predict total cost of medical care based on demographic data.

## EDA
<p align="center"><img src="/output/eda/feature_summary.png" width="600"/></p>

<p align="center"><img src="/output/eda/hist_charges.png" width="400"/></p>

<p align="center"><img src="/output/eda/combined_cat_counts.png" width="900"/></p>

<p align="center"><img src="/output/eda/cat_variables_vs_target.png" width="900"/></p>
<br><br>
Noticed bimodal distribution of charges in smokers. Attempted to subgroup by all other variables, found that BMI explained it well:
<br><br>
<p align="center"><img src="/output/eda/smoker_dist_by_bmi.png" width="700"/></p>

<p align="center"><img src="/output/eda/num_var_combined.png" width="900"/></p>

### Further exploration of Age vs. Charges relationship
'Age vs. Charges' plot looks like three distinct groups. I tried subgrouping by all the categorical variables and found that smoking status explained the groups quite well. In the graph below, you can barely see the orange 'nonsmoker' line as it fits the data points so well.

<p align="center"><img src="/output/eda/age_vs_charges_grp_smoking_status.png" width="500"/></p>

After isolating the nonsmoker data, subgrouping by categorical variables didn't account for the noise above the dense line of data points, but even without subgrouping I got a Pearson's r of 0.63, which is double the Pearson's without subgrouping by smoking status (0.30).

<p align="center"><img src="/output/eda/age_vs_charges_nonsmokers.png" width="500"/></p>

Looking at graph below plotting only data from smokers, there still seems to be two distinct groups. 

<p align="center"><img src="/output/eda/age_vs_charges_smokers.png" width="500"/></p>

Again, I tried subgrouping by all categorical variables and found that BMI separated the groups very well. The Pearson's r is included for each fit line.

<p align="center"><img src="/output/eda/age_vs_charges_smokers_grp_bmi30.png" width="500"/></p>

For the sake of further exploration, I tried adjusting the BMI cutoff to 29 and 31 to see if the data splits better. The average Pearson's r for both cutoffs was 0.59 compared to 0.68 for BMI cutoff of 30.

<p align="center">
  <img src="/output/eda/age_vs_charges_smokers_grp_bmi29.png" width="400"/>
  <img src="/output/eda/age_vs_charges_smokers_grp_bmi31.png" width="400"/>
</p>

I tried subgrouping the nonsmokers by other variables, but none of them uncovered patterns in the data.


### Further exploration of BMI vs. Charges relationship
Subgrouping by smoking status effectively delineated the two groups of data points. And again, you can see that within the smoking group there are two clear clusters on either side of BMI = 30.
<p align="center"><img src="/output/eda/bmi_vs_charges_grp_smoking.png" width="500"/></p>

### Further exploration of Children vs. Charges relationship
No new insights were gained by subgrouping this relationship.

### Relationship Between Numerical Variables
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


### Relationship Between Categorical Variables
Including ordinal variable 'children' in this analysis. After researching how to measure association between ordinal and categorical variables, I found that it is not straightforward. There are multiple ways, including 'Freeman's Theta' which I was unable to find the formula for. So I will treat 'children' as a categorical variable for this part of the analysis. 
