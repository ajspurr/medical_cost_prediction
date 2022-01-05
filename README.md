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
<br><br>
'Age vs. Charges' plot looks like three distinct groups. I tried subgrouping by all the categorical variables and found that smoking status explained the groups quite well. In the graph below, you can barely see the orange 'nonsmoker' line as it fits the data points so well.

<p align="center"><img src="/output/eda/age_vs_charges_grp_smoking_status.png" width="500"/></p>

Looking at graph below plotting only data from smokers, there still seems to be two distinct groups. 

<p align="center"><img src="/output/eda/age_vs_charges_smokers.png" width="500"/></p>

Again, I tried subgrouping by all categorical variables and found that BMI separated the groups very well. The Pearson's r is included for each fit line.

<p align="center"><img src="/output/eda/age_vs_charges_nonsmokers_grp_bmi30.png" width="500"/></p>

For the sake of further exploration, I tried adjusting the BMI cutoff to 29 and 31 to see if the data splits better. The average Pearson's r for both cutoffs was 0.59 compared to 0.68 for BMI cutoff of 30.

<p align="center">
  <img src="/output/eda/age_vs_charges_nonsmokers_grp_bmi29.png" width="400"/>
  <img src="/output/eda/age_vs_charges_nonsmokers_grp_bmi31.png" width="400"/>
</p>

I tried subgrouping the nonsmokers by other variables, but none of them uncovered patterns in the data.
