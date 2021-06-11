# Cormac_assessment
Input Data: Dictionary of attributes (Types of procedure, How long the procedure lasted, Severity of the condition), Dictionary of variables containing success/failure for the specified attributes and dictionary of granular measure of outcomes (recurrence of the condition, pain etc).

Steps:
1. Data collection: 3 functions are created to collect data from cms_procedure library which would convert the collected data to pandas dataframe
2. Data Analysis and preprocessing: Because preocedure outcome variables won't be available for final prediciton, I disregarded these featuere in model training phase. So my full dataset contains only attributes and outcomes.
  
  Assumptions that i considered: a. Number of missing values are less than 5% and they do not have any pattern relating with the outcome. So, I decided to delete all the missing values. The procedure would be different if there were many missing values or if they were not random
  
  b. Variables contained in the attributes make sense in term of domain knowledge. Also, when I looked at correlation plots, I did not discover a high correlation value. A high correlation value between any pair of features would require possible discarding of one of the variables. My final decision was to consider all the attributes for my model.
  
  c. No duplicates in the dataset. Which means I did not need to drop duplicate values.
  
3. Model building: This is a binary classification task. The choice could be using a logstic regression model, k nearest neighbor or a tree based model. Because the statement did not require to use several models, I will stick with one.
  
  
