# Cormac_assessment
Input Data: Dictionary of attributes (Types of procedure, How long the procedure lasted, Severity of the condition), Dictionary of variables containing success/failure for the specified attributes and dictionary of granular measure of outcomes (recurrence of the condition, pain etc).

Steps:
1. Data collection: 3 functions are created to collect data from cms_procedure library which would convert the collected data to pandas dataframe
2. Data Analysis and preprocessing: Because granular outcome variables won't be available for final prediciton, I disregarded these featuere in model training phase. So my full dataset contains only attributes and outcomes.
  
  Assumptions that i considered in this phase:
  a. Number of missing values are less than 5% and they do not have any pattern relating with the outcome. So, I decided to delete all the missing values. The procedure would be different if there were many missing values or if they were not random
  
  b. Variables contained in the attributes make sense in term of domain knowledge. Also, when I looked at correlation plots, I did not discover a high correlation value. A high correlation value between any pair of features would require possible discarding of one of the variables. My final decision was to consider all the attributes for my model.
  
  c. No duplicates in the dataset. Which means I did not need to drop duplicate values.
  
3. Model building: This is a binary classification task. The choice could be using a logstic regression model, k nearest neighbor or a tree based model. Because the statement did not require to use several models, I will stick with one. I will use a Random Forest model because it can handle non-linearity very well, as well as it does not require prior assumptions, can handle categorical veriables well and does not requiure feature scaling. Here are the assumptions i used in this phase:
  a. Granular measures will not be used for final prediction so I did not consider granular features when training the model. 
  
4. Coding: cormac_train.py will extract the data from source, do some simple preprocessing and build a random forest model which will use grid search to get the best sets of parameters. (Tree depth and number of estimators. There are other variables to consider but kept it simple for this project). Our model will output mean test score, mean test standard devision for each of the combinations using print_results function. With the evaluate_model function it will show precision, recall and accuracy for the test data.

  Because I don't exactly have the new sets of attributes, I assumed they are just provided to me. So, in the cormac_prediction.py file, user has to provide the directory of the pre-trained model and we will use that model to get predictions for those new attrbutes.
  
 5. Further imrovements: Here are some proposals on further improvements:
    a. Try to reduce features of the model by using domain knowledge, exploratory analysis and looking at feature importance from our random forest model.
    b. Use Other classification algorthm such as knn, logistic regression or other tree based model such as Xgboost, adaboost and compare accuracy with our model
    
Addedndum:
Here are some assumptions in this assessment that I did not address much at the time I was doing the coding, but I want to mention them here.
1. Dataset size: My assumption is that we have a large dataset. Roughly there are around 3500-4000 rows of data. The statement also aligns with this assumption (It says 'fair volume of data').
2. Feature types: I also overlooked what type of features we collected other than the mentioned ones. Some of the features are mentioned in the statement. Here is what the feature data could look like:
 Attributes: [Type of procedure(Object), How long it lasted (int), Severity (Object),  Gender (Object), Age(int), Smoking (Boolean), Number of days since last follow up (int), Blood Pressure (float)].
3. Relationship between features: My assumption would be some of the features has complex reltionship with target variables and betweeen themselves (non-linear).
4. Latency of new prediction: My assumption is, we don't need a very fast model, in other words, speed is not a crucial factor in our model.

The mentioned assumptions are necessery for Random Forest model that I used. Random Forests need large datasets, the feature relationship need to be complex and they require more time
to train than linear models.

If we had small datasets, fairly simple relationships between variables and if speed of training and making new prediction were cruical, I would probably choose a linear model like
a logistic regression or a Support Vector Machine.

One issue that I need guidance about would be how to use the granular features about the outcome. My assumption in this problem was not to include them while training the model because
we won't have access to them while making final prediciton. 
 
  

  
  
