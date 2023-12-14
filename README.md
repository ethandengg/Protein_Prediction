# Protein_Prediction
### Names: Ethan Deng, Jason Gu

# Overview
Here is our exploratory data analysis on this dataset: https://ethandengg.github.io/Protein_Analysis/


<!-- #region -->
# Framing the Problem <a name="framingtheproblem"></a>
In our project, we will build a model that will predict the amount of protein in a recipe by looking at the features in the nutrition column. The nutrition column includes the calories, total_fat, sugar, sodium, saturated fats, and carbohydrates. These features seem to have a correlation to the amount of protein there is in a recipe. This is a regression problem, not a classification problem because we are trying to predict a quantitative value (amount of protein in grams).

- **Response Variable**: We chose **protein** as the response variable because it is a quantitative discrete variable (as the amount of protein in grams here is always listed as a whole number), meaning that we can build a regression model that predicts the amount of protein based on other features in the data. In the real world, this is important because many gym goers value the amount of protein food has, so being able to predict how much protein there is in a recipe is valuable.


- **Evaluation Metrics**: Since we are using a regression model, we will look at the RMSE, MAE, and R^2 values to evaluate the effectiveness of our model. The MAE measures the average absolute difference between the predicted protein values and the actual protein values in our dataset, with a lower MAE meaning that our model's predictions are closer to the true values on average. The RMSE is similar to the MAE but gives more weight to larger errors as it measures the square root of the average squared differences between predicted and actual values and is also more sensitive to outliers. The R^2 value essential tells us how well our model fits the data by quantifying the proportion of variance in the protein content in the model, ranging from 0 to 1, with 1 being a perfect fit. 


- **Information Known**: At the time of prediction, we have access to the nutrition label, which includes (**'calories', 'total_fat', 'sugar', 'sodium', 'saturated fat', and 'carbs'**) containing all the information except protein. With this known information, we can predict the amount of protein there is in recipes that don't include protein on the nutrition label.
<!-- #endregion -->

<!-- #region -->

# Baseline Model <a name="baselinemodel"></a>

- **Description**: In our baseline model, there are 4 predictor features,  **'minutes', 'calories', 'total_fat', and 'sodium'** which are all quantitative continuous variables. For this baseline model, we did not have categorical features so no encoding or categorical transformation was done on our given features. We kept our quantitative continuous features as raw integer values. In all, we use 'minutes', 'calories', 'total_fat', and 'sodium', as features of our baseline Linear Regression Prediction model.
  
  
- **Feature Transformations**: We performed the Standard Scaling Transformation to the following features: 'minutes', 'calories', 'total_fat', and 'sodium'. The StandardScaler is applied to the specified columns in the ColumnTransformer. Since these features were quantitative features that provide a numerical representation of the nutrition of each recipe, we ensured to subtract the mean value and divide them by the standard deviation of that feature.
  
  
- **Performance**: The Linear Regression Prediction Model is not great in performance. The primary issue with this model is that it is overly simplistic and tends to make predictions that do not accurately capture the complexity of the data. Specifically, the model appears to predict a similar value for most instances, indicating a lack of variation in different recipe predictions. One approach is to address the class imbalance by either oversampling the minority class (recipes with specific protein counts) or undersampling the majority class (recipes with a more common protein count). Another way to improve our protein prediction is to assign weights to the different protein count classes during model training. By giving higher weights to the minority classes, the model can be encouraged to pay more attention to those classes, potentially improving its predictive accuracy while still using the entire dataset.


- The evaluation metrics are shown below:


<img width="444" alt="Screenshot 2023-12-13 at 4 19 21 PM" src="https://github.com/JingChengGu/Protein_Prediction/assets/64511500/78778385-e955-4051-8503-3a95c4e5782b">
<!-- #endregion -->

<!-- #region -->

# Final Model <a name="finalmodel"></a>
For our final model, we moved away from the Linear Regression Prediction Model because it was lacking in performance as seen in the evaluation metrics of R^2, RMSE, and MAE. We decided to try a Random Forest Regressor for a prediction model because random forest is better at dealing with imbalanced data and less prone to overfitting.

- **Description**: Our final model includes 3 additional features, including **'sugar', 'saturated fat', and 'carbs'** which are all quantitative continuous variables. We added these features because of our personal experiences with how these other nutrients are related to protein. For example, foods high in carbs and sugar are typically not rich in protein while foods with more saturated fats typically have more protein, such as meats. From our EDA of this dataset, we found that the number of minutes and average rating was weakly correlated with the amount of protein in a recipe, so we ruled out these features as being appropriate to our prediction model. We are still using RMSE, MAE, and R^2 to evaluate the performance of our model.
  
  
- **Feature Transformations**: Like the first 3 variables used in the basic model, we performed the Standard Scaling Transformation to the following features: 'sugar', 'saturated fat', and 'carbs'. These new features also got the same treatment as the ones from the basic model, 
  
Recalling from DSC40A, adding more features is the key to fitting a model better. This means that the features we added are going to improve the generalization of our final model to unseen data. To further analyze why these features improved our model, here are a few reasons.

- **First**: Adding more features regarding the nutritional information of the recipe will help us decide what kind of food it is. For example, foods with a high amount of sugar are more likely to have less protein than other foods because foods high in sugar consists of deserts that are typically low in protein. 


- **Second**: We don't want to categorize our features into boolean values, such as setting a column where (sugar > x value) because the value we are trying to predict (grams of protein) is a quantitative continuous variable. Categorizing our features from the nutrition data will take away information from our features, making the model have a harder time predicting the amount of protein there is in a recipe.

## Algorithm and Hyperaparmeters 
We chose the Random Forest Regressor to predict our model because it's better at finding non-linear relationships between the inputs and outputs, is less prone to overfitting compared to other complex models like decision trees, and can handle datasets with irrelevant features without significantly impacting performance. 

We used GridSearchCV with varying numbers of folds ranging from 5 to 15 to find the most optimal hyperparameters. These are represented as comments in the code because the code takes a long time to run. 

For hyperparemters, we decided to use a combination of the number of estimators, max depth, and max features. 

The hyperparameters that ended up performing the best in the new model are as follows:

    Number of Estimators (n_estimators): 15
    Maximum Depth of Trees (max_depth): None
    Maximum Features (max_features): 'sqrt'

Comparing the performance of the baseline model to the new model:


- The evaluation metrics are shown below:
<img width="660" alt="Screenshot 2023-12-13 at 8 00 57 PM" src="https://github.com/JingChengGu/Protein_Prediction/assets/64511500/05be2616-ce64-46aa-9c37-a4bc606d9d9b">



  
- **Performance**: The new model significantly outperforms the baseline model in all metrics. The RMSE, which measures the model's prediction error, has substantially decreased from 24.673 to 8.435. The R2, which indicates the proportion of the variance in the target variable explained by the model, has increased from 0.281 to 0.916. Additionally, the MAE, which measures the average absolute error of predictions, has reduced from 19.212 to 4.792. Overall, the new model with hyperparameter tuning demonstrates much better predictive performance, indicating that the grid search for hyperparameters has resulted in a substantially improved model compared to the baseline model.

<!-- #endregion -->

<!-- #region -->
# Fairness Analysis
In this analysis, we are comparing the RMSE of two groups: meat recipes (Group X) and non-meat recipes (Group Y). The meat recipes consist are conducted based on whether or not the words 'beef', 'chicken', 'pork', or 'fish' appear in the ingredients column. RMSE is used as the evaluation metric to measure the difference in prediction accuracy between the two groups. We are interested in determining whether there is a significant difference in the prediction accuracy of protein content (as measured by RMSE) between these two groups.


- **Null Hypothesis**: Our model is fair and there is no significant difference in the prediction accuracy (RMSE) of protein content between meat recipes and non-meat recipes.


- **Alternative Hypothesis**: There is a significant difference in the prediction accuracy (RMSE) of protein content between meat recipes and non-meat recipes.


- **Test Statistic**: We will use a permutation test to assess the significance of the difference in RMSE between the two groups. The test statistic is the difference in RMSE between the observed groups (meat and non-meat recipes).


- **Significance Level (Alpha)** : Let's set the significance level (alpha) to 0.01


- **Procedure**: We will shuffle the labels of meat and non-meat recipes and calculate the RMSE for each shuffled dataset. By repeating this process 1000 times, we can create a distribution of the test statistic under the assumption that there is no difference between the groups. We will then compare the observed test statistic to this distribution to calculate the p-value.

### Results:
**RMSE for Meat Recipes**: 5.660917830525564

**RMSE for Non-Meat Recipes**: 4.362381737844239

**p-value from permutation test**: 0.00

### Conclusion:
The analysis comparing the prediction accuracy with RMSE between meat recipes and non-meat recipes yielded a statistically significant difference. The RMSE for non-meat recipes (4.362) is significantly lower than that for meat recipes (5.667). The p-value of 0 indicates strong evidence against the null hypothesis, suggesting that there is indeed a significant difference in the prediction accuracy of protein content between meat and non-meat recipes based on the RMSE metric, meaning that our model is not fair.
<!-- #endregion -->



