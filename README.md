# Protein_Prediction
### Names: Ethan Deng, Jason Gu

# Overview
Here is our exploratory data analysis on this dataset: https://ethandengg.github.io/Protein_Analysis/


<!-- #region -->
# Framing the Problem <a name="framingtheproblem"></a>
In our project, we will build a model that will predict the amount of protein in a recipe by looking at the features in the nutrition column. The nutrition column includes the calories, total_fat, sugar, sodium, saturated fats, and carbohydrates. These features seem to have a coorelation to the amount of protein there is in a recipe. This is a regression problem, not a classification problem because we are trying to predict a quantitative value (amount of protein in grams).

- **Response Variable**: We chose **protein** as the response variable because it is a quantitative discrete variable (as the amount of protein in grams here is always listed as a whole number), meaning that we can build a regression model that predicts amount of protein based on other features in the data. In the real world, this is an important because many gym goers value the amount of protein food has, so being able to predict how much protein there is in a recipe is valuable.


- **Evaluation Metrics**: Since we are using a regression model, we will look at the RMSE, MAE, and R^2 vlaues to evaluate the effectiveness of our model. The MAE measures the average absolute difference between the predicted protein values and the acutal protein values in our dataset, with a lower MAE meaning that our model's predictions are closer to the true values on average. The RMSE is similar to the MAE but gives more weight to larger errors as it measures the square root of the average squared differences between predicted and actual values and is also more sensitive to outliers. The R^2 value essential tells us how well our model fits the data by quantifying the proportion of variance in the protein content in the model, ranging from 0 to 1, with 1 being a perfect fit. 


- **Information Known**: At the time of prediction, we have access to the nutrition label, which includes (**'calories', 'total_fat', 'sugar', 'sodium', 'saturated fat', and 'cabrs'**) containing all the information except protein. With this known information, we can predict the amount of protein there is in recipes that don't include protein on the nutrition label.
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
For our final model, we moved away from the Linear Regression Prediction Model because it was lacking in performance as seen in the evaluation metrics of R^2, RMSE, and MAE. We decided to a try a Random Forest Regressor for a prediction model because random forest is better at dealing with imbalanced data and less prone to overfitting.

- **Description**: Our final model includes 3 additional features, including **'sugar', 'saturated fat', and 'carbs'** which are all quantitative continuous variables. We added these features because of our personal experiences with how these other nutriets are related to protein. For example, foods high in carbs and sugar are typically not rich in protein while foods with more saturated fats typically have more protein, such as meats. From our EDA of this dataset, we found that number of minutes and average rating were weakly coorelated with the amount of protein in a recipe, so we ruled out these features as being appropriate to our prediction model. We are still using RMSE, MAE, and R^2 to evaluate the performance of our model.
  
  
- **Feature Transformations**: Like the first 3 variables used in the basic model, we performed the Standard Scaling Transformation to the following features: 'sugar', 'saturated fat', and 'carbs'. These new features also got the same treatement as the ones from the basic model, 
  
Recalling from DSC40A, adding more features is the key to fitting a model better. This means that the features could we added are going to improve the generalization of our final model to unseen data. To further analyze why these features improved our model, here are a few reasons.

- **First**: Adding more features regarding the nutritional information of the recipe will help us decide what kind of food that it is. For example, foods with a high amount of sugar is more likely to have less protein than other foods because foods high in sugar consists deserts that are typically low in protein. 


- **Second**: We don't want to categorize our features into boolean values, such as setting a column where (sugar > x value) because the value we are trying to predict (grams of protein) is a quanititative continuous variable. Categorizing our features from the nutrition data will take away information from our features, making the model have a harder time predicted the amount of protein there is in a recipe.

## Algorithm and Hyperaparmeters 
We chose the Random Forest Regressor to predict our model because it's better at finding non-linear relationships between the inputs and outputs, is less prone to overfitting compared to other complex models like decision trees, and can handle datasets with irrelevant features without significantly impacting performance. 

We used GridSearchCV with varying numbers of fold ranging from 5 to 15 to find the most optimal hyperparameters. These are represented as comments in the code because the code takes a long time to run. 

For hyperparemters decided to use a combination of number of estimators, max depth, and max features. 

The hyperparameters that ended up performing the best in the new model are as follows:

    Number of Estimators (n_estimators): 15
    Maximum Depth of Trees (max_depth): None
    Maximum Features (max_features): 'sqrt'

Comparing the performance of the baseline model to the new model:

**Baseline Model**:

![image.jpeg](attachment:image.jpeg)


**New Model**:

![image-2.jpeg](attachment:image-2.jpeg)


  
- **Performance**: The new model significantly outperforms the baseline model in all metrics. The RMSE, which measures the model's prediction error, has substantially decreased from 24.673 to 8.435. The R2, which indicates the proportion of the variance in the target variable explained by the model, has increased from 0.281 to 0.916. Additionally, the MAE, which measures the average absolute error of predictions, has reduced from 19.212 to 4.792. Overall, the new model with hyperparameter tuning demonstrates much better predictive performance, indicating that the grid search for hyperparameters has resulted in a substantially improved model compared to the baseline model.

<!-- #endregion -->
