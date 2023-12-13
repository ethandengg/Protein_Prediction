# Protein_Prediction
### Names: Ethan Deng, Jason Gu

# Overview
Here is our exploratory data analysis on this dataset: https://ethandengg.github.io/Protein_Analysis/



## Framing the Problem
In our project, we will build a model that will predict the amount of protein in a recipe by looking at the features in the nutrition column. The nutrition column includes the calories, total_fat, sugar, sodium, saturated fats, and carbohydrates. These features seem to have a coorelation to the amount of protein there is in a recipe. This is a regression problem, not a classification problem because we are trying to predict a quantitative value.

### Response Variables
We chose protein as the response variable because its a quantitative discrete variable (as the amount of protein in grams here is always listed as a whole number), meaning that we can build a regression model that predicts amount of protein based on other features in the data. In the real world, this is an important because many gym goers value the amount of protein food has, so being able to predict how much protein there is in a recipe is valuable.

### Evaluation Metrics:
Since we are using a regression model, we will look at the RMSE, MAE, and R^2 vlaues to evaluate the effectiveness of our model. The MAE measures the average absolute difference between the predicted protein values and the acutal protein values in our dataset, with a lower MAE meaning that our model's predictions are closer to the true values on average. The RMSE is similar to the MAE but gives more weight to larger errors as it measures the square root of the average squared differences between predicted and actual values and is also more sensitive to outliers. The R^2 value quantifies the proportion of variance in the protein content in the model, ranging from 0 to 1 with 1 being a perfect fit. 

<!-- #region -->

# Baseline Model <a name="baselinemodel"></a>

- **Description**: In our baseline model, there are 6 predictor features, **'calories'** (quantitative continuous), **'total_fat'** (quantitative continuous), **'sugar'** (quantitative continuous), **'sodium'** (quantitative continuous), **'saturated fat'** (quantitative continuous), and **'carbs'** (quantitative continuous). For this baseline model, we did not have categorical features so no encoding or categorical transformation was done on our given features. We kept our quantitative continuous features as raw integer values. In all, we use 'calories', 'total_fat', 'sugar', 'sodium', 'saturated fat', and 'carbs' as features of our baseline Linear Regression Prediction model.
  
  
- **Feature Transformations**: We performed the Standard Scaling Transformation to the following features: 'calories', 'total_fat', 'sugar', 'sodium', 'saturated fat', and 'carbs'. The StandardScaler is applied to the specified columns in the ColumnTransformer. Since these features were quantitative features that provide a numerical representation of the nutrition of each recipe, we ensured to subtract the mean value and divide them by the standard deviation of that feature.
  
  
- **Performance**: The Linear Regression Prediction Model is not great in performance. The primary issue with this model is that it is overly simplistic and tends to make predictions that do not accurately capture the complexity of the data. Specifically, the model appears to predict a similar value for most instances, indicating a lack of variation in different recipe predictions. One approach is to address the class imbalance by either oversampling the minority class (recipes with specific protein counts) or undersampling the majority class (recipes with a more common protein count). Another way to improve our protein prediction is to assign weights to the different protein count classes during model training. By giving higher weights to the minority classes, the model can be encouraged to pay more attention to those classes, potentially improving its predictive accuracy while still using the entire dataset.


- The evaluation metrics are shown below:
<img width="438" alt="Screenshot 2023-12-13 at 1 10 18 AM" src="https://github.com/JingChengGu/Protein_Prediction/assets/64511500/02edf266-69c2-469c-a6e0-cd6aaa2a358f">
<!-- #endregion -->

```python

```

```python

```
