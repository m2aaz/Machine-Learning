
# Multivariate Linear Regression
Dataset To Be Used: Medical Insurance Cost Prediction
Link: [https://www.kaggle.com/datasets/hetmengar/medical-insurance-cost-prediction]

## Present Features 
[Age, Sex, BMI, Children, Smoker, Region, Charges]

## Description
The Medical Insurance Cost Prediction dataset is being used to perform multi-variate linear regression, in order to estimate the expected insurance cost based on certain features.

## Data Handling
### Splitting
From the given dataset, using the **train_test_split()** function, the data is split into training & testing data (80/20 split), the data is read from a pre-downloaded file **medical-charges.csv** containing the dataset. Prior to use, the dataset must be installed and kept within the main folder where the program is to be executed.

### Preprocessing
Since the linear regression requires numerical inputs, OneHotEncoding is used. It assigns numerical values to categorical columns, effectively creating binary columns.
- To avoid perfect multicollinearity caused by dummy variables summing to 1, the first column of each encoded categorical variable is dropped.
Once the data has been encoded, the numerical values are normalised to ensure all features are comparable (ex. one doesn't dominate all the other features)

Preprocessing takes place in the **Preprocess()** function, where using the product of **"BMI"** & **"age"**, a new feature is created to compute the combined effect of these features.

To ensure the same scale & encoder is being used on both training & testing data, as only one can be passed in at a time, the **Preprocess()** function takes extra parameters, **enc** & **scale** which have been set to default, once the training data has been fitted, these features are returned to be used on the testing data.

Once the data has been fitted (scaled & normalised), a new **DataFrame** object is **returned**.

### Final Feature Selection
The final result of preprocessing is a DataFrame, one that is normalised & contains numerical columns. Though not all these features are necessarily relevant to the dataset and may have little to no impact on regression analysis.
- To check this, the correlation with respect to the output variable **"charges"** is compared, features with a correlation less than the absolute threshold value (currently set to 0.3) are dropped.
- The absolute value is compared as the data could have a positive or negative strong correlation, we only need to drop weak correlation, not the type.

The final selection of features that lie **above this threshold** are **returned**, the dataset is now ready for regression analysis.

## Regression Analysis
Note: For educational purposes, the regression model has been implemented from scratch, not using pre-built libraries.

### Feature Fetching 
- **Regress()** takes the DataFrame as a parameter, extracts both input (x) & target (y) features. 

How: Removes the target feature from the DataFrame and assigns the rest to the input feature.

### Prerequisites
- **gradients()** is set up to compute the gradient for weights (dw:vector) & bias (db:scalar), **returns dw & db**.

**dw (Vector):** Calculated via taking the partial derivative with respect to w, resulting in (-2/m) multiplied by the dot product of (y-yhat) & the input feature x (transposed).

**db (Scalar):** Calculated via taking the partial derivative with respect to b, resulting in (-2/m) multiplied by the summation of (y-yhat), as b is a constant, it isn't multiplied by x.

- **gradient_descent()** updates the values of both weights & bias, adjusting them based on the gradient and a pre-defined learning rate until the global minimum is found.

Both w & b are subtracted by their gradients * learning rate, as each subtraction is a step, as w and b converge to the global minimum, the gradient decreases hence each update naturally becomes smaller as we approach the minimum. This ensures the value of w or b doesn't overshoot, or get trapped in a loop.

The learning rate is pre-defined, a higher learning rate may cause the function to not converge at all, a lower learning rate may cause the function to converge very slow (computationally straining)

### Training The Model
All these functions combine into one, the place where the model is actually trained based on the data present. 
- We pass in the value of x & y, compute yhat using **y=wx+b** and use these values to compute the gradients
- With the gradients, values of w & b are updated through gradient descent
- If the model converges (the difference between the weights & bias vs previous values becomes less than the assigned threshold), the loop breaks & the values of w and b are returned.