# MachineLearningModel_LinearRegresion

**1. Why do we have to perform EDA before fitting a model to the data?**
   - Exploratory Data Analysis (EDA) helps us understand the underlying patterns, relationships, and anomalies in the data. It allows us to detect outliers, handle missing values, and choose the appropriate features, thus improving model accuracy and avoiding potential pitfalls.

**2. What is a parameter?**
   - A parameter is a quantity that defines a particular model. In machine learning, parameters refer to the internal variables that the model learns from the data during training (e.g., coefficients in a linear regression model).

**3. What is correlation?**
   - Correlation is a statistical measure that expresses the extent to which two variables are linearly related. It ranges from -1 to 1, where values close to 1 indicate a strong positive relationship, and values close to -1 indicate a strong negative relationship.

**4. What does negative correlation mean?**
   - Negative correlation means that as one variable increases, the other decreases. In a perfectly negative correlation (-1), an increase in one variable corresponds exactly to a decrease in the other.

**5. How can you find correlation between variables in Python?**
   - You can use the `corr()` function from pandas or the `pearsonr()` function from the `scipy.stats` module.

   ```python
   # Example using pandas
   df.corr()

   # Example using scipy
   from scipy.stats import pearsonr
   corr, _ = pearsonr(variable1, variable2)
   ```

**6. What is causation? Explain the difference between correlation and causation with an example.**
   - Causation implies that one event causes another to happen, while correlation simply indicates that two variables move together, but without implying a cause-effect relationship. For example, ice cream sales and drowning incidents may be correlated (both increase during the summer), but eating ice cream doesn’t cause drowning (causation).

**7. Define Linear Regression.**
   - Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

**8. What is univariate linear regression?**
   - Univariate linear regression is a type of linear regression where there is only one independent variable used to predict the dependent variable.

**9. What is multivariate linear regression?**
   - Multivariate linear regression involves two or more independent variables used to predict the dependent variable.

**10. What are weights and bias?**
   - Weights are coefficients applied to input features that determine their influence on the output. Bias is a constant added to the model output to help fit the model more accurately.

**11. What are inputs and targets?**
   - Inputs are the features (independent variables) provided to a model, while targets are the actual outcomes or dependent variables that the model is trying to predict.

**12. What is loss/cost function?**
   - The loss function measures the difference between the predicted and actual values. It is used to assess the performance of a model. Common examples include Mean Squared Error (MSE) and Cross-Entropy Loss.

**13. What is residual?**
   - A residual is the difference between the observed value and the predicted value of the dependent variable. It indicates the error made by the model.

**14. What is RMSE value? When and why do we use it?**
   - RMSE (Root Mean Squared Error) is a metric that calculates the square root of the average of squared differences between predicted and actual values. It’s used to measure the model’s prediction error and is preferred when large errors need to be penalized more.

**15. What is an Optimizer? What are different types of optimizers? Explain each with an example.**
   - An optimizer is an algorithm that adjusts model parameters (weights and biases) to minimize the loss function. Examples include:
     - **Gradient Descent**: Iteratively adjusts parameters to minimize the loss.
     - **Adam**: An advanced optimizer that adapts learning rates for each parameter based on momentum and scaling.
     - **RMSProp**: Optimizer that works well with non-stationary data, adjusting the learning rate based on a moving average of gradients.

**16. What library is available in Python to perform Linear Regression?**
   - `scikit-learn` is a commonly used library for linear regression in Python.

**17. What is sklearn.linear_model?**
   - It is a module in `scikit-learn` that contains classes for linear models, including linear regression, ridge regression, and LASSO.

**18. What does model.fit() do? What arguments must be given?**
   - `model.fit()` trains the model using the provided input (features) and target (labels). The required arguments are `X` (features) and `y` (target).

**19. What does model.predict() do? What arguments must be given?**
   - `model.predict()` generates predictions using the fitted model. It requires the input features `X` for which predictions are needed.

**20. How do we calculate RMSE values?**
   - You can calculate RMSE using the following formula:

   ```python
   from sklearn.metrics import mean_squared_error
   from math import sqrt

   rmse = sqrt(mean_squared_error(y_true, y_pred))
   ```

**21. What is model.coef_?**
   - `model.coef_` returns the coefficients (weights) of the features learned by the model.

**22. What is model.intercept_?**
   - `model.intercept_` returns the bias (intercept) term in the linear regression equation.

**23. What is SGDRegressor? How is it different from Linear Regression?**
   - `SGDRegressor` is a type of linear model that uses stochastic gradient descent to optimize the loss function. It’s different from ordinary linear regression because it updates the model weights iteratively for each data point, making it suitable for large-scale datasets.

**24. Define Machine Learning. What are the main components in Machine Learning?**
   - Machine Learning is the field of study that gives computers the ability to learn from data without explicit programming. The main components include:
     - **Data**: The foundation for training models.
     - **Model**: A function or system that maps inputs to outputs.
     - **Loss Function**: Measures the error in predictions.
     - **Optimizer**: Updates the model to reduce the error.

**25. How does loss value help in determining whether the model is good or not?**
   - A lower loss value indicates a better-performing model, as it means the predictions are closer to the actual target values.

**26. What are continuous and categorical variables?**
   - Continuous variables can take any value within a range (e.g., temperature). Categorical variables represent distinct groups or categories (e.g., gender, colors).

**27. How do we handle categorical variables in Machine Learning? What are the common techniques?**
   - Common techniques for handling categorical variables include:
     - **One-Hot Encoding**: Converts categories into binary columns.
     - **Label Encoding**: Assigns numerical labels to categories.

**28. What is feature scaling? How does it help in Machine Learning?**
   - Feature scaling ensures that all input features have a similar scale, which helps models like gradient descent converge faster and prevents certain features from dominating others.

**29. How do we perform scaling in Python?**
   - You can perform scaling using `StandardScaler` or `MinMaxScaler` from `sklearn.preprocessing`.

   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

**30. What is sklearn.preprocessing?**
   - `sklearn.preprocessing` is a module in scikit-learn that provides methods for scaling, encoding, and normalizing data.

**31. What is a Test set?**
   - A test set is a subset of the data used to evaluate the performance of a trained model. It helps determine how well the model generalizes to unseen data.

**32. How do we split data for model fitting (training and testing) in Python?**
   - You can split data using `train_test_split` from `sklearn.model_selection`.

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```

**33. How do you approach a Machine Learning problem?**
   - The general approach includes:
     1. **Understand the problem**.
     2. **Collect and preprocess data** (handle missing values, outliers, etc.).
     3. **Perform EDA** to explore relationships.
     4. **Select features** and split the data into training and test sets.
     5. **Train a model**.
     6. **Evaluate the model** using metrics like RMSE or accuracy.
     7. **Tune the model** if needed for better performance.
