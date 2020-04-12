## R-Squared Intuition

- Simple Linear Regression
    - Sum of squares of residuals (SS_res): SUM ((y_i - y^_i)^2)
    - Total Sum of Squares (SS_tot): SUM ((y_i - y_avg)^2)
    - R^2 = 1 - SS_res / SS_total
        - How good is the SS_res compared to the SS_avg
        - Values close to 1 are better. The farther away from 1 the worse

## Adjusted R-Squared Intution

- Minimize SS_res.
- Use R^2 as a goodness of fit
    - problems occur when you add more variables to the model
    - R^2 will *Never Decrease* when a new variable is added
        - Adding the new variable will improve but won't not worsen SS_res minimizing
        - Worst case the coeeficient of the new variable will be zero
        - Often though though there is random correlation that results in a very small coefficient for a theoretically uncorellated new variable (i.e. Name)
    - R^2 is pretty much always increasing

- Adjusted R^2:
    - R^2_adj = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
        - p = number of regressors
        - n = sample size
    - has a penalization factor for adding variables that don't help the model much

## Evaluating Regression Models Performance

- The more variables, the larger R^2
- Notice when Adj_R^2 is maximized

## Interpreting coefficients
- Positive coefs are correlated
- Magnitude is more tricky
    - scaling values matter ... units are important
    - always say magnitude in units of variable

## Pros Cons of Regressions
- Linear Regression
    - Pros: Works on any size of dataset, gives information about relevance of features
    - Cons: The Linear Regression Assumptions
- Polynomial Regression
    - Pros: Works on any size of dataset, works very well on non-linear problems
    - Cons: Need to choose the right polynomial degree for a good bias/variance tradeoff
- SVR
    - Pros: Easily adaptable, works very well on non-linear problems, not biased by outliers
    - Cons: Compulsory to apply feature scaling, not well known, more difficult to understand
- Decision Tree Regression
    - Pros: Interpretability, no need for feature scaling, works well on both linear / non-linear problems
    - Cons: Poor results on too small datasets, overfitting can easily occur
- Random Forest Regression
    - Pros: Power and accurate, good performance on many problems, including non-linear
    - Cons: No interpretability, overfitting can easily occur, need to choose the number of trees
