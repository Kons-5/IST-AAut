#!/bin/python3

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor, HuberRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.svm import SVR


def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])


def LOFPlot(y_train, y_pred_outliers, X_scores):
    plt.scatter(range(len(y_train)), y_train, c=y_pred_outliers,
                cmap='coolwarm', s=30)
    plt.title("Outliers detected by LOF")
    plt.xlabel("Sample index")
    plt.ylabel("Residuals")

    # Plot circles with radius proportional to the outlier scores
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    plt.scatter(
        range(len(y_train)),
        y_train,
        s=1000 * radius,
        edgecolors="k",
        facecolors="none",
        label="Outlier scores",
    )
    plt.show()


def ScatterPlot(mask, y_train):
    plt.title("Inliers detected")
    plt.scatter(range(len(y_train)), y_train, c=mask,
                cmap='coolwarm', label='Inliers')
    plt.xlabel("Sample index")
    plt.ylabel("y values (Toxic Algae Concentration)")
    plt.show()


def iterative_lof_outlier_removal(model, X_train, y_train, contamination=0.25, max_iterations=5):
    model = HuberRegressor()
    for iteration in range(max_iterations):
        # Step 1: Fit the model on the data
        model.fit(X_train, y_train)

        # Step 2: Predict Y values and calculate residuals
        y_train_pred = model.predict(X_train)
        residuals = (y_train - y_train_pred)**2

        # Step 3: Apply LOF on the residuals to detect outliers
        lof = LocalOutlierFactor(contamination=contamination, n_neighbors=95)
        outlier_labels = lof.fit_predict(
            residuals.reshape(-1, 1))  # LOF on residuals

        X_scores = lof.negative_outlier_factor_

        # Step 4: Filter out the outliers
        mask = outlier_labels == 1  # Keep only inliers
        X_train_filtered = X_train[mask]
        y_train_filtered = y_train[mask]

        LOFPlot(residuals, outlier_labels, X_scores)

        # If no more outliers are found, break the loop
        if len(X_train_filtered) == len(X_train):
            print(f"No more outliers detected after {iteration} iterations.")
            break

        # Update the training set by removing outliers
        print(
            f"Iteration {iteration + 1}: Removed {len(y_train) - len(y_train_filtered)} outliers.")
        X_train, y_train = X_train_filtered, y_train_filtered

    return X_train_filtered, y_train_filtered


def CrossValidation(X, y, model, param_grid):

    # Define the scoring metric
    scoring = make_scorer(mean_squared_error, greater_is_better=False)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=5,
        n_jobs=-1
    )

    # Fit the GridSearchCV to find the best parameters
    grid_search.fit(X, y)

    # Extract the best parameters and estimator
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(
        model, X_train_cleaned, y_train_cleaned, cv=5, scoring='neg_mean_squared_error')

    # Print the mean RMSE and the variance across folds
    mean_rmse = (-cv_scores.mean()) ** 0.5
    rmse_std = cv_scores.std() ** 0.5

    print(f"\nCross-Validated RMSE: {mean_rmse}")
    print(f"RMSE Std Deviation: {rmse_std}")

    print("Best Hyperparameters:")
    print(best_params)

    return best_model


# Load data
X_train = np.load("../../data/X_train.npy")
y_train = np.load("../../data/y_train.npy")


X_train_cleaned, y_train_cleaned = iterative_lof_outlier_removal(
    LinearRegression(), X_train, y_train, contamination=0.005, max_iterations=50)

# Define parameter grid
param_grid = {
    'alpha': np.arange(0.1, 100, 0.1)
}
"""
param_grid = {
    'C': [1],
    'epsilon': [0.1]
}
"""
model = CrossValidation(X_train_cleaned, y_train_cleaned,
                        Ridge(), param_grid)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train_cleaned, y_train_cleaned, test_size=0.25, random_state=42)

# Refit the best model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_test_pred = model.predict(X_train_cleaned)
residuals = y_train_cleaned - y_test_pred

# Evaluate performance
test_mse = mean_squared_error(y_train_cleaned, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_train_cleaned, y_test_pred)
sse = np.sum((residuals) ** 2)

print("\nPerformance Metrics on Test Set:")
print(
    f"Intercept: {model.intercept_}\nCoefficients: {model.coef_}\n")
print(f"Test RMSE: {test_rmse}")
print(f"Test R-squared: {test_r2}")
print(f"Sum of Squared Errors (SSE): {sse}")

# Plot residuals
plt.scatter(y_test_pred, residuals**2)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.show()
