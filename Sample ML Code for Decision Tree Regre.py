# Sample ML Code for Decision Tree Regression with K-Fold Cross-Validation
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import pandas as pd
import numpy as np

df = pd.read_csv()
# Drop unnecessary columns and impute missing values first
# Check for outliers and handle them if necessary
# Heatmaps and boxplots can be used for visualization

y = df[target_y_column]
X = df.drop(target_y_column, axis=1)

# Split the data into training+validation and test sets (test set not used during K-Fold CV)
train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)
# Define model and hyperparameters
model = RandomForestRegressor(random_state=0)
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']}

# Each value means:
# 'auto' = all features (deprecated in newer versions; now equivalent to 1.0 for regression, or sqrt for classification)
# 'sqrt' = √(number of features), default for classification
# 'log2' = log₂(number of features), more aggressive feature sampling — useful for reducing overfitting

# Gridsearchcv will try all combinations of these hyperparameters
# Total number of combinations = 3 n_estimators × 3 max_depth × 3 min_samples_split × 2 max_features = 54 combinations
# GridSearchCV runs CV (e.g., 5-fold) for each combo, so 54 × 5 = 270 total model fits

# Try different numbers of CV folds (e.g., 3, 5, 10) to see how it affects performance
# More folds = more training time, but can lead to better generalization
cv_values = [3, 5, 7, 10]  # Example values, can be adjusted
cv_scores = []

for cv in cv_values:
    print(f"Running GridSearchCV with {cv}-fold CV...")
    model = RandomForestRegressor(random_state=0)

# Step 3: GridSearchCV with 5-fold CV using negative MAE (since sklearn minimizes loss)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1)

# Fit GridSearch on training+validation data
grid_search.fit(train_X, train_y)
best_score = grid_search.best_score_
print(f"Best CV neg MAE score for cv={cv}: {best_score:.4f}")
cv_scores.append(best_score)

# Choose the cv with the best (highest) negative MAE
best_cv_index = np.argmax(cv_scores)
best_cv = cv_values[best_cv_index]
print(f"\nBest cv value: {best_cv} with neg MAE: {cv_scores[best_cv_index]:.4f}")

# Get best hyperparameters 
print("Best Hyperparameters:", grid_search.best_params_)

# Train final model with train_X set
final_model = grid_search.best_estimator_
final_model.fit(train_X, train_y)

# Evaluate on test set
y_pred = best_model.predict(test_X)

# Evaluate with all metrics
mae = mean_absolute_error(test_y, y_pred)
mse = mean_squared_error(test_y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(test_y, y_pred)

# Print evaluation metrics to 3f
print(f"MAE: {mae:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}")
print(f"R^2: {final_model.score(test_X, test_y):.3f}")

print("\nFeature importances:")
for idx, importance in enumerate(final_model.feature_importances_):
    print(f"Feature {idx}: {importance:.4f}") 

#Train/validation/test: 
# Test set
# 
#  ['
# \\\]
#=0pouj  /] is held out until final evaluation.
#GridSe?.,ng='neg_mean_absolute_error': GridSearch uses negative MAE so that higher = better.
# Metrics reported on test set only (not CV scores) — this is what matters for deployment.