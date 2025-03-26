# Necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Data Loading and Cleaning
# -------------------------------

# Defining column names according to the original dataset
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Reading the dataset
df = pd.read_csv('adult.data', header=None, names=col_names)

# Display income distribution
print("Income distribution:")
print(df.income.value_counts(normalize=True))

# Remove extra spaces in all object-type columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Create binary target variable: 0 for '<=50K', 1 for '>50K'
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# -------------------------------
# 2. Function to Train and Evaluate the Model
# -------------------------------
def tune_random_forest(max_depth_range, X_train, y_train, X_test, y_test, random_state=1):
    """
    Trains RandomForestClassifier models for different depths,
    returns lists of accuracy for training and testing, and the best max_depth value.
    """
    acc_train = []
    acc_test = []
    
    for depth in max_depth_range:
        model = RandomForestClassifier(max_depth=depth, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Predictions for training and testing
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        acc_train.append(accuracy_score(y_train, y_train_pred))
        acc_test.append(accuracy_score(y_test, y_test_pred))
    
    # Determine the best max_depth based on test accuracy
    best_index = np.argmax(acc_test)
    best_depth = max_depth_range[best_index]
    best_acc = acc_test[best_index]
    
    return acc_train, acc_test, best_depth, best_acc

# -------------------------------
# 3. Model with Original Features
# -------------------------------

# Define initial features
feature_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race']

# Generate feature dataframe with dummies for categorical variables (drop_first=True)
X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df['income']

# Split data into training and testing (20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 3.1. Default Model (baseline)
rf_default = RandomForestClassifier(random_state=1)
rf_default.fit(X_train, y_train)
default_acc = rf_default.score(X_test, y_test)
print(f'\nDefault Random Forest accuracy: {default_acc*100:.3f}%')

# 3.2. Tuning max_depth hyperparameter from 1 to 25
depth_range = range(1, 26)
acc_train, acc_test, best_depth, best_acc = tune_random_forest(depth_range, X_train, y_train, X_test, y_test)
print(f'\nBest max_depth (original features): {best_depth}')
print(f'Best test accuracy: {best_acc*100:.3f}%')

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(depth_range, acc_test, 'bo--', label='Test Accuracy')
plt.plot(depth_range, acc_train, 'r*:', label='Train Accuracy')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. max_depth (Original Features)')
plt.legend()
plt.show()

# Train model with best max_depth and display feature importances
best_rf = RandomForestClassifier(max_depth=best_depth, random_state=1)
best_rf.fit(X_train, y_train)
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
print('\nTop 5 features (original features):')
print(feature_importances.head(5))

# -------------------------------
# 4. Model with Additional Features (including education_bin)
# -------------------------------

# Create new education grouping feature
df['education_bin'] = pd.cut(df['education-num'], bins=[0, 9, 13, 16], 
                             labels=['HS or less', 'College to Bachelors', 'Masters or more'])

# Update feature list
feature_cols_extended = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race', 'education_bin']

# Generate feature dataframe with dummies (drop_first=True)
X_extended = pd.get_dummies(df[feature_cols_extended], drop_first=True)

# Recreate training/testing split
X_train_ext, X_test_ext, y_train, y_test = train_test_split(X_extended, y, test_size=0.2, random_state=1)

# Tune max_depth hyperparameter for the new feature set
depth_range_ext = range(1, 26)  # Adjust range as needed
acc_train_ext, acc_test_ext, best_depth_ext, best_acc_ext = tune_random_forest(depth_range_ext, X_train_ext, y_train, X_test_ext, y_test)
print(f'\nBest max_depth (extended features): {best_depth_ext}')
print(f'Best test accuracy (extended features): {best_acc_ext*100:.3f}%')

# Plot results for extended model
plt.figure(figsize=(8, 5))
plt.plot(depth_range_ext, acc_test_ext, 'bo--', label='Test Accuracy')
plt.plot(depth_range_ext, acc_train_ext, 'r*:', label='Train Accuracy')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. max_depth (Extended Features)')
plt.legend()
plt.show()

# Train model with extended features and display feature importances
best_rf_ext = RandomForestClassifier(max_depth=best_depth_ext, random_state=1)
best_rf_ext.fit(X_train_ext, y_train)
feature_importances_ext = pd.DataFrame({
    'feature': X_train_ext.columns,
    'importance': best_rf_ext.feature_importances_
}).sort_values('importance', ascending=False)
print('\nTop 5 features (extended features):')
print(feature_importances_ext.head(5))
