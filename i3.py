import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the dataset
data = pd.read_csv('C:\\Users\\New\\Desktop\\Projects\\Intern_Project\\projectdata.csv')

# Drop irrelevant columns
columns_to_drop = [
    'Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address', 'Locality',
    'Locality Verbose', 'Longitude', 'Latitude', 'Currency', 'Rating color'
]
data_cleaned = data.drop(columns=columns_to_drop)

# Handle missing values
data_cleaned.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
categorical_columns = ['Cuisines', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Rating text']
for column in categorical_columns:
    le = LabelEncoder()
    data_cleaned[column] = le.fit_transform(data_cleaned[column])
    label_encoders[column] = le

# Split data for regression and classification
X = data_cleaned.drop(columns=['Aggregate rating', 'Rating text'])
y_regression = data_cleaned['Aggregate rating']  # For regression
y_classification = data_cleaned['Rating text']    # For classification

# Scaling numerical features for regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test splits
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_regression, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Regression Model with Polynomial Features
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train_reg)
X_poly_test = poly.transform(X_test_reg)

linear_reg_poly = LinearRegression()
linear_reg_poly.fit(X_poly_train, y_train_reg)
y_pred_reg_poly = linear_reg_poly.predict(X_poly_test)
mse_reg_poly = mean_squared_error(y_test_reg, y_pred_reg_poly)
accuracy_reg_poly = 1 - np.sqrt(mse_reg_poly) / y_test_reg.mean()
print("Polynomial Linear Regression Accuracy:", accuracy_reg_poly)

# Hyperparameter Tuning for Decision Tree Classifier
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
grid_search_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train_clf, y_train_clf)
best_tree_clf = grid_search_tree.best_estimator_
y_pred_best_tree = best_tree_clf.predict(X_test_clf)
accuracy_best_tree = accuracy_score(y_test_clf, y_pred_best_tree)
print("Best Decision Tree Accuracy:", accuracy_best_tree)

# Random Forest Classifier
random_forest_clf = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest_clf.fit(X_train_clf, y_train_clf)
y_pred_rf = random_forest_clf.predict(X_test_clf)
accuracy_rf = accuracy_score(y_test_clf, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Gradient Boosting Classifier
gradient_boosting_clf = GradientBoostingClassifier(random_state=42, n_estimators=100)
gradient_boosting_clf.fit(X_train_clf, y_train_clf)
y_pred_gb = gradient_boosting_clf.predict(X_test_clf)
accuracy_gb = accuracy_score(y_test_clf, y_pred_gb)
print("Gradient Boosting Accuracy:", accuracy_gb)
