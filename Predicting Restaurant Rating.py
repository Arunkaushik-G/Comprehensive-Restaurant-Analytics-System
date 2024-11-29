# Importing necessary libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Loading the dataset
df = pd.read_csv('C:\\Users\\New\\Desktop\\Projects\\Intern_Project\\projectdata.csv')

# Dropping unnecessary columns
columns_to_drop = ['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address', 'Locality', 
                   'Locality Verbose', 'Longitude', 'Latitude', 'Cuisines', 'Currency']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Displaying the first few records and the shape of the dataset
df.head(), df.shape

# Checking for missing and duplicated values
missing_values = df.isnull().sum()
duplicated_values = df.duplicated().sum()
df.dropna(inplace=True)

# Visualizing the distribution of 'Price range' feature
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
df['Price range'].value_counts().plot(kind='pie', autopct='%.2f', figsize=(6, 6), title='Price Range Distribution')
sns.barplot(x=df["Rating text"],y=df["Votes"],hue =df["Rating color"])
plt.show()
# Encoding categorical features
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
encoder = LabelEncoder()
categorical_columns = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 
                       'Switch to order menu', 'Rating color', 'Rating text']
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# Splitting the data into training and testing sets
X = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=353)

# Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Evaluating the Linear Regression Model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f"Linear Regression - Mean Squared Error (MSE): {mse_linear:.2f}")
print(f"Linear Regression - R-squared (R2) Error: {r2_linear:.2f}")

# Decision Tree Regressor Model
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor(min_samples_leaf=0.0001)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Evaluating the Decision Tree Regressor Model
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
print(f"Decision Tree Regressor - Mean Squared Error (MSE): {mse_tree:.2f}")
print(f"Decision Tree Regressor - R-squared (R2) Error: {r2_tree:.2f}")

# Conclusion:
# MSE of 0.05 indicates that model's predictions are very accurate with low errors.
# R2 value of 0.98 suggests that the model is highly effective at explaining & predicting the target variable.
# The Decision Tree Regressor model is performing exceptionally well on the test data.
