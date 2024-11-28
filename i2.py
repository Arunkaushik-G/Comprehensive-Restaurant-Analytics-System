'''# Importing required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform
import warnings

# Setting options and warnings
pd.reset_option('display.max_rows')
warnings.filterwarnings("ignore")

# Loading the dataset
df = pd.read_csv("C:\\Users\\New\\Desktop\\Projects\\Intern_Project\\projectdata.csv")

# Selecting relevant columns
df_reco = df[['Restaurant ID', 'Restaurant Name', 'Cuisines', 'Aggregate rating', 'Votes']]

# Function to describe the data
def describe_data(dataframe):
    desc = pd.DataFrame({
        "Column": dataframe.columns,
        "Data Type": dataframe.dtypes,
        "Missing Values": dataframe.isna().sum(),
        "Pct Missing": dataframe.isna().mean() * 100,
        "Unique Values": dataframe.nunique(),
        "Sample Values": [list(dataframe[col].dropna().sample(2).values) for col in dataframe.columns]
    })
    return desc

# Displaying data description
describe_data(df_reco)

# Dropping missing values
df_reco.dropna(inplace=True)

# Renaming columns for clarity
df_reco.rename(columns={
    'Restaurant ID': 'restaurant_id',
    'Restaurant Name': 'restaurant_name',
    'Cuisines': 'cuisines',
    'Aggregate rating': 'aggregate_rating',
    'Votes': 'votes'
}, inplace=True)

# Removing duplicates based on restaurant name, keeping the highest-rated entry
df_reco.sort_values(by=['restaurant_name', 'aggregate_rating'], ascending=False, inplace=True)
df_reco.drop_duplicates(subset='restaurant_name', keep='first', inplace=True)

# Filtering restaurants with an aggregate rating of 4.0 or higher
df_reco = df_reco[df_reco['aggregate_rating'] >= 4.0]

# Splitting cuisines into a list format for each restaurant
df_reco['cuisines'] = df_reco['cuisines'].apply(lambda x: x.split(', '))

# Expanding the dataframe to have one row per cuisine per restaurant
df_reco = df_reco.explode('cuisines')

# Creating a crosstab of restaurant names and their respective cuisines
cuisine_matrix = pd.crosstab(df_reco['restaurant_name'], df_reco['cuisines'])

# Function to compute Jaccard similarity
def compute_similarity(restaurant_a, restaurant_b, matrix):
    return jaccard_score(matrix.loc[restaurant_a], matrix.loc[restaurant_b])

# Computing Jaccard similarity between all restaurants
jaccard_distances = pdist(cuisine_matrix.values, metric='jaccard')
jaccard_similarity = 1 - squareform(jaccard_distances)

# Converting the similarity matrix into a dataframe for easy lookups
similarity_df = pd.DataFrame(jaccard_similarity, index=cuisine_matrix.index, columns=cuisine_matrix.index)

# Function to generate restaurant recommendations
def recommend_restaurants(restaurant_name, similarity_df, df_reco, threshold=0.7, top_n=5):
    similar_restaurants = similarity_df.loc[restaurant_name].sort_values(ascending=False)
    recommendations = similar_restaurants[similar_restaurants >= threshold].index.tolist()
    recommendations.remove(restaurant_name)  # Removing the input restaurant from recommendations

    recommended_df = df_reco[df_reco['restaurant_name'].isin(recommendations)].copy()
    recommended_df = recommended_df.sort_values(by='aggregate_rating', ascending=False).drop_duplicates('restaurant_name').head(top_n)

    return recommended_df[['restaurant_name', 'aggregate_rating']]

# Example of getting recommendations for a specific restaurant
recommended_restaurants = recommend_restaurants('Ooma', similarity_df, df_reco)
print(recommended_restaurants)
plt.show()
### Conclusion:
# The recommendation system provides a list of top 5 restaurants similar to the input restaurant, 
# ensuring that they have high ratings (4.0 and above). This ensures that the recommendations are both relevant and highly rated.'''
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Loading the dataset
df = pd.read_csv('C:\\Users\\New\\Desktop\\Projects\\Intern_Project\\projectdata.csv')  # Adjust the path if needed

# Dropping unnecessary columns
columns_to_drop = ['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address', 
                   'Locality', 'Locality Verbose', 'Longitude', 'Latitude', 'Cuisines', 'Currency']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Displaying the first few records and the shape of the dataset
print(df.head(), df.shape)

# Checking for missing and duplicated values
missing_values = df.isnull().sum()
duplicated_values = df.duplicated().sum()
print(f"Missing Values:\n{missing_values}")
print(f"Duplicated Values: {duplicated_values}")

# Dropping missing values
df.dropna(inplace=True)

# Visualizing the distribution of 'Price range'
plt.figure(figsize=(6, 6))
df['Price range'].value_counts().plot(kind='pie', autopct='%.2f%%', title='Price Range Distribution')
plt.show()

# Visualizing 'Rating text' vs 'Votes' with 'Rating color' as hue
plt.figure(figsize=(10, 6))
sns.barplot(x=df["Rating text"], y=df["Votes"], hue=df["Rating color"])
plt.title("Votes by Rating Text and Color")
plt.show()

# Encoding categorical features
encoder = LabelEncoder()
categorical_columns = ['Has Table booking', 'Has Online delivery', 
                       'Is delivering now', 'Switch to order menu', 
                       'Rating color', 'Rating text']
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# Splitting the data into training and testing sets
X = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=353)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Evaluating the Linear Regression Model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f"Linear Regression - Mean Squared Error (MSE): {mse_linear:.2f}")
print(f"Linear Regression - R-squared (R2) Error: {r2_linear:.2f}")

# Decision Tree Regressor Model
tree_model = DecisionTreeRegressor(min_samples_leaf=0.0001)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Evaluating the Decision Tree Regressor Model
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
print(f"Decision Tree Regressor - Mean Squared Error (MSE): {mse_tree:.2f}")
print(f"Decision Tree Regressor - R-squared (R2) Error: {r2_tree:.2f}")

# Grouping and visualizing the most popular cuisines by city
try:
    popular_cuisines_by_city = df.groupby('City')['Cuisines'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    ).dropna()

    # Plotting the top 10 popular cuisines by city
    plt.figure(figsize=(12, 6))
    popular_cuisines_by_city.head(10).plot(kind='bar', color='teal')
    plt.title("Top 10 Popular Cuisines by City")
    plt.xlabel("City")
    plt.ylabel("Cuisine")
    plt.xticks(rotation=45)
    plt.show()
except KeyError:
    print("Column 'Cuisines' or 'City' not found in the dataset.")

# Conclusion:
# The Decision Tree Regressor performs well with a very low MSE and high R-squared value.

