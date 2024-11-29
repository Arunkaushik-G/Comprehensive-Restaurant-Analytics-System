# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import KMeans
import numpy as np

# Loading the dataset
df = pd.read_csv("C:\\Users\\New\\Desktop\\Projects\\Intern_Project\\projectdata.csv")

# Grouping by the 'City' column to analyze restaurant distribution
restaurants_by_city = df.groupby('City')['Restaurant Name'].count()

# Visualizing the distribution of restaurants by city/local area
plt.figure(figsize=(12, 8))
plt.bar(restaurants_by_city.index, restaurants_by_city.values)
plt.xlabel('City/Local Area')
plt.ylabel('Number of Restaurants')
plt.title('Restaurant Concentration by City/Local Area')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# K-means clustering based on geographical coordinates
X = df[['Latitude', 'Longitude']]
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Plotting the clusters on a scatter plot
plt.figure(figsize=(10, 8))
colors = ['r', 'g', 'b', 'c', 'm']
for cluster_num in range(k):
    cluster_data = df[df['Cluster'] == cluster_num]
    plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], 
                c=colors[cluster_num], label=f'Cluster {cluster_num}')
    
# Adding cluster centroids to the plot
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], 
            s=200, c='black', label='Centroids')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('K-means Clustering of Restaurants')
plt.legend()
plt.tight_layout()
plt.show()

# Analyzing average ratings by city/local area
average_ratings_by_city = df.groupby('City')['Aggregate rating'].mean()
plt.figure(figsize=(12, 8))
plt.bar(average_ratings_by_city.index, average_ratings_by_city.values)
plt.xlabel('City/Local Area')
plt.ylabel('Average Rating')
plt.title('Average Ratings by City/Local Area')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Determining the most popular cuisines by city/local area
#popular_cuisines_by_city = df.groupby('City')['Cuisines'].agg(lambda x: x.mode().iloc[0])
# Grouping by 'City' to find the most popular cuisine in each city
'''popular_cuisines_by_city = df.groupby('City')['Cuisines'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
)'''
popular_cuisines_by_city = df.groupby('City')['Cuisines'].agg(lambda x: x.mode().first_valid_index())

print(popular_cuisines_by_city.head())

plt.figure(figsize=(12, 8))
popular_cuisines_by_city = pd.DataFrame({'City': popular_cuisines_by_city.index, 'Popular Cuisines': popular_cuisines_by_city.values})
plt.bar(popular_cuisines_by_city['City'], popular_cuisines_by_city['Popular Cuisines'])
plt.xlabel('City/Local Area')
plt.ylabel('Popular Cuisines')
plt.title('Popular Cuisines by City/Local Area')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Analyzing the most common price range by city/local area
common_price_range_by_city = df.groupby('City')['Price range'].agg(lambda x: x.mode().iloc[0])
plt.figure(figsize=(12, 8))
plt.bar(common_price_range_by_city.index, common_price_range_by_city.values)
plt.xlabel('City/Local Area')
plt.ylabel('Common Price Range')
plt.title('Common Price Range by City/Local Area')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Analyzing cuisine diversity by city/local area
cuisine_diversity_by_city = df.groupby('City')['Cuisines'].apply(lambda x: len(set(x)))
plt.figure(figsize=(12, 8))
plt.bar(cuisine_diversity_by_city.index, cuisine_diversity_by_city.values)
plt.xlabel('City/Local Area')
plt.ylabel('Cuisine Diversity (Number of Unique Cuisines)')
plt.title('Cuisine Diversity by City/Local Area')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

### Conclusion:
# The geographical analysis revealed patterns in restaurant distribution, average ratings, popular cuisines, common price ranges, and cuisine diversity across different cities or local areas.
