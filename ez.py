import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv("./nexus_data.csv")
df["Region"] = df["Region"].fillna("")

categorical_features = ['Region', 'Game Genre', 'Preferred Game Mode', 'Platform', 'Playstyle Tags', 'Skill Level']
numerical_features = ['Toxicity Score', 'Reports', 'Friend List Overlap', 'Age']

feature_weights = {
    'Region': 0.35,
    'Game Genre': 0.3,  
    'Preferred Game Mode': 0.05,  
    'Platform': 0.30,
    'Playstyle Tags': 0.05,  
    'Skill Level': 0.05,  
    'Toxicity Score': 0.2,
    'Reports': 0.2,
    'Friend List Overlap': 0.05,  
    'Age': 0.1
}

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X = preprocessor.fit_transform(df[numerical_features + categorical_features])

encoded_categorical_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features).tolist()
all_feature_names = numerical_features + encoded_categorical_features

weights = np.array(
    [feature_weights[feature] for feature in numerical_features] + 
    [feature_weights[feature.split('_')[0]] for feature in encoded_categorical_features]
)

if hasattr(X, "toarray"):  
    X = X.toarray()

weights = weights.flatten()

X_weighted = X * weights

# plus 1 more because user x is not included
model_knn = NearestNeighbors(n_neighbors=50, metric='cosine')
model_knn.fit(X_weighted)


distances, indices = model_knn.kneighbors(X_weighted)

def get_recommendations_ml(player_id, df, knn_model):

    player_index = df[df["Player ID"] == player_id].index[0]
    player_data = X_weighted[player_index].reshape(1, -1)

    #finds similar players using kNN
    distances, indices = knn_model.kneighbors(player_data)
    
    #retrieve recommended player indices (excluding the first, which is the player itself)
    similar_indices = indices.flatten()[1:]
    similar_distances = distances.flatten()[1:]

    max_dist = np.max(similar_distances) if len(similar_distances) > 0 else 1
    similarities = 1 - (similar_distances / max_dist)


    recommendations = pd.DataFrame({
        'Player ID': df.iloc[similar_indices]['Player ID'].values,
        'Similarity': similarities,
        'Platform': df.iloc[similar_indices]['Platform'].values,
        'Region': df.iloc[similar_indices]['Region'].values
    })

    #gets each users information
    full_recommendations = pd.merge(recommendations, df, on='Player ID', suffixes=('_rec', ''))


    return full_recommendations[['Player ID', 'Region', 'Game Genre', 'Platform', 'Playstyle Tags', 
                                 'Preferred Game Mode', 'Skill Level', 'Toxicity Score', 
                                 'Similarity']]

example_player_id = 0
recommendations = get_recommendations_ml(example_player_id, df, model_knn)
print(recommendations.to_string(index=False))

