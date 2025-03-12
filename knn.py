import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv("./nexus_data.csv", keep_default_na=False)

categorical_features = ['Region', 'Game Genre', 'Preferred Game Mode', 'Platform', 'Playstyle Tags', 'Skill Level']
numerical_features = ['Toxicity Score', 'Reports', 'Friend List Overlap', 'Age']

feature_weights = {
    'Region': 0.3,
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

#ensures it is an array so it can be properly weighted
X = X.toarray()
X_weighted = X * weights

#plus 1 more because user x is not included
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(X_weighted)


def get_recommendations_ml(player_id, df, knn_model):

    player_data = X_weighted[player_id].reshape(1, -1)

    #finds similar players using kNN
    distance, indices = knn_model.kneighbors(player_data)

    #retrieve recommended players except itself
    similar_indices = indices.flatten()[1:]

    recommendations = pd.DataFrame({
        'Player ID': df.iloc[similar_indices]['Player ID'].values,

    })

    #gets each users information
    full_recommendations = pd.merge(recommendations, df, on='Player ID')
    
    return full_recommendations[['Player ID', 'Region', 'Game Genre', 'Platform', 'Playstyle Tags', 
                                 'Preferred Game Mode', 'Skill Level', 'Toxicity Score', 'Reports', 'Age']]

example_player_id = 1
recommendations = get_recommendations_ml(example_player_id, df, knn)
print(recommendations.to_string(index=False))

