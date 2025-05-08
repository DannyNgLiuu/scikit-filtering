import pandas as pd
import numpy as np
import ast
from sklearn.neighbors import NearestNeighbors
from config.database import engine

default_weights = {
    'Region': 0.3,
    'Game Genre': 0.3,
    'Preferred Game Mode': 0.05,
    'Platform': 0.30,
    'Playstyle Tags': 0.05,
    'Skill Level': 0.05,
    'Reputation': 0.2,
    'Reports': 0.2,
    'Friend List Overlap': 0.05,
    'Age': 0.1
}

X_weighted = None
knn = None
df = None
preprocessor = None
encoded_categorical_features = []

numerical_features = ['Reputation', 'Reports', 'Friend List Overlap', 'Age']
categorical_features = ['Region', 'Game Genre', 'Preferred Game Mode', 'Platform', 'Playstyle Tags', 'Skill Level']


def init_model(prep):
    global df, preprocessor, X_weighted, knn, encoded_categorical_features

    preprocessor = prep
    df = pd.read_sql_query("SELECT * FROM user_profiles", engine)

    X = preprocessor.fit_transform(df[numerical_features + categorical_features]).toarray()

    encoded_categorical_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features).tolist()

    dweights = np.array(
        [default_weights[f] for f in numerical_features] +
        [default_weights[f.split('_')[0]] for f in encoded_categorical_features]
    )

    X_weighted = X * dweights

    knn = NearestNeighbors(n_neighbors=len(df) - 1, metric='cosine')
    knn.fit(X_weighted)


def refresh_model():
    init_model(preprocessor)


def apply_custom_weights(preferences):
    pweights = np.array(
        [preferences[feature] for feature in numerical_features] +
        [preferences[feature.split('_')[0]] for feature in encoded_categorical_features]
    )

    if all(v == 0 for v in preferences.values()):
        return X_weighted

    return X_weighted * pweights


def get_recommendations_ml(player_index, num_recommendations=5, weighted_matrix=None):
    if weighted_matrix is None:
        weighted_matrix = X_weighted

    player_data = weighted_matrix[player_index].reshape(1, -1)
    _, indices = knn.kneighbors(player_data)

    similar_indices = indices.flatten()[1:]

    user_id = df.iloc[player_index]['Player ID']
    user_friends = np.array(ast.literal_eval(df[df['Player ID'] == user_id]['Friends List'].values[0]))

    candidate_ids = df.iloc[similar_indices]['Player ID'].values
    new_ids = candidate_ids[~np.isin(candidate_ids, user_friends)]

    result = pd.DataFrame({'Player ID': new_ids[:num_recommendations]})

    return pd.merge(result, df, on='Player ID')[
        ['Player ID', 'Region', 'Game Genre', 'Reputation', 'Reports',
         'Friend List Overlap', 'Preferred Game Mode', 'Platform',
         'Playstyle Tags', 'Skill Level', 'Age', 'Friends List']
    ]
