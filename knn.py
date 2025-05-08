import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("./nexus_data.csv", keep_default_na=False)

categorical_features = ['Region', 'Game Genre', 'Preferred Game Mode', 'Platform', 'Playstyle Tags', 'Skill Level']
numerical_features = ['Reputation', 'Reports', 'Friend List Overlap', 'Age']

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

#features with 0's will not be shown in the frontend for preferences
preferences = {
    'Region': False,
    'Game Genre': False,
    'Preferred Game Mode': False,
    'Platform': False,
    'Playstyle Tags': False,
    'Skill Level': False,
    'Reputation': 0,
    'Reports': 0,
    'Friend List Overlap': 0,
    'Age': 0
}

for key in preferences:
    if preferences[key] is True:
        preferences[key] = 1
    elif preferences[key] is False:
        preferences[key] = 0

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X = preprocessor.fit_transform(df[numerical_features + categorical_features])

#combines categorical features with their data ex. Region_NA, Region_EU, etc
encoded_categorical_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features).tolist()

#combines numerical features with encoded_categorical_features
all_feature_names = numerical_features + encoded_categorical_features

#applies the weights to the features
dweights = np.array(
    [default_weights[feature] for feature in numerical_features] + 
    [default_weights[feature.split('_')[0]] for feature in encoded_categorical_features]
)

pweights = np.array(
    [preferences[feature] for feature in numerical_features] + 
    [preferences[feature.split('_')[0]] for feature in encoded_categorical_features]
)

#ensures it is an array so it can be properly weighted
X = X.toarray()

if any(preferences.values()):
    X_weighted = X * pweights
else:
    X_weighted = X * dweights


#plus 1 more because user x is not included
knn = NearestNeighbors(n_neighbors=20, metric='cosine')
knn.fit(X_weighted)

def get_recommendations_ml(player_id, df, knn_model):

    player_data = X_weighted[player_id].reshape(1, -1)

    #finds similar players using kNN
    _, indices = knn_model.kneighbors(player_data)

    #retrieve recommended players except itself
    similar_indices = indices.flatten()[1:]

    #removed users that the person is friends with already
    user_friends = df.loc[df['Player ID'] == player_id, 'Friends List'].iloc[0]
    user_friends = np.array(ast.literal_eval(user_friends))
    new_friends = similar_indices[~np.isin(similar_indices, user_friends)]

    recommendations = pd.DataFrame({
        #returns top 5 users
        'Player ID': df.iloc[new_friends[:5]]['Player ID'].values,
    })

    #gets each users information
    full_recommendations = pd.merge(recommendations, df, on='Player ID')
    
    return full_recommendations[['Player ID', 'Region', 'Game Genre', 'Reputation', 'Reports', 
                                 'Friend List Overlap', 'Preferred Game Mode', 'Platform', 'Playstyle Tags', 'Skill Level', 'Age', 'Friends List']]

user_id = 1
recommendations = get_recommendations_ml(user_id, df, knn)

print(df[df['Player ID'] == user_id].to_string(index=False))

print(recommendations.to_string(index=False))


