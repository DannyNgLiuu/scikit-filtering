import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("./nexus_data.csv")

df["Region"] = df["Region"].fillna("")

#weight of each attribute
df["Enhanced Game Genre"] = (df["Game Genre"] + " ") * 3
df["Enhanced Playstyle"] = (df["Playstyle Tags"] + " ") * 1
df["Enhanced Platform"] = (df["Platform"] + " ") * 5
df["Enhanced PGM"] = (df["Preferred Game Mode"] + " ") * 2
df["Enhanced Region"] = (df["Region"] + " ") * 4

#content based filtering is good for text data
vectorizer = TfidfVectorizer()
game_genre_matrix = vectorizer.fit_transform(df["Enhanced Game Genre"] + " " + df["Enhanced Playstyle"] + " " + df["Enhanced Platform"] + " " + df["Enhanced PGM"] + " " + df["Enhanced Region"])

#TF-IDF computation    #Term Frequency (TF) increases    #Inverse Document Frequency (IDF)
similarity_matrix = cosine_similarity(game_genre_matrix)


print(similarity_matrix[:10, :10])

num_rows = df.shape[0]

similarity_subset = similarity_matrix[:num_rows, :num_rows]
df_subset = pd.DataFrame(similarity_subset).round(2)
similarity_matrix_csv_path = './similarity_matrix_subset.csv'
df_subset.to_csv(similarity_matrix_csv_path, index=False)



numerical_features = ['Toxicity Score', 'Reports', 'Friend List Overlap', 'Age']

scaler = StandardScaler()
numerical_data = scaler.fit_transform(df[numerical_features])




# X = df[["Toxicity Score", "Friend List Overlap", "Reports"]].values

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
# knn.fit(X_scaled)

# #find 5 nearest neighbors for each user
# distances, indices = knn.kneighbors(X_scaled)

# alpha = 0.5
# beta = 0.5 

# def hybrid_recommend(user_index):
    
#     knn_similar_users = indices[user_index]
#     knn_sim_scores = np.exp(-distances[user_index])
#     #knn_sim_scores /= np.sum(knn_sim_scores)
    
#     tfidf_similar_games = np.argsort(-similarity_matrix[user_index])[1:5+1]
#     tfidf_sim_scores = similarity_matrix[user_index][tfidf_similar_games]
#     # tfidf_sim_scores /= np.sum(tfidf_sim_scores)
    
#     hybrid_scores = (alpha * knn_sim_scores) + (beta * tfidf_sim_scores[:len(knn_sim_scores)])
    
#     recommended_users = df.iloc[knn_similar_users]["Player ID"].tolist()
    
#     return recommended_users, hybrid_scores

# user_index = 1

# similar_users, hybrid_scores = hybrid_recommend(user_index)

# recommendations_df = pd.DataFrame({
#     "Recommended Users": similar_users,
#     "Hybrid Scores": hybrid_scores
# })

