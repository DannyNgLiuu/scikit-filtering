import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

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












#collaborative filtering is good for numeric data
X = df[["Toxicity Score", "Friend List Overlap", "Voice/Text Chat Activity", "Win/Loss Ratio"]].values

knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(X)

distances, indices = knn.kneighbors([X[0]])


# print("\nIndices of 5 Nearest Neighbors:")
# print(indices)

# print("\nDistances to Nearest Neighbors:")
# print(distances)