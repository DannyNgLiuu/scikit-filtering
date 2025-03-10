import numpy as np
import pandas as pd

similarity_matrix = pd.read_csv("similarity_matrix_subset.csv").values
data_set = pd.read_csv("nexus_data.csv").values


def get_similar_users(user_id, top_n):
    print(f"this is the user {user_id}'s row: {data_set[user_id]}")
    most_similar_users = np.argsort(-similarity_matrix[user_id])[1:top_n+1]
    most_similar_users = most_similar_users
    return most_similar_users

def similar_users_threshold(user_id, threshold, top_n):
    #set players with similarity above threshold
    similar_users = [(i, float(similarity_matrix[user_id][i])) for i in range(len(similarity_matrix)) if i != user_id and similarity_matrix[user_id][i] > threshold]
    
    #sort by highest similarity
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[:top_n]
    
    return similar_users


