from parsing_data import get_similar_users
from parsing_data import similar_users_threshold

user_id = 0

#user_id, num of users
top_matches = get_similar_users(user_id, 10)

#user_id, threshold, num of users
threshold_matches = similar_users_threshold(user_id, 0.6, 10)

print(f"Best matches for User {user_id}: {top_matches}")

print(f"Best matches for User {user_id}: {threshold_matches}")
