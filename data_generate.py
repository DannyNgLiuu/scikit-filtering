import pandas as pd
import numpy as np
import random

np.random.seed(1)

regions = ['NA', 'EU', 'ASIA', 'OCE', 'LATAM', 'AUS', 'MENA', 'KR']
game_genres = ['FPS', 'MOBA', 'Battle Royale', 'MMO', 'RPG', 'Strategy', 'Sports', 'Racing', 'Card Game']
game_modes = ['Competitive', 'Casual', 'Solo', 'Squad', 'Ranked', 'Custom', 'Tournament']
platforms = ['PC', 'PlayStation', 'Xbox', 'Switch', 'Mobile', 'PC/Console']
playstyle_options = ['Aggressive', 'Defensive', 'Support', 'Tactician', 
                    'Camper', 'Solo', 'Team-player', 'Resource-gatherer',
                    'Explorer', 'Completionist', 'Casual', 'Competitive']
skill_levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert', 'Professional']

#number of users
num_rows = 500
data = []

for i in range(0, num_rows):
    player_id = i
    
    region = random.choice(regions)
    game_genre = random.choice(game_genres)
    game_mode = random.choice(game_modes)
    platform = random.choice(platforms)
    reputation = int(np.clip(np.random.normal(450, 200), -100, 1000))
    
    reports = np.random.poisson(max(1, (15 * (1 - (reputation + 100)/1100))))
    friend_overlap = np.random.binomial(30, 0.3)
    playstyle_tags = random.choice(playstyle_options)
    skill_level = random.choice(skill_levels)
    age_value = np.random.normal(25, 7)
    age = int(np.clip(age_value, 13, 65))

    data.append({
        'Player ID': player_id,
        'Region': region,
        'Game Genre': game_genre,
        'Reputation': reputation,
        'Reports': reports,
        'Friend List Overlap': friend_overlap,
        'Preferred Game Mode': game_mode,
        'Platform': platform,
        'Playstyle Tags': playstyle_tags,
        'Skill Level': skill_level,
        'Age': age
    })

df = pd.DataFrame(data)

print(df.head())

df.to_csv('nexus_data.csv', index=False)
