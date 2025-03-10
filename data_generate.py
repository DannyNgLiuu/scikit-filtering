import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(1)

# Define sample values for categorical variables
regions = ['NA', 'EU', 'ASIA', 'OCE', 'LATAM', 'CIS', 'MENA', 'SEA']
game_genres = ['FPS', 'MOBA', 'Battle Royale', 'MMO', 'RPG', 'Strategy', 'Sports', 'Racing', 'Card Game']
game_modes = ['Competitive', 'Casual', 'Solo', 'Squad', 'Ranked', 'Custom', 'Tournament']
platforms = ['PC', 'PlayStation', 'Xbox', 'Switch', 'Mobile', 'PC/Console']
playstyle_options = ['Aggressive', 'Defensive', 'Support', 'Sniper', 'Rusher', 'Tactician', 
                    'Camper', 'Objective-focused', 'Solo', 'Team-player', 'Resource-gatherer',
                    'Explorer', 'Achievement-hunter', 'Completionist', 'Casual', 'Competitive']
skill_levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert', 'Professional']

num_rows = 500
data = []

for i in range(0, num_rows):
    player_id = i
    
    region = random.choice(regions)
    game_genre = random.choice(game_genres)
    game_mode = random.choice(game_modes)
    platform = random.choice(platforms)
    toxicity_score = round(np.random.beta(1, 4), 2)
    reports = np.random.poisson((toxicity_score * 15 + 1))
    friend_overlap = np.random.binomial(30, 0.3)
    playstyle_tags = random.choice(playstyle_options)
    skill_level = random.choice(skill_levels)
    age_value = np.random.normal(25, 7)
    age = int(np.clip(age_value, 13, 65)) 

    data.append({
        'Player ID': player_id,
        'Region': region,
        'Game Genre': game_genre,
        'Toxicity Score': toxicity_score,
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
