#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:01:02 2024

@author: Chloe
"""

# Instructions: 
# Please download the following JSON files and save them in your 'Documents' directory before running this script:
# - matches_England.json
# - matches_Spain.json
# - matches_France.json
# - matches_Germany.json
# - matches_Italy.json
# - players.json
# - events_England.json
# - events_Spain.json
# - events_France.json
# - events_Germany.json
# - events_Italy.json
# You can download the files from the following link: https://figshare.com/collections/Soccer_match_event_dataset/4415000/2


import pandas as pd
import numpy as np
import json
# plotting
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from scipy.stats import binned_statistic_2d
# statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf
#opening data
import os
import pathlib
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

#%%
# Update paths to where your files have been saved
path = os.path.join(str(pathlib.Path().resolve()), 'Documents', 'Python projects', 'Football analytics', 'Soccermatics','Wyscout Data','matches', 'matches_England.json') 
with open(path) as f:
    data = json.load(f)
#save it in a dataframe
df_matches = pd.DataFrame(data)

path = os.path.join(str(pathlib.Path().resolve()), 'Documents', 'Python projects', 'Football analytics', 'Soccermatics','Wyscout Data', 'players.json') 
with open(path) as f:
    data = json.load(f)
#save it in a dataframe
df_players = pd.DataFrame(data)

path = os.path.join(str(pathlib.Path().resolve()), 'Documents', 'Python projects', 'Football analytics', 'Soccermatics', 'Wyscout Data','events', 'events_England.json') 
with open(path) as f: # delete #
    data = json.load(f) # delete #
df_events = pd.DataFrame(data) 

#%%
# Apply transformations to the 'positions' list and create start and end columns
df_events['x_start'] = df_events['positions'].apply(lambda pos: (pos[0]['x']) * 1.05)
df_events['y_start'] = df_events['positions'].apply(lambda pos: pos[0]['y'] * 0.68)
df_events['x_end'] = df_events['positions'].apply(lambda pos: (pos[1]['x']) * 1.05 if len(pos) > 1 else None)
df_events['y_end'] = df_events['positions'].apply(lambda pos: pos[1]['y'] * 0.68 if len(pos) > 1 else None)

#%%
df_players.rename(columns = {'wyId':'playerId'}, inplace = True)

df_players['firstName'] = df_players['firstName'].apply(lambda name: name.encode('latin1').decode('unicode_escape'))
df_players['lastName'] = df_players['lastName'].apply(lambda name: name.encode('latin1').decode('unicode_escape'))
#%%
df_players['currentTeamId'].unique
#%%
corners = df_events[df_events['subEventName'] == 'Corner']
#%%
# Filter players for goalkeepers
df_goalkeepers = df_players[df_players['role'].apply(lambda x: 'GK' in x['code2'] if isinstance(x, dict) else False)]

#%%

def get_corners_and_shots(df, shot_window=8):
    """Calculate corners and check if a shot occurred within a specific window after the corner."""
    all_corners = []

    for match_id in df['matchId'].unique():
        match = df[df['matchId'] == match_id]
        
        for team_id in match['teamId'].unique():
            for period, period_int in zip(['1H', '2H'], [1, 2]):
                
                corners = match[(match['eventName'] == 'Free Kick') & 
                                (match['teamId'] == team_id) & 
                                (match['matchPeriod'] == period) &
                                (match['subEventName'] == 'Corner')]
                
                shots = match[(match['eventName'] == 'Shot') & 
                              (match['teamId'] == team_id) & 
                              (match['matchPeriod'] == period)]
                
                shot_start = shots['eventSec'] - shot_window
                shot_start = shot_start.apply(lambda x: max(x, (period_int - 1) * 45))
                

                # Add 'Shot' column: 1 if a shot occurred within window after corner, 0 otherwise
                corners['Shot'] = corners['eventSec'].apply(
                    lambda x: 1 if any((shot_start < x) & (x < shots['eventSec']).values) else 0
                )
                corners['matchId'] = match_id
                all_corners.append(corners)
    
    all_corners_df = pd.concat(all_corners, ignore_index=True)
    return all_corners_df

def calculate_corner_features(corners):
    """Add additional features for each corner."""
    corners['Accurate'] = corners['tags'].apply(lambda x: 1 if {'id': 1801} in x else 0)
    corners['CornerHeight'] = corners['tags'].apply(
        lambda x: 1 if {'id': 801} in x else (-1 if {'id': 802} in x else 0))
    corners["CornerSide"] = corners["y_start"].apply(lambda y: 0 if y > 34 else 1)
        
    return corners

def add_goalkeeper_height(corners, df_matches, df_goalkeepers, df_players):
    """Add the height of the opposition goalkeeper for each corner event."""
    opposition_gk_heights = []
    
    # Step 1: Create a mapping of match_id to team_ids
    match_teams = {
        row['wyId']: list(row['teamsData'].keys())
        for _, row in df_matches.iterrows()
    }

    for _, corner_row in corners.iterrows():
        match_id = corner_row['matchId']
        player_id = corner_row['playerId']
        
        # Step 2: Determine the player's team_id from players_df
        player_team_id = df_players.loc[df_players['playerId'] == player_id, 'currentTeamId'].values[0]

        # Step 3: Get the opponent's team_id
        # In each match, there are two teams, so find the one that is not the player's team
        team_ids = match_teams.get(match_id, [])
        opp_team_id = next((tid for tid in team_ids if int(tid) != player_team_id), None)
        
        # Step 4: Find the opposition goalkeeper in the lineup and get their height
        if opp_team_id:
            gk = next((p for p in df_matches.loc[df_matches['wyId'] == match_id, 'teamsData'].iloc[0][opp_team_id]['formation']['lineup'] 
                       if p['playerId'] in df_goalkeepers['playerId'].values), None)
            
            if gk:
                # Retrieve the height of the found goalkeeper
                height = df_goalkeepers.loc[df_goalkeepers['playerId'] == gk['playerId'], 'height'].values[0]
                opposition_gk_heights.append({
                    'matchId': match_id,
                    'playerId': player_id,
                    'opposition_gk_height': height
                })
    # Ensure that opposition_gk_heights has unique (matchId, playerId) pairs
    opposition_gk_heights = pd.DataFrame(opposition_gk_heights).drop_duplicates(subset=['matchId', 'playerId'])

    # Merge the opposition goalkeeper height information with the corners DataFrame
    return corners.merge(pd.DataFrame(opposition_gk_heights), on=['matchId', 'playerId'], how='left')


def add_corner_direction(corners, df_players):
    """Add player footedness and determine corner in-swinging or out-swinging direction."""
    
    # Rename 'wyId' to 'playerId' in df_players
    df_players.rename(columns={'wyId': 'playerId'}, inplace=True)
    
    # Merge corners with df_players on 'playerId'
    corners = corners.merge(df_players[['playerId', 'foot']], on='playerId', how='left')
    
    # Fill NaN values in 'foot' column with 'right' (assuming most players are right-footed)
    corners['foot'].fillna('right', inplace=True)
    
    def get_swing_type(row):
        if row['CornerSide'] is None:
            return None
        return 1 if ((row['CornerSide'] == 0 and row['foot'] == 'right') or 
                      (row['CornerSide'] == 1 and row['foot'] == 'left')) else 0
    
    # Determine the SwingType (in-swinging or out-swinging)
    corners['SwingType'] = corners.apply(get_swing_type, axis=1)
        
    # Convert 'CornerSide' from 0 and 1 to 'left' and 'right'
    corners_df['CornerSideName'] = corners_df['CornerSide'].apply(lambda x: 'left' if x == 0 else 'right' if x == 1 else None)
    
    # Map 'left' to 0 and 'right' to 1
    corners_df['CornerSideName'] = corners_df['CornerSideName'].map({'left': 0, 'right': 1})

    return corners


def assign_penalty_box_zone(corners):
    """Assign simplified corner kick target zones based on location."""
    
    def zone_label(row):
        x, y = row['x_end'], row['y_end']
        
        # Define each zone with boundary checks
        if 88.5 <= x <= 105 and 0 <= y < 13.84:
            return '1'
        elif 88.5 <= x <= 105 and 13.84 <= y < 24.84:
            return '2'
        elif 88.5 <= x <= 99.5 and 24.84 <= y < 34:
            return '3'
        elif 88.5 <= x <= 99.5 and 34 <= y < 43.16:
            return '4'
        elif 99.5 <= x <= 105 and 24.84 <= y < 34:
            return '5'
        elif 99.5 <= x <= 105 and 34 <= y < 43.16:
            return '6'
        elif 88.5 <= x <= 105 and 43.16 <= y < 54.16:
            return '7'
        elif 88.5 <= x <= 105 and 54.16 <= y <= 68:
            return '8'
        else:
            return 'Outside'

    # Apply the zone_label function to each row
    corners['Zone'] = corners.apply(zone_label, axis=1)
    return corners
#%%
corners_df = get_corners_and_shots(df_events)

# Calculate additional corner features
corners_df = calculate_corner_features(corners_df)

# Filter out rows where playerId is 0
corners_df = corners_df[corners_df['playerId'] != 0].reset_index(drop=True)

# Add opposition goalkeeper height
corners_df = add_goalkeeper_height(corners_df, df_matches, df_goalkeepers, df_players)

# Add corner direction based on player footedness
corners_df = add_corner_direction(corners_df, df_players)

# Assign penalty box zone based on corner locations
corners_df = assign_penalty_box_zone(corners_df)

# Filter out rows where x_end is between 0 and 50
corners_df = corners_df[(corners_df['x_end'] < 0) | (corners_df['x_end'] > 52.5)]

print(corners_df.head())

#%%

def plot_corners_on_pitch(corners):
    """Plot corner endpoints on an mplsoccer pitch."""
    
    # Create the pitch
    pitch = Pitch(pitch_type='custom', 
              pitch_length=105, pitch_width=68)
    
    # Create the figure and axis
    fig, ax = pitch.draw()
    
    # Extract the coordinates
    x = corners['x_end']
    y = corners['y_end']
    
    # Plot the corner endpoints
    scatter = ax.scatter(x, y, cmap='viridis', alpha=0.6, edgecolor='black', color='red')
    
    plt.title('End points of corners')
    plt.show()

# Call the function with corners dataframe
plot_corners_on_pitch(corners_df)

#%%
def plot_corners_on_pitch(corners):
    """Plot corner start points on an mplsoccer pitch."""
    
    # Create the pitch
    pitch = Pitch(pitch_type='custom', 
              pitch_length=105, pitch_width=68)
    
    # Create the figure and axis
    fig, ax = pitch.draw()
    
    # Extract the coordinates
    x = corners['x_start']
    y = corners['y_start']
    
    # Plot the corner endpoints
    scatter = ax.scatter(x, y, cmap='viridis', alpha=0.6, edgecolor='black')
    
    plt.title('Corner start points on Pitch')
    plt.show()

# Call the function with corners dataframe
plot_corners_on_pitch(corners_df)

#%%
print("X Start Range:", corners_df['x_start'].min(), "-", corners_df['x_start'].max())
print("Y Start Range:", corners_df['y_start'].min(), "-", corners_df['y_start'].max())

print("X end Range:", corners_df['x_end'].min(), "-", corners_df['x_end'].max())
print("Y end Range:", corners_df['y_end'].min(), "-", corners_df['y_end'].max())

#%%

# Define pitch dimensions
pitch_length = 105
pitch_width = 68

# Create the pitch
pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width)
fig, ax = pitch.draw()

# Define the zones with coordinates 
zones = {
    '1': [(88.5, 0), (105, 13.84)],
    '2': [(88.5, 13.84), (105, 24.84)],
    '3': [(88.5, 24.84), (99.5, 34)],
    '4': [(88.5, 34), (99.5, 43.16)],
    '5': [(99.5, 24.84), (105, 34)],
    '6': [(99.5, 34), (105, 43.16)],
    '7': [(88.5, 43.16), (105, 54.16)],    
    '8': [(88.5, 54.16), (105, 68)],   
}
# Define colors for each zone
colors = {
    '1': 'lightblue',
    '2': 'lightgreen',
    '3': 'lightcoral',
    '4': 'lightsalmon',
    '5': 'lightpink',
    '6': 'lightyellow',
    '7': 'lightgray',
    '8': 'lightcyan'
}

# Plot each zone with a different color
for zone_name, ((x_start, y_start), (x_end, y_end)) in zones.items():
    width = x_end - x_start
    height = y_end - y_start
    color = colors[zone_name]  # Get the color for this zone
    ax.add_patch(plt.Rectangle((x_start, y_start), width, height, edgecolor='black', facecolor=color, alpha=0.4))
    ax.text(x_start + width / 2, y_start + height / 2, zone_name, ha='center', va='center', fontsize=8, color='black')

# Display the plot
plt.title('Corner Kick Target Zones')
plt.show()

#%%
#%%
# Implement model 

# Convert categorical variables to dummy variables
corners_df = pd.get_dummies(corners_df, columns=['Zone'], drop_first=True)
#%%
corners_df['Shot'] = corners_df['Shot'].astype(int)  

#Fit the GLM model
model = smf.glm(
    formula='Shot ~ Accurate + CornerHeight + opposition_gk_height + SwingType + CornerSide + Zone_Outside + Zone_2 + Zone_3 + Zone_4 + Zone_5 + Zone_6 + Zone_7 + Zone_8',
    data=corners_df,
    family=sm.families.Binomial()
).fit()

print(model.summary())
#%%
#Fit the GLM model - remove non statistically significant features
model = smf.glm(
    formula='Shot ~ Accurate + Zone_Outside + Zone_2 + Zone_3 + Zone_4 + Zone_5 + Zone_6 + Zone_7 + Zone_8',
    data=corners_df,
    family=sm.families.Binomial()
).fit()

print(model.summary())
#%%
#%%
from sklearn.metrics import roc_auc_score

# True labels
y_true = corners_df['Shot']

# Predicted probabilities from the model
y_pred_proba = model.predict(corners_df)

# Calculate AUC-ROC score
auc_score = roc_auc_score(y_true, y_pred_proba)
print("AUC-ROC Score:", auc_score)
#%%
#%%
print(corners_df.columns)
#%%
#%%
#Calculate shot probability per player
#Shot probability
def calculate_shot_probability(corners_df, model_coeffs):
    # Unpack model coefficients
    intercept = model_coeffs['Intercept']
    accurate_coef = model_coeffs['Accurate']
    
    zone_coefs = {
        'Zone_Outside': model_coeffs['Zone_Outside[T.True]'],
        'Zone_2': model_coeffs['Zone_2[T.True]'],
        'Zone_3': model_coeffs['Zone_3[T.True]'],
        'Zone_4': model_coeffs['Zone_4[T.True]'],
        'Zone_5': model_coeffs['Zone_5[T.True]'],
        'Zone_6': model_coeffs['Zone_6[T.True]'],
        'Zone_7': model_coeffs['Zone_7[T.True]'],
        'Zone_8': model_coeffs['Zone_8[T.True]'],
    }
    
    # Calculate probabilities
    log_odds = (intercept +
                accurate_coef * corners_df['Accurate'])
    
    for zone, coef in zone_coefs.items():
        log_odds += coef * corners_df[zone].astype(int)  # Ensure binary variables are treated as integers
    
    # Convert log-odds to probabilities
    corners_df['Shot_Probability'] = 1 / (1 + np.exp(-log_odds))
    
    return corners_df

# Coefficients from model
model_coeffs = model.params.to_dict()

# Calculate probabilities and add them to corners_df
corners_df_with_prob = calculate_shot_probability(corners_df, model_coeffs)

# Merge with players DataFrame to get player names
merged_df = corners_df_with_prob.merge(df_players, on='playerId', how='left')

# Retain 'endX' and 'endY' for each corner to plot later
merged_df['x_end'] = corners_df_with_prob['x_end']
merged_df['y_end'] = corners_df_with_prob['y_end']

# Count the number of corners taken by each player
corner_counts = merged_df.groupby('playerId').size().reset_index(name='Corner_Count')

# Merge the counts back to the merged DataFrame to filter later
merged_with_counts = merged_df.merge(corner_counts, on='playerId')

# Group by playerId, firstName, lastName and calculate the mean shot probability
player_dangerous_corners = merged_with_counts.groupby(['playerId', 'firstName', 'lastName'])['Shot_Probability'].mean().reset_index()

# Filter to include only players who have taken at least 10 corners
player_dangerous_corners = player_dangerous_corners[player_dangerous_corners['playerId'].isin(merged_with_counts[merged_with_counts['Corner_Count'] >= 10]['playerId'])]

# Sort by highest average shot probability
player_dangerous_corners_sorted = player_dangerous_corners.sort_values(by='Shot_Probability', ascending=False)

# Display the results
print(player_dangerous_corners_sorted)

#%%
# Count the number of corners taken by each player
corner_counts = merged_df.groupby('playerId').size().reset_index(name='Corner_Count')

# Calculate the average corner count
average_corners = corner_counts['Corner_Count'].mean()

# Display the average corner count
print(f"Average number of corners taken per player: {average_corners:.2f}")
#%%
#Apply model to all European leagues
#%%
# Set the base directory path
base_dir = os.path.join(str(pathlib.Path().resolve()), 'Documents', 'Python projects', 'Football analytics', 'Soccermatics','Wyscout Data', 'events')

# Define your list of filenames
files_to_use = ["events_England.json", "events_France.json", "events_Germany.json", "events_Italy.json", "events_Spain.json"]

# Initialize an empty DataFrame to store all events
all_events = pd.DataFrame()

# Loop over each file and load the data
for data_file in files_to_use:
    # Construct the full path
    file_path = os.path.join(base_dir, data_file)
    
    # Open the file and load it into the DataFrame
    with open(file_path) as f:
        data = json.load(f)
        all_events = pd.concat([all_events, pd.DataFrame(data)], ignore_index=True)
  
        
# Apply transformations to the 'positions' list and create start and end columns
all_events['x_start'] = all_events['positions'].apply(lambda pos: (pos[0]['x']) * 1.05)
all_events['y_start'] = all_events['positions'].apply(lambda pos: pos[0]['y'] * 0.68)
all_events['x_end'] = all_events['positions'].apply(lambda pos: (pos[1]['x']) * 1.05 if len(pos) > 1 else None)
all_events['y_end'] = all_events['positions'].apply(lambda pos: pos[1]['y'] * 0.68 if len(pos) > 1 else None)     

euro_corners = all_events[all_events['subEventName'] == 'Corner']
        
#%%
# Set the base directory path
base_dir = os.path.join(str(pathlib.Path().resolve()), 'Documents', 'Python projects', 'Football analytics', 'Soccermatics', 'Wyscout Data', 'matches')

files_to_use = ["matches_England.json", "matches_France.json", "matches_Germany.json", "matches_Italy.json", "matches_Spain.json"]
# Initialize an empty DataFrame to store all events
all_matches = pd.DataFrame()

# Loop over each file and load the data
for data_file in files_to_use:
    # Construct the full path
    file_path = os.path.join(base_dir, data_file)
    
    # Open the file and load it into the DataFrame
    with open(file_path) as f:
        data = json.load(f)
        all_matches = pd.concat([all_matches, pd.DataFrame(data)], ignore_index=True)
        

#%%
def get_corners_and_shots(df, shot_window=8):
    """Calculate corners and check if a shot occurred within a specific window after the corner."""
    all_corners = []

    for match_id in df['matchId'].unique():
        match = df[df['matchId'] == match_id]
        
        for team_id in match['teamId'].unique():
            for period, period_int in zip(['1H', '2H'], [1, 2]):
                
                corners = match[(match['eventName'] == 'Free Kick') & 
                                (match['teamId'] == team_id) & 
                                (match['matchPeriod'] == period) &
                                (match['subEventName'] == 'Corner')]
                
                shots = match[(match['eventName'] == 'Shot') & 
                              (match['teamId'] == team_id) & 
                              (match['matchPeriod'] == period)]
                
                shot_start = shots['eventSec'] - shot_window
                shot_start = shot_start.apply(lambda x: max(x, (period_int - 1) * 45))
                

                # Add 'Shot' column: 1 if a shot occurred within window after corner, 0 otherwise
                corners['Shot'] = corners['eventSec'].apply(
                    lambda x: 1 if any((shot_start < x) & (x < shots['eventSec']).values) else 0
                )
                corners['matchId'] = match_id
                all_corners.append(corners)
    
    all_corners_df = pd.concat(all_corners, ignore_index=True)
    return all_corners_df

def calculate_corner_features(corners):
    """Add additional features for each corner."""
    corners['Accurate'] = corners['tags'].apply(lambda x: 1 if {'id': 1801} in x else 0)
    corners['CornerHeight'] = corners['tags'].apply(
        lambda x: 1 if {'id': 801} in x else (-1 if {'id': 802} in x else 0))
    corners["CornerSide"] = corners["y_start"].apply(lambda y: 0 if y > 34 else 1)
        
    return corners

def add_goalkeeper_height(corners, all_matches, df_goalkeepers, df_players):
    """Add the height of the opposition goalkeeper for each corner event."""
    opposition_gk_heights = []
    
    # Step 1: Create a mapping of match_id to team_ids
    match_teams = {
        row['wyId']: list(row['teamsData'].keys())
        for _, row in df_matches.iterrows()
    }

    for _, corner_row in corners.iterrows():
        match_id = corner_row['matchId']
        player_id = corner_row['playerId']
        
        # Step 2: Determine the player's team_id from players_df
        player_team_id = df_players.loc[df_players['playerId'] == player_id, 'currentTeamId'].values[0]

        # Step 3: Get the opponent's team_id
        # In each match, there are two teams, so find the one that is not the player's team
        team_ids = match_teams.get(match_id, [])
        opp_team_id = next((tid for tid in team_ids if int(tid) != player_team_id), None)
        
        # Step 4: Find the opposition goalkeeper in the lineup and get their height
        if opp_team_id:
            gk = next((p for p in df_matches.loc[df_matches['wyId'] == match_id, 'teamsData'].iloc[0][opp_team_id]['formation']['lineup'] 
                       if p['playerId'] in df_goalkeepers['playerId'].values), None)
            
            if gk:
                # Retrieve the height of the found goalkeeper
                height = df_goalkeepers.loc[df_goalkeepers['playerId'] == gk['playerId'], 'height'].values[0]
                opposition_gk_heights.append({
                    'matchId': match_id,
                    'playerId': player_id,
                    'opposition_gk_height': height
                })
    # Ensure that opposition_gk_heights has unique (matchId, playerId) pairs
    opposition_gk_heights = pd.DataFrame(opposition_gk_heights).drop_duplicates(subset=['matchId', 'playerId'])

    # Merge the opposition goalkeeper height information with the corners DataFrame
    return corners.merge(pd.DataFrame(opposition_gk_heights), on=['matchId', 'playerId'], how='left')


def add_corner_direction(corners, df_players):
    """Add player footedness and determine corner in-swinging or out-swinging direction."""
    
    # Rename 'wyId' to 'playerId' in df_players
    df_players.rename(columns={'wyId': 'playerId'}, inplace=True)
    
    # Merge corners with df_players on 'playerId'
    corners = corners.merge(df_players[['playerId', 'foot']], on='playerId', how='left')
    
    # Fill NaN values in 'foot' column with 'right' (assuming most players are right-footed)
    corners['foot'].fillna('right', inplace=True)
    
    def get_swing_type(row):
        if row['CornerSide'] is None:
            return None
        return 1 if ((row['CornerSide'] == 0 and row['foot'] == 'right') or 
                      (row['CornerSide'] == 1 and row['foot'] == 'left')) else 0
    
    # Determine the SwingType (in-swinging or out-swinging)
    corners['SwingType'] = corners.apply(get_swing_type, axis=1)
        
    # Convert 'CornerSide' from 0 and 1 to 'left' and 'right'
    corners_df['CornerSideName'] = corners_df['CornerSide'].apply(lambda x: 'left' if x == 0 else 'right' if x == 1 else None)
    
    # Map 'left' to 0 and 'right' to 1
    corners_df['CornerSideName'] = corners_df['CornerSideName'].map({'left': 0, 'right': 1})

    return corners


def assign_penalty_box_zone(corners):
    """Assign simplified corner kick target zones based on location."""
    
    def zone_label(row):
        x, y = row['x_end'], row['y_end']
        
        # Define each zone with boundary checks
        if 88.5 <= x <= 105 and 0 <= y < 13.84:
            return '1'
        elif 88.5 <= x <= 105 and 13.84 <= y < 24.84:
            return '2'
        elif 88.5 <= x <= 99.5 and 24.84 <= y < 34:
            return '3'
        elif 88.5 <= x <= 99.5 and 34 <= y < 43.16:
            return '4'
        elif 99.5 <= x <= 105 and 24.84 <= y < 34:
            return '5'
        elif 99.5 <= x <= 105 and 34 <= y < 43.16:
            return '6'
        elif 88.5 <= x <= 105 and 43.16 <= y < 54.16:
            return '7'
        elif 88.5 <= x <= 105 and 54.16 <= y <= 68:
            return '8'
        else:
            return 'Outside'

    # Apply the zone_label function to each row
    corners['Zone'] = corners.apply(zone_label, axis=1)
    return corners

#%%
euro_corners_df = get_corners_and_shots(all_events)
euro_corners_df = calculate_corner_features(euro_corners_df)
euro_corners_df = euro_corners_df[euro_corners_df['playerId'] != 0].reset_index(drop=True)
euro_corners_df = add_goalkeeper_height(euro_corners_df, all_matches, df_goalkeepers, df_players)
euro_corners_df = add_corner_direction(euro_corners_df, df_players)
euro_corners_df = assign_penalty_box_zone(euro_corners_df)
euro_corners_df = euro_corners_df[(euro_corners_df['x_end'] < 0) | (euro_corners_df['x_end'] > 52.5)]

print(euro_corners_df.head())
#%%
#%%
# Convert categorical variables to dummy variables
euro_corners_df = pd.get_dummies(euro_corners_df, columns=['Zone'], drop_first=True)
#%%
#Shot probability
def calculate_shot_probability(corners_df, model_coeffs):
    # Unpack model coefficients
    intercept = model_coeffs['Intercept']
    accurate_coef = model_coeffs['Accurate']
    
    zone_coefs = {
        'Zone_Outside': model_coeffs['Zone_Outside[T.True]'],
        'Zone_2': model_coeffs['Zone_2[T.True]'],
        'Zone_3': model_coeffs['Zone_3[T.True]'],
        'Zone_4': model_coeffs['Zone_4[T.True]'],
        'Zone_5': model_coeffs['Zone_5[T.True]'],
        'Zone_6': model_coeffs['Zone_6[T.True]'],
        'Zone_7': model_coeffs['Zone_7[T.True]'],
        'Zone_8': model_coeffs['Zone_8[T.True]'],
    }
    
    # Calculate probabilities
    log_odds = (intercept +
                accurate_coef * corners_df['Accurate'])
    
    for zone, coef in zone_coefs.items():
        log_odds += coef * corners_df[zone].astype(int)  # Ensure binary variables are treated as integers
    
    # Convert log-odds to probabilities
    corners_df['Shot_Probability'] = 1 / (1 + np.exp(-log_odds))
    
    return corners_df

# Coefficients from model
model_coeffs = model.params.to_dict()

# Calculate probabilities and add them to corners_df
euro_corners_with_prob = calculate_shot_probability(euro_corners_df, model_coeffs)

# Merge with players to get player details
merged_df = euro_corners_with_prob.merge(df_players, on='playerId', how='left')

# Retain 'endX' and 'endY' for each corner to plot later
merged_df['x_end'] = euro_corners_with_prob['x_end']
merged_df['y_end'] = euro_corners_with_prob['y_end']
merged_df['id'] = euro_corners_with_prob['id']

# Count corners per player
corner_counts = merged_df.groupby('playerId').size().reset_index(name='Corner_Count')
merged_with_counts = merged_df.merge(corner_counts, on='playerId')

# Group by player and calculate mean shot probability
player_dangerous_corners = merged_with_counts.groupby(['playerId', 'firstName', 'lastName'])['Shot_Probability'].mean().reset_index()
player_dangerous_corners = player_dangerous_corners[player_dangerous_corners['playerId'].isin(merged_with_counts[merged_with_counts['Corner_Count'] >= 10]['playerId'])]

# Sort by highest shot probability
player_dangerous_corners_sorted = player_dangerous_corners.sort_values(by='Shot_Probability', ascending=False)


# Display the results
print(player_dangerous_corners_sorted)

#%%
#%%
# Plot specific players
specific_player_id = 18591  

# Filter corners_df for corners taken by the specific player
specific_corners_df = euro_corners_with_prob[euro_corners_with_prob['playerId'] == specific_player_id]

# Define pitch dimensions
pitch_length = 105
pitch_width = 68

# Create the pitch
pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width)
fig, ax = pitch.draw()

# Retrieve player name for labeling
player_name = df_players.loc[df_players['playerId'] == specific_player_id, ['firstName', 'lastName']].iloc[0]
label = f"{player_name['firstName']} {player_name['lastName']}"

# Plot corner end points for the specific player
pitch.scatter(specific_corners_df['x_end'], specific_corners_df['y_end'], ax=ax, s=100, edgecolor='black', lw=1,
              label=label, color='purple', alpha=0.7)  # Customize marker size, edge color, and transparency

# Add legend and title
plt.legend(loc='upper left', title="Corners Taken by Player")
plt.title(f"Corner Endpoints for {label}", fontsize=16)
plt.show()

#%%
#%%
#%%
# Plot top players on same graph
top_n = 6

# Select the top players based on shot probability
top_players = player_dangerous_corners_sorted.head(top_n)

# Filter euro_corners_with_prob for corners taken by the top players
top_corners_df = euro_corners_with_prob[euro_corners_with_prob['playerId'].isin(top_players['playerId'])]

# Sort top_corners_df by Shot_Probability in descending order
top_corners_df_sorted = top_corners_df.sort_values(by='Shot_Probability', ascending=False)

# Define pitch dimensions
pitch_length = 105
pitch_width = 68

# Set up subplots: 2 rows, 3 columns for a 3x2 grid
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Adjust figsize to fit all subplots nicely
axes = axes.flatten()  # Flatten to easily iterate over each subplot axis

# Sort top_players by Shot_Probability in descending order (ensure they are sorted correctly)
top_players_sorted = top_players.sort_values(by='Shot_Probability', ascending=False)

# Now loop through the sorted top players to plot in correct order
for ax, player_id in zip(axes, top_players_sorted['playerId']):
    # Get all corners taken by this player
    group = top_corners_df_sorted[top_corners_df_sorted['playerId'] == player_id]

    # Create pitch on each subplot
    pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width)
    pitch.draw(ax=ax)

    # Retrieve player name for labeling
    player_name = df_players.loc[df_players['playerId'] == player_id, ['firstName', 'lastName']].iloc[0]
    label = f"{player_name['firstName']} {player_name['lastName']}"

    # Plot corner end points for this player on the subplot
    pitch.scatter(group['x_end'], group['y_end'], ax=ax, s=100, edgecolor='black', lw=1,
                  label=label, color = 'purple', alpha=0.6)  # 

    # Add title for each subplot
    ax.set_title(f"Corner Endpoints for {label}", fontsize=16)
    ax.legend(loc='upper left', title="Shot Probability")

# Remove empty subplots if top_n < total grid size
for ax in axes[len(top_players_sorted):]:
    ax.axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

#%%
#%%
#Function to look at the shots before corners, filter per player at the end
def get_corners_and_shots(df, shot_window=8, player_ids=None):
    """Calculate corners and shots within a specific window after each corner, including whether shots resulted in goals."""
    all_events = []

    for match_id in df['matchId'].unique():
        match = df[df['matchId'] == match_id]
        
        for team_id in match['teamId'].unique():
            for period, period_int in zip(['1H', '2H'], [1, 2]):
                
                # Filter corners for the team in the current match and period
                corners = match[(match['eventName'] == 'Free Kick') & 
                                (match['teamId'] == team_id) & 
                                (match['matchPeriod'] == period) &
                                (match['subEventName'] == 'Corner')].copy()
                
                # If specific player_ids are provided, filter to include only those players
                if player_ids is not None:
                    corners = corners[corners['playerId'].isin(player_ids)]
                
                # Check if corners is empty after filtering
                if corners.empty:
                    continue  # Skip to the next loop iteration if no corners for specified players
                
                # Filter shots for the team in the current match and period
                shots = match[(match['eventName'] == 'Shot') & 
                              (match['teamId'] == team_id) & 
                              (match['matchPeriod'] == period)].copy()
                
                # Only keep shots that occur within `shot_window` seconds after each corner
                filtered_shots = pd.DataFrame()
                for _, corner in corners.iterrows():
                    valid_shots = shots[(shots['eventSec'] > corner['eventSec']) & 
                                        (shots['eventSec'] <= corner['eventSec'] + shot_window)].copy()
                    filtered_shots = pd.concat([filtered_shots, valid_shots], ignore_index=True)

                # Add 'ShotOccurred' column to corners: 1 if a shot occurred within the window, else 0
                corners['ShotOccurred'] = corners['eventSec'].apply(
                    lambda x: 1 if any((filtered_shots['eventSec'] > x) & 
                                       (filtered_shots['eventSec'] <= x + shot_window)) else 0
                )
                
                # Add additional identifying columns for each event
                corners['matchId'] = match_id
                corners['teamId'] = team_id
                corners['eventType'] = 'Corner'

                # For filtered shots, add relevant columns and mark the event type
                filtered_shots['ShotOccurred'] = 1  # Mark all shots as 1 since they're actual shots
                filtered_shots['matchId'] = match_id
                filtered_shots['teamId'] = team_id
                filtered_shots['eventType'] = 'Shot'

                # Add 'is_goal' column to indicate if each shot is a goal (id 101 in tags)
                filtered_shots['is_goal'] = filtered_shots['tags'].apply(
                    lambda tags: 1 if any(tag['id'] == 101 for tag in tags) else 0
                )

                # Combine corners and filtered shots
                combined_events = pd.concat([corners, filtered_shots], ignore_index=True)
                
                # Append to the list of all events
                all_events.append(combined_events)
    
    # Combine all events across matches into a single DataFrame
    all_events_df = pd.concat(all_events, ignore_index=True)
    return all_events_df


player_ids = [18591] 
euro_shots_corners_df = get_corners_and_shots(all_events, player_ids=player_ids)
print(euro_shots_corners_df.head())

#%%
