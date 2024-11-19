#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:17:30 2024

@author: Chloe

# Beyond the Touchline - Football Analytics
# This script generates Lauren James' carries heatmap based on Statsbomb data.
# Read the full analysis on Substack: https://beyondthetouchline.substack.com/p/the-art-of-the-carry-how-lauren-james?r=3gmn6f

"""

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen
import pandas as pd
from adjustText import adjust_text

# Initialize the parser
parser = Sbopen()

# Step 1: Find the competition and season for the Women's World Cup
df_competitions = parser.competition()

# Women's World Cup (Competition ID = 72), Season 2023 (Season ID = 107)
competition_id = 72
season_id = 107
#%%

# Step 1: Fetch all matches for Women's World Cup (competition_id = 72, season_id = 107)
df_matches = parser.match(competition_id=72, season_id=107)

# Step 2: Fetch events for each match and concatenate them into a single DataFrame
df_events = pd.concat([parser.event(match_id)[0] for match_id in df_matches['match_id']])

#%%
# Initialize a list to collect lineup data
lineup_data = []

# Fetch lineups for each match and store in the list
for match_id in df_matches['match_id']:
    # Fetch the lineup for the match
    lineup = parser.lineup(match_id)
    
    # Add the lineup data to the list (can convert to DataFrame if needed)
    # Each lineup fetched will be a DataFrame
    if isinstance(lineup, pd.DataFrame):
        lineup_data.append(lineup)

# Concatenate all lineup data into a single DataFrame
df_lineups = pd.concat(lineup_data, ignore_index=True)


#%%
wingers = df_events[df_events['position_name'].isin(['Left Wing', 'Right Wing'])]

forwards = df_events[df_events['position_name'].isin(['Left Wing', 'Right Wing', 'Left Center Forward','Right Center Forward','Center Forward', 'Right Attacking Midfield', 'Left Attacking Midfield'])]

#%%


#%%
# Filter for events involving carries by Left Wing and Right Wing players
forwards_carries = forwards[(forwards['type_name'] == 'Carry')]

# Create a DataFrame of shot events
shots = df_events[df_events['type_name'] == 'Shot']

# Merge the carries with the shots on the same match ID
merged = pd.merge(forwards_carries, shots, on='match_id', suffixes=('_carry', '_shot'))

# Calculate time in seconds for both carries and shots
merged['carry_time'] = merged['minute_carry'] * 60 + merged['second_carry']
merged['shot_time'] = merged['minute_shot'] * 60 + merged['second_shot']

# Calculate the time difference in seconds
merged['time_diff'] = merged['shot_time'] - merged['carry_time']

# Filter to keep only the pairs where the shot occurred within 8 seconds after the carry
# AND the shot is made by the same team as the carry
carries_leading_to_shots = merged[
    (merged['time_diff'] <= 8) & 
    (merged['team_id_carry'] == merged['team_id_shot'])  # Ensure the same team
]

# Drop duplicates based on carry event information (keeping one entry per carry)
unique_carries = carries_leading_to_shots.drop_duplicates(subset=['id_carry'])  

# Number of unique players in the dataset
num_players = forwards['player_id'].nunique()

# Create the pitch
pitch = Pitch(pitch_type='statsbomb', line_color='black', line_zorder=2)

# Calculate the bin statistic for the unique carries leading to shots
bin_statistic_avg = pitch.bin_statistic(unique_carries['x_carry'], unique_carries['y_carry'], 
                                        statistic='count', bins=(6, 5))

# Calculate the average carries leading to shots per player
average_statistic = bin_statistic_avg['statistic'] / num_players


# Filter Lauren James' carries leading to shots
james_carries_leading_to_shots = unique_carries[unique_carries['player_name_carry'] == 'Lauren James']

# Calculate the bin statistic for Lauren James' carries leading to shots
bin_statistic_james = pitch.bin_statistic(james_carries_leading_to_shots['x_carry'], 
                                          james_carries_leading_to_shots['y_carry'], 
                                          statistic='count', bins=(6, 5))

# Ensure the lengths of the statistics are consistent
if len(bin_statistic_james['statistic']) != len(average_statistic):
    print("Mismatch in lengths of bin statistics.")

# Subtract the average from Lauren James' carries to see if she's above or below average in each grid
comparison_statistic = bin_statistic_james['statistic'] - average_statistic

# Create the heatmap plot
fig, ax = plt.subplots(figsize=(10, 6))
pitch.draw(ax=ax)

# Here we need to create a new structure for the comparison statistic to plot correctly
comparison_statistic_grid = np.zeros_like(bin_statistic_james['statistic'])  # Start with zeros
comparison_statistic_grid[:] = comparison_statistic  # Fill with comparison values

# Plot the heatmap for Lauren James compared to the average player
heatmap = pitch.heatmap({
    'x_grid': bin_statistic_james['x_grid'],
    'y_grid': bin_statistic_james['y_grid'],
    'statistic': comparison_statistic_grid
}, ax=ax, cmap='Blues', edgecolors='#22312b')

# Add a color bar to visualize the scale (-ve values indicate below average, +ve above average)
cbar = plt.colorbar(heatmap, ax=ax)
cbar.set_label("James' Carries Leading to Shots\n(Lighter = Below Avg, Darker = Above Avg)")

# Scatter plot for Lauren James' carries leading to shots
ax.scatter(james_carries_leading_to_shots['x_carry'], james_carries_leading_to_shots['y_carry'], 
           color='black', s=12, label="Lauren James' Carries", zorder=3)

# Add legend
ax.legend(loc='lower left')


# Add title
plt.title("Lauren James' Carries Leading to Shots (within 8 seconds)\nCompared to Average Forward", fontsize=10, fontfamily='monospace')


# Show the plot
plt.show()


#%%
# 1. Filter for carries
carries = wingers[wingers['type_name'] == 'Carry']

# 2. Count the number of carries for each player
carry_count = carries['player_name'].value_counts().reset_index()
carry_count.columns = ['player_name', 'carry_count']

# 3. Calculate the 90th percentile (top quartile) carry count
top_quartile_carry_count = carry_count['carry_count'].quantile(0.9)

print(f"Top quartile carry count (90th percentile): {top_quartile_carry_count:.2f}")


#%%

#%%
# 1. Filter for carries and dispossessions
carries = wingers[wingers['type_name'] == 'Carry']
dispossessions = wingers[wingers['type_name'] == 'Dispossessed']

# 2. Calculate average duration of carries
average_duration = carries.groupby('player_name')['duration'].mean().reset_index()
average_duration.columns = ['player_name', 'average_carry_duration']

# 3. Count the number of dispossessions
dispossession_count = dispossessions['player_name'].value_counts().reset_index()
dispossession_count.columns = ['player_name', 'dispossession_count']

# 4. Count the number of carries for each player and apply a minimum threshold (e.g., 10 carries)
carry_count = carries['player_name'].value_counts().reset_index()
carry_count.columns = ['player_name', 'carry_count']

# Set the minimum number of carries (e.g., 10)
min_carry_threshold = 82.5
filtered_carry_count = carry_count[carry_count['carry_count'] >= min_carry_threshold]

# Merge the average carry duration and dispossession counts with carry counts
merged_data = pd.merge(average_duration, dispossession_count, on='player_name', how='outer')
merged_data = pd.merge(merged_data, filtered_carry_count, on='player_name', how='inner')  # Keep only players with >= min_carry_threshold

# Merge with lineup data to get player nicknames
merged_data = pd.merge(merged_data, df_lineups[['player_name', 'player_nickname']], on='player_name', how='left')

# 5. Plotting with switched axes (Dispossessions on X-axis, Average Carry Duration on Y-axis)
plt.figure(figsize=(10, 6))

# Plot all players in blue except Lauren James
plt.scatter(merged_data[merged_data['player_name'] != 'Lauren James']['dispossession_count'],
            merged_data[merged_data['player_name'] != 'Lauren James']['average_carry_duration'], 
            color='blue', label="Other Players")

# Plot Lauren James in a different color (e.g., red)
plt.scatter(merged_data[merged_data['player_name'] == 'Lauren James']['dispossession_count'],
            merged_data[merged_data['player_name'] == 'Lauren James']['average_carry_duration'], 
            color='red', label="Lauren James", s=100)

plt.title(f'Number of Times Dispossessed vs Average Duration of Carries', fontfamily='monospace')
plt.xlabel('Number of Times Dispossessed', fontfamily='monospace')
plt.ylabel('Average Carry Duration (seconds)', fontfamily='monospace')
plt.grid(True, linestyle='--', alpha=0.5)

# Horizontal and vertical reference lines
plt.axhline(0, color='black', linewidth=0.2, ls='--')  # Horizontal line at y=0 for reference
plt.axvline(0, color='black', linewidth=0.2, ls='--')  # Vertical line at x=0 for reference

# Add annotations for all players with nicknames
texts = []
for i in range(len(merged_data)):
    nickname = merged_data['player_nickname'].iloc[i]  # Use nickname instead of player name
    texts.append(plt.text(merged_data['dispossession_count'].iloc[i], 
                          merged_data['average_carry_duration'].iloc[i],
                          nickname,  # Update to use nickname
                          fontsize=8, fontfamily='monospace', alpha=0.5))


# Automatically adjust the text positions to avoid overlaps
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5))

# Add explanatory text
plt.text(
    x=0.95, y=0.1, s='Comparing against all wingers in the\n90th percentile for number of carries', fontsize=10,
    ha='right', va='center', transform=plt.gca().transAxes, fontfamily='monospace'
)

plt.tight_layout()

# Show the plot
plt.show()

#%%

df_events['type_name'].unique()
#%%

# Filter for Lauren James (you need to confirm her player_id, for this example I'll use her name)
player_name = 'Lauren James'
df_lauren_james = df_events[df_events['player_name'] == player_name]

# Filter for shots and assists
df_shots = df_lauren_james[df_lauren_james['type_name'] == 'Shot']

# Find the action prior to each shot and assist
# Sort by time and get the action right before each shot and assist
df_events_sorted = df_events.sort_values(by=['match_id', 'minute', 'second'])

# Finding previous actions for shots
df_shots_previous = pd.DataFrame()
for idx, row in df_shots.iterrows():
    previous_action = df_events_sorted[(df_events_sorted['match_id'] == row['match_id']) &
                                       (df_events_sorted['period'] == row['period']) &
                                       (df_events_sorted['minute'] <= row['minute']) &
                                       (df_events_sorted['second'] < row['second'])].tail(1)
    df_shots_previous = pd.concat([df_shots_previous, previous_action])


# Combine shots and assists with previous actions
df_shots_and_previous = pd.concat([df_shots, df_shots_previous], ignore_index=True)




