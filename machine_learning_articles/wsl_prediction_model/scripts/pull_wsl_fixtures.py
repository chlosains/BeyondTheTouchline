# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import argparse

# Function to scrape fixtures from FBref
def scrape_fixtures(urls):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    dataframes = []

    for url in urls:
        driver.get(url)
        driver.implicitly_wait(10)
        table_element = driver.find_element(By.XPATH, "//table[starts-with(@id, 'sched_')]")
        table_html = table_element.get_attribute('outerHTML')
        df = pd.read_html(table_html)[0]
        dataframes.append(df)

    driver.quit()
    return pd.concat(dataframes, ignore_index=True)


def clean_data(df):
    """
    Clean and preprocess the fixtures data.
    Args:
        df (pandas.DataFrame): Raw fixtures DataFrame.
    Returns:
        pandas.DataFrame: Cleaned and processed DataFrame with added columns.
    """
    # Drop irrelevant columns
    df = df.drop(columns=['Attendance', 'Time', 'Match Report', 'Notes'], errors='ignore')
    
    # Clean other columns
    df['Venue'] = df['Venue'].str.replace("'", "", regex=False)
    df = df[df['Date'] != 'Date']

    # Ensure 'Date' column is properly parsed and drop invalid rows
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['Date'])  # Remove rows where 'Date' could not be parsed

    # Fill missing values
    df['xG'] = df['xG'].fillna(0.0)
    df['xG.1'] = df['xG.1'].fillna(0.0)
    df['Referee'] = df['Referee'].fillna('Unknown')

    # Create 'home_goals' and 'away_goals'
    df[['home_goals', 'away_goals']] = df['Score'].str.split('–', expand=True).astype(float).fillna(0).astype(int)

    # Add the 'result' column (only for played matches)
    df['result'] = df.apply(
        lambda row: 'Home win' if row['home_goals'] > row['away_goals']
        else 'Away win' if row['away_goals'] > row['home_goals']
        else 'Draw' if row['Date'] < pd.Timestamp.now() else None,
        axis=1
    )

    return df



def add_result_column(df):
    """
    Add a 'result' column to the dataset based on goals scored.
    Args:
        df (pandas.DataFrame): The DataFrame containing match data.
    Returns:
        pandas.DataFrame: The DataFrame with the 'result' column added.
    """
    df['result'] = df.apply(
        lambda row: 'Home win' if row['home_goals'] > row['away_goals']
        else 'Away win' if row['away_goals'] > row['home_goals']
        else 'Draw',
        axis=1
    )
    return df

# Function to calculate rolling averages
def calculate_features(df):
    for team in df['Home'].unique():
        temp_df = df[(df['Home'] == team) | (df['Away'] == team)].sort_values('Date').copy()
        
        # Calculate rolling averages for home matches
        temp_df['home_goals_rolling'] = (
            temp_df[temp_df['Home'] == team]['home_goals']
            .shift(1)  # Exclude the current match
            .rolling(window=5, min_periods=1)
            .mean()
        )
        temp_df['home_xG_rolling'] = (
            temp_df[temp_df['Home'] == team]['xG']
            .shift(1)  # Exclude the current match
            .rolling(window=5, min_periods=1)
            .mean()
        )
        
        # Calculate rolling averages for away matches
        temp_df['away_goals_rolling'] = (
            temp_df[temp_df['Away'] == team]['away_goals']
            .shift(1)  # Exclude the current match
            .rolling(window=5, min_periods=1)
            .mean()
        )
        temp_df['away_xG_rolling'] = (
            temp_df[temp_df['Away'] == team]['xG.1']
            .shift(1)  # Exclude the current match
            .rolling(window=5, min_periods=1)
            .mean()
        )
        
        # Assign the rolling averages back to the main dataframe
        df.loc[temp_df[temp_df['Home'] == team].index, 'home_rolling_avg_goals'] = temp_df['home_goals_rolling']
        df.loc[temp_df[temp_df['Away'] == team].index, 'away_rolling_avg_goals'] = temp_df['away_goals_rolling']
        df.loc[temp_df[temp_df['Home'] == team].index, 'home_rolling_avg_xG'] = temp_df['home_xG_rolling']
        df.loc[temp_df[temp_df['Away'] == team].index, 'away_rolling_avg_xG'] = temp_df['away_xG_rolling']

    return df.fillna(0)



# Placeholder function for predictions
def predict_results(df):
    df['predicted_result'] = df.apply(lambda row: 'Home win' if row['home_rolling_avg_xG'] > row['away_rolling_avg_xG'] else 'Away win' if row['home_rolling_avg_xG'] < row['away_rolling_avg_xG'] else 'Draw', axis=1)
    return df

# Function to save predictions
def save_predictions(df, output_path):
    df.to_csv(output_path, index=False)

# Main function for CLI integration
def main():
    parser = argparse.ArgumentParser(description="WSL Prediction Script")
    parser.add_argument('--fixtures', type=str, help="Path to the fixtures file (CSV)", required=False)
    parser.add_argument('--urls', nargs='+', help="List of URLs to scrape fixtures from", required=False)
    parser.add_argument('--output', type=str, help="Path to save the predictions", required=True)
    args = parser.parse_args()

    if args.fixtures:
        fixtures_df = pd.read_csv(args.fixtures)
    elif args.urls:
        fixtures_df = scrape_fixtures(args.urls)
    else:
        raise ValueError("Either --fixtures or --urls must be provided.")

    cleaned_df = clean_data(fixtures_df)
    features_df = calculate_features(cleaned_df)
    predictions_df = predict_results(features_df)
    save_predictions(predictions_df, args.output)

if __name__ == "__main__":
    # List of URLs for all historical seasons
    urls = [
        'https://fbref.com/en/comps/189/2020-2021/schedule/2020-2021-Womens-Super-League-Scores-and-Fixtures',
        'https://fbref.com/en/comps/189/2021-2022/schedule/2021-2022-Womens-Super-League-Scores-and-Fixtures',
        'https://fbref.com/en/comps/189/2022-2023/schedule/2022-2023-Womens-Super-League-Scores-and-Fixtures',
        'https://fbref.com/en/comps/189/2023-2024/schedule/2023-2024-Womens-Super-League-Scores-and-Fixtures',
        'https://fbref.com/en/comps/189/schedule/Womens-Super-League-Scores-and-Fixtures',  # Current season
    ]


# Scrape raw data
scraped_df = scrape_fixtures(urls)

# Clean and process data
cleaned_df = clean_data(scraped_df)

# Calculate rolling averages and other features
features_df = calculate_features(cleaned_df)

# Identify future fixtures without results
future_fixtures = features_df[features_df['Date'] >= pd.Timestamp.now()]
future_fixtures = future_fixtures[['Wk', 'Day', 'Date', 'Home', 'Away', 'Venue', 'Referee']]

# Ensure future fixtures are not already in the dataset
future_fixtures = future_fixtures[~future_fixtures['Date'].isin(features_df['Date'])]

# Assign rolling averages and append only new rows
for index, row in future_fixtures.iterrows():
    home_team = row['Home']
    away_team = row['Away']

    # Fetch rolling averages for the home and away teams
    row['home_rolling_avg_goals'] = features_df[features_df['Home'] == home_team]['home_rolling_avg_goals'].iloc[-1]
    row['home_rolling_avg_xG'] = features_df[features_df['Home'] == home_team]['home_rolling_avg_xG'].iloc[-1]
    row['away_rolling_avg_goals'] = features_df[features_df['Away'] == away_team]['away_rolling_avg_goals'].iloc[-1]
    row['away_rolling_avg_xG'] = features_df[features_df['Away'] == away_team]['away_rolling_avg_xG'].iloc[-1]

# Append new rows
features_df = pd.concat([features_df, future_fixtures], ignore_index=True)

# Deduplicate after adding new rows
features_df = features_df.drop_duplicates()


# Save the updated dataset
features_df.to_csv('wsl_all_fixtures.csv', index=False)
print("Processed fixtures saved to wsl_all_fixtures.csv")


#%%


