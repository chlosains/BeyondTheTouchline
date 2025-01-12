import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Load fixtures
def load_fixtures(file_path):
    fixtures = pd.read_csv(file_path)
    fixtures['Date'] = pd.to_datetime(fixtures['Date'], errors='coerce')  # Ensure 'Date' is datetime
    return fixtures


def calculate_recent_form_streak(fixtures, result_column, n=5):
    """
    Calculate each team's recent form as a win streak over the last n games,
    combining both home and away results.
    """
    # Combine home and away results into a long format
    home_results = fixtures[['Date', 'Home', result_column]].copy()
    home_results.columns = ['Date', 'Team', 'Result']
    home_results['Win'] = home_results['Result'].apply(lambda x: 1 if x == 'Home win' else 0)

    away_results = fixtures[['Date', 'Away', result_column]].copy()
    away_results.columns = ['Date', 'Team', 'Result']
    away_results['Win'] = away_results['Result'].apply(lambda x: 1 if x == 'Away win' else 0)

    # Combine and sort by date
    combined_results = pd.concat([home_results, away_results]).sort_values(by='Date').reset_index(drop=True)

    # 🛠️ FIX: Reset index after rolling to avoid duplicates
    combined_results['Recent_form_streak'] = (
        combined_results.groupby('Team')['Win']
        .rolling(window=n, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)  # Reset only the groupby level
    )

    # Merge back to fixtures for both Home and Away teams
    fixtures = fixtures.merge(
        combined_results[['Date', 'Team', 'Recent_form_streak']],
        left_on=['Date', 'Home'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={'Recent_form_streak': 'Home_recent_form'}).drop(columns=['Team'])

    fixtures = fixtures.merge(
        combined_results[['Date', 'Team', 'Recent_form_streak']],
        left_on=['Date', 'Away'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={'Recent_form_streak': 'Away_recent_form'}).drop(columns=['Team'])

    return fixtures



def preprocess_data(fixtures):
    """
    Minimal cleaning of the fixtures data and handling NaN values.
    """
    # Ensure required columns exist
    required_columns = [
        'home_rolling_avg_goals', 'away_rolling_avg_goals',
        'home_rolling_avg_xG', 'away_rolling_avg_xG',
        'xG', 'xG.1', 'home_goals', 'away_goals', 'result'
    ]
    missing_columns = [col for col in required_columns if col not in fixtures.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Apply the streak feature BEFORE one-hot encoding
    fixtures = calculate_recent_form_streak(fixtures, 'result', n=5)

    # One-hot encode categorical variables
    fixtures = pd.get_dummies(fixtures, columns=['Day', 'Venue', 'Home', 'Away', 'Referee'], drop_first=True)
    
    # Fill NaN values for unplayed fixtures
    fixtures['xG'] = fixtures['xG'].fillna(0.0)
    fixtures['xG.1'] = fixtures['xG.1'].fillna(0.0)
    fixtures['home_goals'] = fixtures['home_goals'].fillna(0)
    fixtures['away_goals'] = fixtures['away_goals'].fillna(0)

    return fixtures


# Train the model with automated class weight tuning
def train_model(data):
    """
    Train a Random Forest model using historical data with automated class weight tuning.
    """
    # Exclude features not known before the match
    excluded_features = ['Date', 'Score', 'result', 'Home', 'Away', 'Venue', 'Referee',
                         'season_start', 'home_goals', 'away_goals', 'xG', 'xG.1']
    
    # Define features for training
    features = [col for col in data.columns if col not in excluded_features]
    print(f"Features used for training: {features}")

    # Filter out future fixtures (unplayed matches)
    data = data[(data['Date'] < pd.Timestamp.now()) & (data['result'].notna())]
    
    # Split data chronologically
    split_date = pd.Timestamp("2023-08-01")
    train_data = data[data['Date'] < split_date]
    test_data = data[data['Date'] >= split_date]
    
    X_train = train_data[features]
    y_train = train_data['result']
    X_test = test_data[features]
    y_test = test_data['result']

    # 🔍 Automated Class Weight Tuning
    weight_options = [
        {'Home win': 1, 'Away win': 1, 'Draw': w} for w in [3, 5, 7, 10, 15]
    ]

    # Define Random Forest Classifier
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Add class_weight to the grid search parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'class_weight': weight_options
    }

    # Grid search to find the best class weight
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"🔍 Best class weight found: {grid_search.best_params_['class_weight']}")

    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy: {accuracy * 100:.2f}%")

    # Evaluate model performance in detail
    evaluate_model(y_test, y_pred)

    return best_model


def predict_fixtures(fixtures, model):
    """
    Predict match outcomes using the trained model.
    """
    # Exclude features that shouldn't be used for prediction
    excluded_features = ['Date', 'Score', 'result', 'Home', 'Away', 'Venue', 'Referee',
                         'season_start', 'home_goals', 'away_goals', 'xG', 'xG.1']

    # Extract features used for prediction
    features = [col for col in fixtures.columns if col not in excluded_features]
    print(f"Features used for prediction: {features}")

    # Predict outcomes
    fixtures = fixtures.copy()
    fixtures['predicted_result'] = model.predict(fixtures[features])

    # Decode Day, Venue, and Referee
    for col in ['Day', 'Venue', 'Referee']:
        encoded_cols = [c for c in fixtures.columns if c.startswith(f"{col}_")]
        if encoded_cols:
            fixtures[col] = fixtures[encoded_cols].idxmax(axis=1).str.replace(f"{col}_", "")

    # Decode Home and Away teams (EXCLUDING recent form columns)
    home_team_cols = [c for c in fixtures.columns if c.startswith('Home_') and 'recent_form' not in c]
    away_team_cols = [c for c in fixtures.columns if c.startswith('Away_') and 'recent_form' not in c]

    if home_team_cols:
        fixtures['Home'] = fixtures[home_team_cols].idxmax(axis=1).str.replace('Home_', "")

    if away_team_cols:
        fixtures['Away'] = fixtures[away_team_cols].idxmax(axis=1).str.replace('Away_', "")

    return fixtures[['Wk', 'Day', 'Date', 'Home', 'Away', 'Venue', 'Referee', 'predicted_result']]


def evaluate_model(y_test, y_pred):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Home win', 'Draw', 'Away win'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Home win', 'Draw', 'Away win'])
    
    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=['Home win', 'Draw', 'Away win']))


# Save model
def save_model(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)

# Load model
def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots the top_n feature importances for the trained model.
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.gca().invert_yaxis()  # Highest on top
    plt.tight_layout()
    plt.show()


def load_referee_assignments(file_path):
    """
    Load referee assignments and ensure date formatting matches fixtures.
    """
    referees = pd.read_csv(file_path)
    referees['Date'] = pd.to_datetime(referees['Date'], format='%d/%m/%Y', errors='coerce')
    return referees

def merge_referee_data(fixtures, referee_data):
    """
    Merge referee assignments with fixtures data.
    """
    fixtures = fixtures.merge(
        referee_data,
        on=['Date', 'Home', 'Away'],
        how='left',
        suffixes=('', '_assigned')
    )

    # Fill missing referees with 'Unknown'
    fixtures['Referee'] = fixtures['Referee_assigned'].fillna('Unknown')
    fixtures.drop(columns=['Referee_assigned'], inplace=True)

    return fixtures


def main():
    print("🔍 Starting the script...")  # Debug

    # Hardcoded file paths for Spyder
    fixtures_path = "wsl_all_fixtures.csv"  # Your CSV file path
    referee_path = "referee_assignments.csv"  
    model_path = None  # Set to a model file if loading one
    output_path = "predictions_week_X.csv"  # Where to save predictions


    print("📂 Loading data...")
    # Load data
    fixtures = load_fixtures(fixtures_path)
    print(f"✅ Data loaded. Shape: {fixtures.shape}")
    
    # Load referee assignments
    print("📂 Loading referee assignments...")
    referee_path = "referee_assignments.csv"  
    referee_data = load_referee_assignments(referee_path)
    fixtures = merge_referee_data(fixtures, referee_data)
    print("✅ Referee assignments merged.")

    # Preprocess
    print("🧹 Preprocessing data...")
    fixtures = preprocess_data(fixtures)
    print("✅ Preprocessing done.")

    # Identify the next matchweek
    today = pd.Timestamp.now()
    current_season_start = pd.Timestamp("2024-09-01")  # Adjust this to the start of the current season

    print(f"📊 Total fixtures before filtering: {fixtures.shape[0]}")
    upcoming_fixtures = fixtures[(fixtures['Date'] >= today) & (fixtures['Date'] >= current_season_start)]
    print(f"📊 Future fixtures in the current season: {upcoming_fixtures.shape[0]}")

    next_matchweek = upcoming_fixtures['Wk'].min()
    print(f"📅 Next matchweek identified: {next_matchweek}")
    prediction_fixtures = upcoming_fixtures[upcoming_fixtures['Wk'] == next_matchweek]
    print(f"📅 Fixtures for next matchweek: {prediction_fixtures.shape[0]}")


    training_data = fixtures[fixtures['Date'] < today]
    training_data = training_data.dropna(subset=['result'])
    print(f"📚 Training data after removing NaNs in 'result': {training_data.shape[0]}")
    
    print(training_data['result'].value_counts(normalize=True))
    print(training_data['result'].unique())


    if prediction_fixtures.empty:
        print("⚠️ No upcoming fixtures to predict. Exiting.")
        return

    # Train the model on historical data
    print("🚀 Training the model...")
    model = train_model(training_data)
    print("✅ Model trained.")

    # Extract feature names used for training
    excluded_features = ['Date', 'Score', 'result', 'Home', 'Away', 'Venue', 'Referee', 'season_start', 'home_goals', 'away_goals', 'xG', 'xG.1']
    feature_names = [col for col in fixtures.columns if col not in excluded_features]

    print("📊 Plotting feature importance...")
    plot_feature_importance(model, feature_names)

    # Predict results for the next matchweek
    print("🔮 Predicting results...")
    predictions = predict_fixtures(prediction_fixtures, model)
    print("✅ Predictions generated.")

    print("🔍 Predicted results distribution:")
    print(predictions['predicted_result'].value_counts(normalize=True))

    # Save predictions
    predictions.to_csv(output_path, index=False)
    print(f"💾 Predictions saved to {output_path}")

if __name__ == "__main__":
    print("🔔 Script is running!")  # Debug print
    main()


#%%