import os
import sys

# Dynamically add the root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from backend.data_utils import load_race_session, get_lap_data
except ModuleNotFoundError as e:
    raise

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def train_lap_time_model(year: int, grand_prix: str, driver: str):
    """
    Trains a Random Forest Regression model to predict lap time for a specific driver.
    Returns the trained model, MAE, and processed DataFrame.
    """
    session = load_race_session(year, grand_prix, 'R')
    if session is None:
        return None, None, None

    laps = get_lap_data(session, driver)
    if laps.empty:
        print("No lap data found.")
        return None, None, None

    # Select and preprocess features
    features = ['LapNumber', 'Stint', 'TyreLife', 'TrackStatus', 'Compound']
    laps = laps.dropna(subset=features + ['LapTime(s)'])
    X = laps[features]
    y = laps['LapTime(s)']

    # One-hot encode the 'Compound' feature
    X = pd.get_dummies(X, columns=['Compound'], drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Lap Time Prediction MAE: {mae:.2f} seconds")
    print(f"R^2 Score: {r2:.2f}")

    # Return model and full dataset for visualization
    return model, mae, laps

def predict_lap_time(model, lap_features: dict):
    """
    Predicts lap time for given input features using a trained model.
    lap_features: dict must include keys like 'LapNumber', 'Stint', etc.
    """
    input_df = pd.DataFrame([lap_features])
    input_df = pd.get_dummies(input_df)

    # Ensure input has all expected features
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    return prediction

def plot_lap_time_predictions(model, df):
    """
    Plots actual vs predicted lap times using the trained model.
    """
    features = ['LapNumber', 'Stint', 'TyreLife', 'TrackStatus', 'Compound']
    df = df.dropna(subset=features + ['LapTime(s)'])
    X = df[features]
    X = pd.get_dummies(X, columns=['Compound'], drop_first=True)
    y = df['LapTime(s)']

    # Align features
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0
    X = X[model.feature_names_in_]

    y_pred = model.predict(X)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df['LapNumber'], y=y, label='Actual', color='blue')
    sns.lineplot(x=df['LapNumber'], y=y_pred, label='Predicted', color='red', linestyle='--')
    plt.xlabel('Lap Number')
    plt.ylabel('Lap Time (s)')
    plt.title('Actual vs Predicted Lap Times')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    year = 2025
    race = 'Monaco'
    driver = 'NOR'

    model, mae, lap_df = train_lap_time_model(year, race, driver)
    if model:
        print(f'Model trained. MAE: {mae:.2f}')
        # Predict a custom lap
        sample_input = {
            'LapNumber': 20,
            'Stint': 2,
            'TyreLife': 8,
            'TrackStatus': 1,
            'Compound': 'Medium'
        }
        prediction = predict_lap_time(model, sample_input)
        print(f'Predicted Lap Time: {prediction:.2f} seconds')

        # Plot predictions
        plot_lap_time_predictions(model, lap_df)
