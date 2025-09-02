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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def prepare_degradation_data(year: int, grand_prix: str):
    """
    Prepares the dataset for tire degradation modeling using all drivers.
    Returns lap-level data.
    """
    session = load_race_session(year, grand_prix, 'R')
    if session is None:
        return pd.DataFrame()

    laps = get_lap_data(session)
    if laps.empty:
        return pd.DataFrame()

    # Clean and filter laps
    laps = laps.dropna(subset=['Compound', 'TyreLife', 'LapTime(s)', 'TrackStatus'])
    laps = laps[laps['TrackStatus'] == 1]  # Only green flag laps

    return laps


def train_tire_degradation_model(df, compound='Medium'):
    """
    Trains a regression model to estimate lap time degradation for a given compound.
    """
    compound_df = df[df['Compound'] == compound]
    if compound_df.empty:
        print(f"No data for compound: {compound}")
        return None, None

    # Features
    X = compound_df[['TyreLife', 'Stint', 'LapNumber']]
    y = compound_df['LapTime(s)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Compound: {compound}")
    print(f"MSE: {mse:.2f}, R^2 Score: {r2:.2f}")

    return model, compound_df


def predict_degradation(model, tyre_life: int, stint: int, lap_number: int):
    """
    Predict lap time based on tire age and additional context.
    """
    input_df = pd.DataFrame([[tyre_life, stint, lap_number]], columns=['TyreLife', 'Stint', 'LapNumber'])
    return model.predict(input_df)[0]


def plot_degradation_curve(model, df):
    """
    Plot degradation curve (lap time vs. tyre age).
    """
    df = df.sort_values(by='TyreLife')
    X = df[['TyreLife', 'Stint', 'LapNumber']]
    y = df['LapTime(s)']

    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.plot(df['TyreLife'], y, label='Actual', color='blue')
    plt.plot(df['TyreLife'], y_pred, label='Predicted', color='red', linestyle='--')
    plt.xlabel('Tyre Life (laps)')
    plt.ylabel('Lap Time (s)')
    plt.title('Tire Degradation Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import time

if __name__ == '__main__':
    year = 2024
    race = 'Monza'

    df = prepare_degradation_data(year, race)
    for compound in df['Compound'].unique():
        print(f"Training model for compound: {compound}")
        model, compound_df = train_tire_degradation_model(df, compound=compound)
        if model:
            plot_degradation_curve(model, compound_df)
            print("Prediction at TyreLife = 12:", predict_degradation(model, 12, 2, 35))
            time.sleep(10)  # <-- ⏳ this holds the plot window open for 10 seconds
