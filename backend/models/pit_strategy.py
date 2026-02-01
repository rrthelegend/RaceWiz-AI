import os
import sys


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
    session = load_race_session(year, grand_prix, 'R')
    if session is None:
        return None, None, None

    laps = get_lap_data(session, driver)
    if laps.empty:
        print("No lap data found.")
        return None, None, None

    
    features = ['LapNumber', 'Stint', 'TyreLife', 'TrackStatus', 'Compound']
    laps = laps.dropna(subset=features + ['LapTime(s)'])
    X = laps[features]
    y = laps['LapTime(s)']

    
    X = pd.get_dummies(X, columns=['Compound'], drop_first=True)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Lap Time Prediction MAE: {mae:.2f} seconds")
    print(f"R^2 Score: {r2:.2f}")

    
    return model, mae, laps

def predict_lap_time(model, lap_features: dict):

    input_df = pd.DataFrame([lap_features])
    input_df = pd.get_dummies(input_df)

    
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    return prediction

def plot_lap_time_predictions(model, df):
   
    features = ['LapNumber', 'Stint', 'TyreLife', 'TrackStatus', 'Compound']
    df = df.dropna(subset=features + ['LapTime(s)'])
    X = df[features]
    X = pd.get_dummies(X, columns=['Compound'], drop_first=True)
    y = df['LapTime(s)']

    
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0
    X = X[model.feature_names_in_]

    y_pred = model.predict(X)

    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df['LapNumber'], y=y, label='Actual', color='blue')
    sns.lineplot(x=df['LapNumber'], y=y_pred, label='Predicted', color='red', linestyle='--')
    plt.xlabel('Lap Number')
    plt.ylabel('Lap Time (s)')
    plt.title('Actual vs Predicted Lap Times')
    plt.legend()
    plt.tight_layout()
    plt.show()
