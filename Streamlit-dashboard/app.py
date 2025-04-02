import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def load_race_data(year, race):
    session = fastf1.get_session(year, race, "R")
    session.load()
    return session

def preprocess_lap_data(session):
    laps = session.laps
    laps = laps[['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'TrackStatus']]
    laps = laps.dropna()
    laps['LapTime(s)'] = laps['LapTime'].dt.total_seconds()
    laps = laps.drop('LapTime', axis=1)  
    return laps

def train_lap_time_model(lap_data):
    lap_data = pd.get_dummies(lap_data, columns=['Compound', 'TrackStatus'])
    X = lap_data.drop(columns=['Driver', 'LapTime(s)'])
    y = lap_data['LapTime(s)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    return model, error

def plot_lap_times(lap_data, selected_drivers):
    fig, ax = plt.subplots(figsize=(12, 6))
    for driver in selected_drivers:
        driver_laps = lap_data[lap_data['Driver'] == driver]
        ax.plot(driver_laps['LapNumber'], driver_laps['LapTime(s)'], label=driver, marker='o')
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Lap Time (s)')
    ax.set_title('Lap Times Comparison')
    ax.legend()
    ax.grid(True)
    return fig

def plot_tyre_life(lap_data, selected_drivers):
    fig, ax = plt.subplots(figsize=(12, 6))
    for driver in selected_drivers:
        driver_laps = lap_data[lap_data['Driver'] == driver]
        ax.plot(driver_laps['LapNumber'], driver_laps['TyreLife'], label=driver, marker='o')
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Tyre Life')
    ax.set_title('Tyre Life Comparison')
    ax.legend()
    ax.grid(True)
    return fig

def main():
    st.title("RaceWiz AI - F1 Analytics")
    year = st.selectbox("Select Year", list(range(2018, 2025)))
    race = st.text_input("Enter Race Name (e.g., Monza)")

    if st.button("Analyze"):
        with st.spinner("Loading Data..."):
            session = load_race_data(year, race)
            lap_data = preprocess_lap_data(session)
            drivers = lap_data['Driver'].unique().tolist()
            selected_drivers = st.multiselect("Select Drivers for Comparison", drivers, default=drivers[:2])

            lap_model, lap_error = train_lap_time_model(lap_data)

            st.success("Models Trained Successfully!")
            st.write(f"Lap Time Model MAE: {lap_error:.2f} sec")

            st.subheader("Lap Time Data")
            st.dataframe(lap_data.head())

            st.subheader("Lap Times Comparison")
            st.pyplot(plot_lap_times(lap_data, selected_drivers))

            st.subheader("Tyre Life Comparison")
            st.pyplot(plot_tyre_life(lap_data, selected_drivers))

            st.subheader("Lap Time Distribution")
            fig_box, ax_box = plt.subplots()
            sns.boxplot(x='Driver', y='LapTime(s)', data=lap_data, ax=ax_box)
            st.pyplot(fig_box)

            st.subheader("Average Lap Times")
            avg_lap_data = lap_data[lap_data['Driver'].isin(selected_drivers)] 
            avg_lap_times = avg_lap_data.groupby('Driver')['LapTime(s)'].mean().sort_values()
            st.bar_chart(avg_lap_times)

if __name__ == "__main__":
    main()