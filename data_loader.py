import fastf1
from preprocessing import engineer_features
import pandas as pd
import os 

#cache enabling
fastf1.Cache.enable_cache('./data/raw') 


#function for load data
def load_f1_data(year, race_name, session_type='R'):
    
    #load session
    session = fastf1.get_session(year, race_name, session_type)
    session.load()

    #extract laps
    laps = session.laps

    #extract pit stops
    pit_stops = laps[laps['PitOutTime'].notnull()]

    #extract weather data
    weather = session.weather_data

    #display
    print("\n--- Exemple de tours ---")
    print(laps[['Driver', 'LapNumber', 'LapTime', 'Compound', 'Stint']].head())

    print("\n--- Exemple d'arrêts aux stands ---")
    print(pit_stops[['Driver', 'LapNumber', 'PitInTime', 'PitOutTime', 'Compound']].head())
    print(weather)
    missing_values1 = weather.isnull().sum()
    print(missing_values1)
    missing_values2 = laps.isnull().sum()
    print(missing_values2)
    print(len(laps))
    return laps, weather, pit_stops,session

def load_all_races(races):
    dfs = []
    for race in races:
       for year in range(2021,2025):
         laps, weather,pitStops,session = load_f1_data(year, race, 'R')
         df_features = engineer_features(laps, weather, session,pitStops)
         dfs.append(df_features)

    dfs= pd.concat(dfs, ignore_index=True)
    return dfs
    
