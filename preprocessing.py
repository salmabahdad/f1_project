import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
def engineer_features(laps_df, weather_df, session, pit_stops):

    df = laps_df.copy()
    
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()

    #  degradation indicator
    df['AvgLapTime_3'] = (
        df.groupby('Driver')['LapTimeSeconds']
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df['AvgLapTime_3'] = df['AvgLapTime_3'].round(3)

    # fusion with weather
    weather_df = weather_df.copy()
    weather_df['Time'] = pd.to_timedelta(weather_df['Time'])

    df = pd.merge_asof(
        df.sort_values('LapStartTime'),
        weather_df.sort_values('Time'),
        left_on='LapStartTime',
        right_on='Time',
        direction='backward'
    )

    # safety car status (TrackStatus == 4 ou 5 → SC ou VSC)
    sc_data = session.track_status
    sc_data = sc_data.copy()
    sc_data['Time'] = pd.to_timedelta(sc_data['Time'])

    def is_safety_car_active(time):
        return any(
            (sc_data['Time'] <= time) &
            (sc_data['Status'].isin([4, 5]))  # 4 = SC, 5 = VSC
        )

    df['IsSafetyCar'] = df['LapStartTime'].apply(is_safety_car_active).astype(int)

   
    df['NextPit'] = np.nan  
    
    for driver in df['Driver'].unique():
      driver_laps = df[df['Driver'] == driver]
      driver_pit_stops = pit_stops[pit_stops['Driver'] == driver]
    
      for i, lap in driver_laps.iterrows():
          next_pit_stop = driver_pit_stops[driver_pit_stops['LapNumber'] > lap['LapNumber']]
        
          if not next_pit_stop.empty:
              next_pit_lap = next_pit_stop.iloc[0]['LapNumber']
              df.loc[i, 'NextPit'] = next_pit_lap - lap['LapNumber']
          else:
            df.loc[i, 'NextPit'] = np.nan  
            
    df['NextPit'].fillna(0,inplace=True)
   
    df['PitNextLap'] = 0  

    for driver in df['Driver'].unique():
    
       driver_laps = df[df['Driver'] == driver].sort_values('LapNumber')
    
    
       pit_laps = pit_stops[pit_stops['Driver'] == driver]['LapNumber'].values

       for i, lap in driver_laps.iterrows():
          if lap['LapNumber'] + 1 in pit_laps:
            df.loc[i, 'PitNextLap'] = 1


    df['GapToCarAhead'] = 0 


    for _, group in df.groupby(['LapNumber']): 
    # Trie par position (la voiture en tête est la 1ère)
      sorted_group = group.sort_values('Position')
      times = sorted_group['LapStartTime'].values

    # Calcul du gap avec la voiture devant (différence des temps de départ des tours)
      gap_ahead = [0] + list(np.diff(times) / np.timedelta64(1, 's'))  # Premier pilote = 0 autres calculs de gap
      gap_behind = list(-np.diff(times) / np.timedelta64(1, 's')) + [0]  # Dernier pilote = 0

    # Affectation des résultats aux colonnes
      df.loc[sorted_group.index, 'GapToCarAhead'] = gap_ahead
      df.loc[sorted_group.index, 'GapToCarBehind'] = gap_behind

   
    columns_to_keep = [
        'Driver', 'LapNumber', "LapStartTime",'Stint', 'Compound',
        'LapTimeSeconds', 'AvgLapTime_3',
        'AirTemp', 'Humidity', 'Rainfall', 'IsSafetyCar',
        'NextPit', 'PitNextLap','Position','TyreLife',"FreshTyre",
        'GapToCarAhead', 'GapToCarBehind'
    ]


    return df[columns_to_keep]



def prepare_data(laps):
    # Colonnes numeriques a standardiser et normaliser
    num_features_std = ['Stint','LapTimeSeconds','AvgLapTime_3','AirTemp','Humidity','TyreLife']
    num_features_minmax = ['Position','GapToCarAhead','GapToCarBehind']

    # Colonnes categorielles à encoder
    cat_features = ['Compound','Rainfall','IsSafetyCar','FreshTyre']

    # Verifie si le CSV pre-traite existe
    if not os.path.exists("F1_ALL_RACES.csv"):
        print("➡️ Fichier introuvable, création et prétraitement...")

        # Nettoyage
        df = laps.dropna(subset=['LapTimeSeconds', 'Compound']).copy()

        # Encodage categoriel avec get_dummies
        df = pd.get_dummies(df, columns=cat_features, drop_first=False)
        # Standardisation des colonnes numeriques
        if num_features_std:
            scaler_std = StandardScaler()
            df[num_features_std] = scaler_std.fit_transform(df[num_features_std])

        # Normalisation MinMax des colonnes numeriques
        if num_features_minmax:
            scaler_minmax = MinMaxScaler()
            df[num_features_minmax] = scaler_minmax.fit_transform(df[num_features_minmax])

        
        df.to_csv("F1_ALL_RACES.csv", index=False)
        print(" Données pré-traitées sauvegardées dans F1_ALL_RACES.csv")

      

    else:
        print(" Fichier trouvé, chargement direct...")
        df = pd.read_csv("F1_ALL_RACES.csv")

  
    X_reg = df.drop(columns=['NextPit','PitNextLap','LapStartTime','Driver'], errors='ignore')
    y_reg = df['NextPit']


    X_class = df.drop(columns=['NextPit','PitNextLap','LapStartTime','Driver'], errors='ignore')
    y_class = df['PitNextLap']

   
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )

    # Appliquer SMOTE
    smote = SMOTE(random_state=42)
    X_class_train, y_class_train = smote.fit_resample(X_class_train, y_class_train)

    return (
        X_reg_train, X_reg_test, y_reg_train, y_reg_test,
        X_class_train, X_class_test, y_class_train, y_class_test
    )


  
