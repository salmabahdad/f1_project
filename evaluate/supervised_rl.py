import argparse
import pandas as pd
import joblib
import json
import numpy as np
from stable_baselines3 import DQN, PPO, A2C

DEFAULT_FEATURE_COLS = [
    'Driver', 'LapNumber', "LapStartTime",'Stint', 'Compound',
        'LapTimeSeconds', 'AvgLapTime_3',
        'AirTemp', 'Humidity', 'Rainfall', 'IsSafetyCar',
        'NextPit', 'PitNextLap','Position','TyreLife',"FreshTyre",
        'GapToCarAhead', 'GapToCarBehind'
]

DEFAULT_LAPTIME_COL = "LapTimeSeconds"


def load_data(csv_path, driver):
    df = pd.read_csv(csv_path)
    df_driver = df[df["Driver"] == driver].copy()
    if df_driver.empty:
        raise ValueError(f"Aucune donnée trouvée pour le driver {driver}")
    return df_driver


def predict_supervised(df, model_path, feature_cols):
    model = joblib.load(model_path)
    X = df[feature_cols]
    preds = model.predict(X)
    df["supervised_pit_decision"] = preds
    return df



def predict_rl(df, model_path, feature_cols):

    if model_path.endswith(".zip"):
        rl_model = DQN.load(model_path)
    else:
        raise ValueError("RL model non supporté, attend un .zip de stable-baselines3")

    rl_actions = []
    for _, row in df.iterrows():
        obs = row[feature_cols].values.astype(np.float32)
        action, _ = rl_model.predict(obs, deterministic=True)
        rl_actions.append(action)

    df["rl_pit_decision"] = rl_actions
    return df



def compare_strategies(df, laptime_col, pit_cost, post_pit_boost, post_pit_boost_tours, output_file):
    results = {
        "supervised_total_time": 0,
        "rl_total_time": 0,
        "n_laps": len(df)
    }


    for strategy in ["supervised_pit_decision", "rl_pit_decision"]:
        total_time = 0
        boost_counter = 0

        for _, row in df.iterrows():
            lap_time = row[laptime_col]

            
            if row[strategy] == 1:
                total_time += pit_cost
                boost_counter = post_pit_boost_tours
            else:
                if boost_counter > 0:
                    lap_time = lap_time / post_pit_boost
                    boost_counter -= 1

            total_time += lap_time

        results[f"{strategy}_time"] = total_time

 
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Résultats sauvegardés dans", output_file)
    print(results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparer stratégie supervisée vs RL")

    parser.add_argument("--csv", type=str, required=True, help="Chemin vers le fichier CSV")
    parser.add_argument("--driver", type=str, required=True, help="Nom du driver à analyser")
    parser.add_argument("--ml_model", type=str, required=True, help="Chemin vers modèle supervisé (.pkl)")
    parser.add_argument("--rl_model", type=str, required=True, help="Chemin vers modèle RL (.zip)")
    parser.add_argument("--feature_cols", type=str, default=",".join(DEFAULT_FEATURE_COLS),
                        help="Colonnes utilisées comme features (séparées par ,)")
    parser.add_argument("--laptime_col", type=str, default=DEFAULT_LAPTIME_COL, help="Colonne temps au tour")
    parser.add_argument("--pit_cost", type=float, default=20, help="Coût en secondes d’un pit stop")
    parser.add_argument("--post_pit_boost", type=float, default=1.5, help="Boost de performance après pit stop")
    parser.add_argument("--post_pit_boost_tours", type=int, default=3, help="Tours pendant lesquels le boost est actif")
    parser.add_argument("--output", type=str, default="results/comparison.json", help="Fichier sortie résultats")

    args = parser.parse_args()

    feature_cols = args.feature_cols.split(",")


    df = load_data(args.csv, args.driver)


    df = predict_supervised(df, args.ml_model, feature_cols)


    df = predict_rl(df, args.rl_model, feature_cols)


    compare_strategies(
        df,
        laptime_col=args.laptime_col,
        pit_cost=args.pit_cost,
        post_pit_boost=args.post_pit_boost,
        post_pit_boost_tours=args.post_pit_boost_tours,
        output_file=args.output
    )

