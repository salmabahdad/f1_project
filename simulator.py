import pandas as pd
import joblib
import numpy as np


df = pd.read_csv("F1_ALL_RACES.csv")  

clf = joblib.load("models/pitNextLap_lgb.pkl")


feature_cols = joblib.load("models/feature_cols.pkl")

driver = df['Driver'].unique()[0]  
episode = df[df['Driver'] == driver].sort_values("LapNumber")

threshold = 0.48 

print(f"Simulation pour le pilote : {driver}\n")
for _, row in episode.iterrows():

    X = row[feature_cols].values.reshape(1, -1)

    prob = clf.predict_proba(X)[0, 1]
    

    pred_action = 1 if prob >= threshold else 0
    

    true_action = row["PitNextLap"]
    
    # Affichage
    print(f"Lap {int(row['LapNumber'])} | Prob={prob:.2f} | Prédit={pred_action} | Vérité={true_action}")


y_true = episode["PitNextLap"].values
y_pred = np.array([
    1 if clf.predict_proba(row[feature_cols].values.reshape(1, -1))[0, 1] >= threshold else 0
    for _, row in episode.iterrows()
])

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\n--- Résultats globaux ---")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred))
