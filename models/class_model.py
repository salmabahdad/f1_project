"""from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_classification_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("📊 Évaluation du modèle de classification :")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
"""
'''
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb



def train_classification_model(X_train, y_train):
# Créer le modèle XGBoost
 model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.01,
    scale_pos_weight= (len(y_train[y_train==0]) / len(y_train[y_train==1])),  # pour classes déséquilibrées
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
  )

# Entraîner le modèle
 model.fit(X_train, y_train)
 return model

def evaluate_classification_model(model, X_test, y_test):
# Prédictions
 y_pred = model.predict(X_test)

# Évaluation
 print("Accuracy:", accuracy_score(y_test, y_pred))
 print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 print("Classification Report:\n", classification_report(y_test, y_pred))
'''
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy.stats import randint, uniform
import joblib

def train_classification_model(X_train, y_train):
    """
    Entraîne un modèle LightGBM avec GridSearchCV pour maximiser le F1-score
    sur la classe minoritaire.
    """
    # Créer le modèle LightGBM
    model = lgb.LGBMClassifier(
        class_weight='balanced',  # pour gérer le déséquilibre
        random_state=42
    ) 
    
    # Grille d'hyperparamètres pour GridSearchCV
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50]
    }
    
    # Validation croisée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # GridSearchCV pour optimiser les hyperparamètres
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',  # On maximise le F1-score pour la classe minoritaire
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    # Entraîner le modèle
    grid_search.fit(X_train, y_train)
    
    # Afficher les meilleurs paramètres
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleur F1-score :", grid_search.best_score_)
    # --- FIX pour feature_cols ---
    if isinstance(X_train, pd.DataFrame):
     feature_cols = X_train.columns.tolist()
    else:
     feature_cols = [f"feature_{i}" for i in range(X_train.shape[1])]
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    return grid_search


def evaluate_classification_model(model, X_test, y_test, threshold=0.48):
    """
    Évalue le modèle LightGBM sur les données de test et applique un threshold custom.
    """
    best_model = model.best_estimator_
    
    # Prédiction des probabilités pour la classe 1
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Appliquer le threshold personnalisé
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Évaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    joblib.dump(model, "models/pitNextLap_lgb.pkl")

