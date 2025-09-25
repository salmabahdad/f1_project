from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import joblib


def train_regression_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("📈 Évaluation du modèle de régression :")
    print(f"MAE (Mean Absolute Error): {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"MSE (Mean Squared Error): {mean_squared_error(y_test, y_pred):.3f}")
    r2 = r2_score(y_test, y_pred)

    print("R² Score:", r2)
    joblib.dump(model, "models/nextPit_lgb.pkl")
    
