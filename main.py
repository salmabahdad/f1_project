import os
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_all_races
from preprocessing import prepare_data
from models.regression_model import train_regression_model, evaluate_regression_model
from models.classification_model import train_classification_model, evaluate_classification_model


clean_data_path = 'F1_ALL_RACES.csv'
if os.path.exists(clean_data_path):
  df_all_races= pd.read_csv(clean_data_path)
else:
  races = ['Monza', 'Silverstone', 'Spa', 'Austria','Brazil','Canada', 'France', 'Belgium', 'Hungary', 'Italy', 'Japan', 'Singapore', 'United States', 'Mexico', 'Abu Dhabi']
  df_all_races = load_all_races(races) 


X_reg_train, X_reg_test, y_reg_train, y_reg_test, X_class_train, X_class_test, y_class_train, y_class_test = prepare_data(df_all_races)

 
class_model = train_classification_model(X_class_train, y_class_train)
evaluate_classification_model(class_model, X_class_test, y_class_test)

reg_model = train_regression_model(X_reg_train, y_reg_train)
evaluate_regression_model(reg_model, X_reg_test, y_reg_test)
