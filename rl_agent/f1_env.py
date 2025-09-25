import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class F1PitstopEnv(gym.Env):
    def __init__(self, df, Driver):
        super(F1PitstopEnv, self).__init__()
        # Garder uniquement les données du pilote choisi
        self.df = df[df["Driver"] == Driver].reset_index(drop=True)
        self.current_step = 0

        # Colonnes de features (exclure label + colonnes non utiles)
        self.feature_cols = self.df.drop(columns=["PitNextLap", "Driver", "LapNumber"]).columns
        n_features = len(self.feature_cols)

        # L’espace d’observation (features déjà prétraitées)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(n_features,), 
            dtype=np.float32
        )

        # L’espace d’action : 0 = ne pas pitter, 1 = pitter
        self.action_space = spaces.Discrete(2)

  

    def reset(self, seed=None, options=None):
     super().reset(seed=seed)   # obligatoire pour Gymnasium

     self.current_step = 0

     # sécurité : vérifier que le DataFrame n’est pas vide
     if len(self.df) == 0:
        raise ValueError("Le DataFrame est vide, impossible de reset l’environnement")

     obs = self._get_state()
     info = {}

     return obs, info



    def step(self, action):
     
    # Get current features
     rain = self.df.loc[self.current_step, "Rainfall_True"]
     safety_car = self.df.loc[self.current_step, "IsSafetyCar_0"]
     fresh_tyre = self.df.loc[self.current_step, "FreshTyre_True"]
     true_action = self.df.loc[self.current_step, "PitNextLap"]

    #Base imitation reward
     reward = 1 if action == true_action else -1 
   
    # Rain logic
     if rain == 1 and action == 1:
        reward += 3   # smart to pit when rain starts
     if rain == 1 and action == 0:
        reward -= 3   # ignoring rain is bad

    #Safety Car logic
     if safety_car == 1 and action == 1:
        reward += 2   # good timing to pit
     if safety_car == 0 and action == 1:
        reward -= 1   # bad timing under green flag

    # Fresh Tyre logic
     if fresh_tyre == 1 and action == 1:
        reward -= 2   # unnecessary pit

    # 5️Advance step
     self.current_step += 1
     terminated = self.current_step >= len(self.df) - 1
     truncated = False
     obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._get_state()

     info = {
        "rain": rain,
        "safety_car": safety_car,
        "fresh_tyre": fresh_tyre,
        "true_action": true_action
     }

     return obs, reward, terminated, truncated, info





    def _get_state(self):
      """Retourner les features normalisées/standardisées à l’instant courant."""
      features = self.df[self.feature_cols].iloc[self.current_step].values.copy()

      for i, val in enumerate(features):
        # Si c'est un timedelta, convertir en secondes
        if isinstance(val, pd.Timedelta):
            features[i] = val.total_seconds()
        # Si c'est un string de type '0 days 01:02:14.632000'
        elif isinstance(val, str) and 'days' in val:
            td = pd.to_timedelta(val)
            features[i] = td.total_seconds()
    
      return features.astype(np.float32)
