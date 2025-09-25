import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from f1_env import F1PitstopEnv

df = pd.read_csv("F1_ALL_RACES.csv")

env = DummyVecEnv([lambda: F1PitstopEnv(df, Driver="VER")])

model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

model.save("models/dqn_f1")

obs = env.reset()
for _ in range(50):
    action, _ = model.predict(obs)   
    obs, reward, done, info = env.step(action)
    print(f"Action={action}, Reward={reward}")
    if done.any():
        obs = env.reset()

