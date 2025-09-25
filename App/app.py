import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_agent.f1_env import F1PitstopEnv


df = pd.read_csv("F1_ALL_RACES.csv")


env = DummyVecEnv([lambda: F1PitstopEnv(df, Driver="HAM")])


model = DQN.load("models/dqn_f1")


obs = env.reset()
actions, rewards, true_actions = [], [], []

for _ in range(50): 
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)


    true_action = env.envs[0].df.loc[env.envs[0].current_step-1, "PitNextLap"]

    actions.append(int(action))
    rewards.append(float(reward))
    true_actions.append(int(true_action))

    if done.any():
        break


st.title("🏎️ F1 Pitstop Simulation")


st.subheader("📊 Timeline des actions de l’agent")
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(actions, label="Agent", marker="o")
ax.plot(true_actions, label="Pilote (réel)", marker="x")
ax.set_xlabel("Tour")
ax.set_ylabel("Action (0=rouler, 1=pit)")
ax.legend()
st.pyplot(fig)


st.subheader("📈 Récompenses de l’agent")
cum_rewards = pd.Series(rewards).cumsum()
fig2, ax2 = plt.subplots()
ax2.plot(rewards, label="Reward par tour", alpha=0.6)
ax2.plot(cum_rewards, label="Cumul des rewards", linewidth=2)
ax2.legend()
ax2.set_xlabel("Tour")
ax2.set_ylabel("Reward")
st.pyplot(fig2)


st.subheader("📋 Résumé des décisions")
results_df = pd.DataFrame({
    "Tour": range(1, len(actions)+1),
    "Action Agent": actions,
    "Action Réelle": true_actions,
    "Reward": rewards
})
st.dataframe(results_df)
