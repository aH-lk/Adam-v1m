import os
from stable_baselines3 import SAC
import gymnasium as gym
env = gym.make('Humanoid-v4', render_mode='human')

model_path = 'SAC_1225000.zip'
model = SAC.load(model_path, env=env)
obs = env.reset()[0]
done = False
extra_steps = 500
while True:
    action, _ = model.predict(obs)
    obs, _, done, _, _ = env.step(action)

    if done:
        extra_steps -= 1

        if extra_steps < 0:
            break

env.close()
