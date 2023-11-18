import os
from stable_baselines3 import SAC
import gymnasium as gym
# folder path
dir_path = r'/Users/goks/Desktop/MyCode/Python/Adam-v1/saved_models'

# list to store files
resi = []
res = []
models = []

# make env
env = gym.make('Humanoid-v4', render_mode='human')

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        resi.append(path)

for i in resi:
    res.append(i.replace('SAC_', '').replace('.zip', ''))

for fi in range(len(res)):
    models.append(res[fi])

models.sort(key=int)
model_path = 'saved_models/' + 'SAC_' + models[-1] + '.zip'
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
