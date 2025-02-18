import gym
from pinochle_env import PinochleEnv
from stable_baselines3 import DQN

# Create the environment.
env = PinochleEnv()

# Initialize the DQN model with an MLP policy.
model = DQN("MlpPolicy", env, verbose=1)

# Train the model (adjust total_timesteps as needed).
model.learn(total_timesteps=10000)

# Test the trained model.
obs = env.reset()
done = False
total_reward = 0
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward

print("Test episode reward:", total_reward)