import os
from pinochle_bidding_env import PinochleBiddingEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

# Define a callback to evaluate win rate over time
class WinRateCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.win_rates = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            wins = 0
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                # Run one complete episode
                while not done:
                    action, _ = self.model.predict(obs)
                    obs, reward, done, info = self.eval_env.step(action)
                # Use info from the final step to decide win (win if ml_total > opp_total)
                if info.get("ml_total", 0) > info.get("opp_total", 0):
                    wins += 1
            win_rate = wins / self.n_eval_episodes
            self.win_rates.append(win_rate)
            if self.verbose:
                print(f"Win rate after {self.n_calls} steps: {win_rate*100:.1f}%")
        return True

# Create training and evaluation environments
env = PinochleBiddingEnv()
eval_env = PinochleBiddingEnv()

# Initialize the DQN model with MultiInputPolicy since our observation is a dict.
model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

# Create our win rate callback
win_rate_callback = WinRateCallback(eval_env, eval_freq=1000, n_eval_episodes=10, verbose=1)

# Train the model
model.learn(total_timesteps=20000, callback=win_rate_callback)