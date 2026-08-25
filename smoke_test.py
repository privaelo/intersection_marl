"""
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt

env = gym.make("intersection-multi-agent-v2", render_mode="rgb_array")
env.reset()
frame = env.render()
plt.imsave("intersection-multi-agent-v2.png", frame)
"""
from env_config import make_env

env = make_env()
obs, info = env.reset()

print("Number of agents:", len(obs))
print("Obs shape per agent:", obs[0].shape)
print("Action space:", env.action_space)

obs, reward, terminated, truncated, info = env.step((1, 1))  # both IDLE
print("Reward:", reward, "| Terminated:", terminated, "| Truncated:", truncated)
