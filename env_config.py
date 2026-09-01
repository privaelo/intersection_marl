import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt

INTERSECTION_CONFIG = {
    "duration": 20,             #seconds per episode
    "initial_vehicle_count": 5, #number of vehicles at the beginning of the episode
    "spawn_probability": 0.4,   
}

def make_env(render_mode=None, normalize_obs=True):

    env=gym.make("intersection-multi-agent-v2", 
                    render_mode=render_mode, 
                    config=INTERSECTION_CONFIG)
    if not normalize_obs:
        env.unwrapped.config["observation"]["observation_config"]["normalize"] = False
    return env

