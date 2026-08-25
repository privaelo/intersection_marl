import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt

INTERSECTION_CONFIG = {
    "duration": 20,             #seconds per episode
    "initial_vehicle_count": 5, #number of vehicles at the beginning of the episode
    "spawn_probability": 0.4,   
}

def make_env(render_mode=None, normalize_obs=True):
    """normalize_obs=False gives raw meters / (m/s) instead of [-1,1]-scaled
    features. Phase 1's rule thresholds are written in meters, so it needs
    normalize_obs=False. Phases 2-3 should leave this True (default) —
    normalized inputs are standard practice for stable NN training. This
    only changes observation *preprocessing*, not the environment's physics,
    action space, or reward — so it doesn't break the shared-env comparison."""

    env=gym.make("intersection-multi-agent-v2", 
                    render_mode=render_mode, 
                    config=INTERSECTION_CONFIG)
    if not normalize_obs:
        env.unwrapped.config["observation"]["observation_config"]["normalize"] = False
    return env

