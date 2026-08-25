import imageio

def save_gif(frames, path, fps=5):
    imageio.mimsave(path, frames, fps=fps, loop=0)  # 0 = infinite loop