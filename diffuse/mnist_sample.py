import os
import argparse
from functools import partial
from typing import Tuple, Callable
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append("./")
from diffuse.sde import SDE, SDEState
from diffuse.sde import LinearSchedule
from diffuse.unet import UNet

SCORE_NN_PATH = "./ann_2999.npz"
GROUND_TRUTH_SHAPE = (28, 28, 1)

def plot(sample, dir_path, i):
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(sample.squeeze(), cmap="gray")
    ax.axis("off")
    plt.savefig(f"{dir_path}/sample_{i}.png")
    plt.close()

def main(num_generations: int, key: PRNGKeyArray, dir_path: str):
    # Initialize SDE parameters
    tf = 2.0
    n_t = 256
    dt = tf / n_t
    ts = jnp.linspace(0, tf, n_t)
    dts = jnp.diff(ts)

    # Define beta schedule and SDE
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

    # Initialize ScoreNetwork
    score_net = UNet(dt, 64, upsampling="pixel_shuffle")
    nn_trained = jnp.load(SCORE_NN_PATH, allow_pickle=True)
    params = nn_trained["params"].item()

    # Define neural network score function
    def nn_score(x, t):
        return score_net.apply(params, x, t)

    sde = SDE(beta=beta)

    # Sampling code
    init_samples = jax.random.normal(key, (num_generations, *GROUND_TRUTH_SHAPE)) # Sample from prior     
    tfs = jnp.zeros((num_generations,)) + tf

    # Denoise
    state_f = SDEState(position=init_samples, t=tfs)
    revert_sde = partial(sde.reverso, score=nn_score, dts=dts)

    # Split keys
    keys = jax.random.split(key, num_generations)
    state_f, history = jax.vmap(revert_sde)(keys, state_f)
    
    # plot 
    for i in range(num_generations):
        plot(state_f.position[i], dir_path, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rng_key", type=int, default=0)
    parser.add_argument("--num_generations", type=int, default=25)
    parser.add_argument("--prefix", type=str, default="")

    args = parser.parse_args()
    num_generations = args.num_generations
    key_int = args.rng_key

    key = jax.random.PRNGKey(key_int)
    dir_path = f"mnist_samples/{key_int}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(dir_path, exist_ok=True)

    main(num_generations, key, dir_path)
