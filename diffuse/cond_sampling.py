from functools import partial
import argparse
from datetime import datetime
import os
import sys
sys.path.append("./")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from diffuse.conditional import CondSDEImplicit
from diffuse.sde import SDE, SDEState
from diffuse.sde import LinearSchedule
from diffuse.unet import UNet
from diffuse.vae_train import Encoder
from diffuse.conditional import impl_log_likelihood

# to do some automatic logging
import torch
import torchvision
import torchvision.transforms as transforms

# diffusion hyperparameters 
tf = 2.0
n_t = 256
dt = tf / n_t
ts = jnp.linspace(0, tf, n_t)
dts = jnp.diff(ts)
beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=tf)
ground_truth_shape = (28, 28, 1)

data_path = "dataset/mnist.npz"

def load_vae_likelihood_fn(args):
    encoder = Encoder(args.latent_dim)
    weight_path = args.weight_path
    weights = jnp.load(weight_path, allow_pickle=True)
    encoder_params = weights['encoder'].item()

    likelihood_fn = partial(impl_log_likelihood, encoder, encoder_params, b=0, a=1, m=0.0001, measurement_noise=0.5)

    return likelihood_fn

def load_sde(args, likelihood_fn):
    # load score network
    score_nn_path = args.score_nn_path
    score_net = UNet(dt, 64, upsampling="pixel_shuffle")
    nn_trained = jnp.load(score_nn_path, allow_pickle=True)
    params = nn_trained["params"].item()

    def nn_score(x, t):
        return score_net.apply(params, x, t)

    cond_sde = CondSDEImplicit(beta=beta, likelihood_fn=likelihood_fn, tf=tf, score=nn_score)

    return cond_sde, nn_score

def main(args):
    # Load 
    likelihood_fn = load_vae_likelihood_fn(args)
    cond_sde, nn_score = load_sde(args, likelihood_fn)

    # Load data
    data = jnp.load(data_path)
    xs = jnp.concatenate([data["x_train"], data["x_test"]], axis=0)

    def preprocess(data): # Normalize to [-1, 1]
        max_val = data.max()
        min_val = data.min()
        data = (data - min_val) / (max_val - min_val) * 2 - 1
        return data
    
    # pick a random _xi_
    xs = preprocess(xs)
    idx = jax.random.randint(jax.random.PRNGKey(args.seed), (1,), 0, xs.shape[0])
    xi = xs[idx]

    # plot and save
    plt.imshow(xi.squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("xi")
    plt.savefig(f"{args.run_path}/xi.png")
    plt.close()

    key = jax.random.PRNGKey(args.seed)
    num_generations = args.num_generations

    # Sampling code
    init_samples = jax.random.normal(key, (num_generations, *ground_truth_shape)) # Sample from prior     
    keys = jax.random.split(key, num_generations)
    tfs = jnp.zeros((num_generations,)) + tf
    state_f = SDEState(position=init_samples, t=tfs)

    revert_sde = partial(cond_sde.reverso, score=nn_score, dts=dts, y=args.observation_value, xi=xi) 

    # Samples
    state_f, _ = jax.vmap(revert_sde)(keys, state_f)
    
    # plot the final result
    for i in range(num_generations):
        plt.imshow(state_f.position[i].squeeze(), cmap="gray")
        plt.axis("off")
        plt.savefig(f"{args.run_path}/sample_{i}.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_nn_path", type=str, default="ann_3499.npz")
    parser.add_argument("--weight_path", type=str, default="vae_model-10/20250310_004631/vae_params.npz")
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_generations", type=int, default=25)
    # sampling hyperparameters
    parser.add_argument("--observation_value", type=float, default=10)
    # start looking into sampling for a range of (xi, y) pairs to see if this makes sense
    args = parser.parse_args()
    
    # run_path
    hparams_string = f"obs={args.observation_value}-lat={args.latent_dim}/"
    args.run_path = f"mnist_samples/{hparams_string}/{args.seed}/"

    # make the directory
    os.makedirs(args.run_path, exist_ok=True)
    main(args)