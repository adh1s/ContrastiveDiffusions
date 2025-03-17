import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from flax.training.train_state import TrainState
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import os
import argparse

# Encoder
class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mean, logvar

# Decoder
class Decoder(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(256)(z)
        z = nn.relu(z)
        z = nn.Dense(512)(z)
        z = nn.relu(z)
        z = nn.Dense(self.output_dim)(z)
        z = z.reshape((z.shape[0], int(self.output_dim**0.5), int(self.output_dim**0.5)))
        return nn.tanh(z)  # Change to tanh to output range -1 to 1

# Reparameterization Trick
def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, shape=mean.shape)
    return mean + eps * std

# Loss Function
def train_step(state, batch, rng, latent_dim=10, input_dim=28*28):
    def loss_fn(params):
        mean, logvar = Encoder(latent_dim).apply(params["encoder"], batch)
        z = reparameterize(rng, mean, logvar)
        x_recon = Decoder(input_dim).apply(params["decoder"], z)

        recon_loss = jnp.mean(jnp.sum((x_recon - batch) ** 2, axis=-1))
        kl_loss = -0.5 * jnp.mean(1 + logvar - jnp.square(mean) - jnp.exp(logvar))
        return recon_loss + kl_loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Sample from Prior
def sample_from_prior(rng, decoder, decoder_params, n_samples=10, latent_dim=10):
    z = jax.random.normal(rng, shape=(n_samples, latent_dim))  # Sample from standard normal
    samples = decoder.apply(decoder_params, z)  # Pass through decoder to generate samples
    return samples
    
def plot_sample(sample, dir_path, i):
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(sample.squeeze(), cmap="gray")
    ax.axis("off")
    plt.savefig(f"{dir_path}/sample_{i}.png")
    plt.close()

# Training Loop
def train_vae(epochs=100, latent_dim=10, batch_size=256, learning_rate = 1e-3):
    input_dim = 28 * 28

    encoder = Encoder(latent_dim)
    decoder = Decoder(input_dim)
    rng = jax.random.PRNGKey(0)

    dummy_input = jnp.ones((1, input_dim))
    encoder_params = encoder.init(rng, dummy_input)
    decoder_params = decoder.init(rng, jnp.ones((1, latent_dim)))

    def preprocess(data): # Normalize to [-1, 1]
        max_val = data.max()
        min_val = data.min()
        data = (data - min_val) / (max_val - min_val) * 2 - 1
        return data

    data = jax.numpy.load("dataset/mnist.npz")
    data = jax.random.permutation(rng, data["x_train"])
    data = preprocess(data)

    optimizer = optax.adam(learning_rate)
    state = TrainState.create(
        apply_fn=None,  # Not needed, we apply manually
        params={"encoder": encoder_params, "decoder": decoder_params},
        tx=optimizer
        )
    
    nsteps_per_epoch = data.shape[0] // batch_size
    
    now = datetime.datetime.now()
    samples_dir = f"vae_samples-{latent_dim}/{now.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(samples_dir, exist_ok=True)

    n_samples = 10

    for epoch in range(epochs):
        key, rng = jax.random.split(rng)
        idx = jax.random.choice(key, data.shape[0], (nsteps_per_epoch, batch_size), replace=False)
        p_bar = tqdm(range(nsteps_per_epoch))
        for i in p_bar:
            batch = data[idx[i]]
            state, loss = train_step(state, batch, key, latent_dim, input_dim)

        if epoch % 25 == 0:
            samples = sample_from_prior(key, decoder, state.params["decoder"], n_samples, latent_dim)
            for i in range(n_samples):
                save_path = f"{samples_dir}/epoch_{epoch}"
                os.makedirs(save_path, exist_ok=True)
                plot_sample(samples[i], save_path, i)
    
    # save model
    model_dir = f"vae_model-{latent_dim}/{now.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_dir, exist_ok=True)
    np.savez(f"{model_dir}/vae_params.npz", encoder=encoder_params, decoder=decoder_params)

if __name__ == "__main__":
    # parse args
    # set the backend to be cpu
    jax.config.update("jax_platform_name", "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=10)
    args = parser.parse_args()
    train_vae(args.epochs, args.latent_dim)