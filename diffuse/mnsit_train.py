import jax
import jax.numpy as jnp
import einops
import sys
import os
sys.path.append("./")
from diffuse.unet import UNet
from diffuse.score_matching import score_match_loss
from diffuse.sde import SDE, LinearSchedule
from functools import partial
import numpy as np
import optax
from tqdm import tqdm
from diffuse.sde import SDE, SDEState
import matplotlib.pyplot as plt
import datetime

data = jnp.load("dataset/mnist.npz")
key = jax.random.PRNGKey(0)

xs = jnp.concatenate([data["x_train"], data["x_test"]], axis=0)

batch_size = 256
n_epochs = 3500
n_t = 256
tf = 2.0
dt = tf / n_t
ts = jnp.linspace(0, tf, n_t)
dts = jnp.diff(ts)

xs = jax.random.permutation(key, xs, axis=0)
data = einops.rearrange(xs, "b h w -> b h w 1")
shape_sample = data.shape[1:]

max_val = data.max()
min_val = data.min() 
data = (data - min_val) / (max_val - min_val) * 2 - 1

beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
sde = SDE(beta)

nn_unet = UNet(dt, 64, upsampling="pixel_shuffle")
init_params = nn_unet.init(
    key, jnp.ones((batch_size, *shape_sample)), jnp.ones((batch_size,))
)

def weight_fun(t):
    int_b = sde.beta.integrate(t, 0).squeeze()
    return 1 - jnp.exp(-int_b)


loss = partial(score_match_loss, lmbda=jax.vmap(weight_fun), network=nn_unet)

nsteps_per_epoch = data.shape[0] // batch_size
until_steps = int(0.95 * n_epochs) * nsteps_per_epoch
lr = 2e-4
schedule = optax.cosine_decay_schedule(
    init_value=lr, decay_steps=until_steps, alpha=1e-2
)
optimizer = optax.adam(learning_rate=schedule)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
ema_kernel = optax.ema(0.99)


@jax.jit
def step(key, params, opt_state, ema_state, data):
    val_loss, g = jax.value_and_grad(loss)(params, key, data, sde, n_t, tf)
    updates, opt_state = optimizer.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    ema_params, ema_state = ema_kernel.update(params, ema_state)
    return params, opt_state, ema_state, val_loss, ema_params


params = init_params
opt_state = optimizer.init(params)
ema_state = ema_kernel.init(params)

# Plot code
def plot(sample, dir_path, i):
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(sample.squeeze(), cmap="gray")
    ax.axis("off")
    plt.savefig(f"{dir_path}/sample_{i}.png")
    plt.close()

def plot_samples(key, dir_path, num_generations=2):
    def nn_score(x, t):
        return nn_unet.apply(params, x, t)

    # Sampling code
    init_samples = jax.random.normal(key, (num_generations, *shape_sample)) # Sample from prior     
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

# create samples dir from time
now = datetime.datetime.now()
dir_path = f"samples/{now.strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(dir_path)

for epoch in range(n_epochs):
    subkey, key = jax.random.split(key)
    idx = jax.random.choice(
        subkey, data.shape[0], (nsteps_per_epoch, batch_size), replace=False
    )
    p_bar = tqdm(range(nsteps_per_epoch))
    list_loss = []
    for i in p_bar:
        subkey, key = jax.random.split(key)
        params, opt_state, ema_state, val_loss, ema_params = step(
            subkey, params, opt_state, ema_state, data[idx[i]]
        )
        p_bar.set_postfix({"loss=": val_loss})
        list_loss.append(val_loss)
    print(f"epoch=: {epoch} | mean_loss=: {sum(list_loss) / nsteps_per_epoch}")

    if (epoch + 1) % 500 == 0:
        np.savez(f"ann_{epoch}.npz", params=params, ema_params=ema_params)
    
    if (epoch + 1) % 50 == 0:
        save_path = f"{dir_path}/{epoch}"
        os.makedirs(save_path)
        plot_samples(key, save_path, num_generations=2)

np.savez("ann_end.npz", params=params, ema_params=ema_params)
