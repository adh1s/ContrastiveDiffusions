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
from optax import GradientTransformation
import matplotlib.pyplot as plt
from jax_tqdm import scan_tqdm
import datetime
import einops
import sys
sys.path.append("./")
from diffuse.sde import SDE, SDEState
from diffuse.conditional import CondSDE
from diffuse.plotting import log_samples, plotter_random, plot_comparison, metric_l2, metric_ssim
from diffuse.images import SquareMask
from diffuse.inference import generate_cond_sampleV2
from diffuse.optimizer import ImplicitState, impl_step, impl_one_step
from diffuse.sde import LinearSchedule
from diffuse.unet import UNet
import json

SIZE = 7
SCORE_NETWORK = 'score_networks/ann_3499.npz'
DATASET = 'dataset/mnist.npz'

def save_results(l2_hist, ssim_hist, output_dir, seed, random=False):
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "l2_hist": l2_hist,
        "ssim_hist": ssim_hist
    }
    if random:
        output_file = os.path.join(output_dir, f"random_seed_{seed}.json")
    else:
        output_file = os.path.join(output_dir, f"seed_{seed}.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

def initialize_experiment(key: PRNGKeyArray):
    data = np.load(DATASET)
    xs = jnp.concatenate([data["x_train"], data["x_test"]], axis=0)

    xs = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)  # Add channel dimension
    # Center to [-1, 1]
    xs = (xs - xs.min()) / (xs.max() - xs.min()) * 2 - 1

    # Initialize parameters
    tf = 2.0
    n_t = 256

    # Define beta schedule and SDE
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

    # Initialize ScoreNetwork
    score_net = UNet(tf / n_t, 64, upsampling="pixel_shuffle")
    nn_trained = jnp.load(SCORE_NETWORK, allow_pickle=True)
    params = nn_trained["params"].item()

    # Define neural network score function
    def nn_score(x, t):
        return score_net.apply(params, x, t)

    # Set up mask and measurement
    ground_truth = jax.random.choice(key, xs)
    mask = SquareMask(SIZE, ground_truth.shape)

    # Set up conditional SDE
    cond_sde = CondSDE(beta=beta, mask=mask, tf=tf, score=nn_score)
    sde = SDE(beta=beta)

    return sde, cond_sde, mask, ground_truth, tf, n_t, nn_score


def optimize_design(
    key_step: PRNGKeyArray,
    implicit_state: ImplicitState,
    past_y: Array,
    mask_history: Array,
    optimizer: GradientTransformation,
    ts: Array,
    dt: float,
    cond_sde: CondSDE,
):
    opt_steps = 300
    keys_opt = jax.random.split(key_step, opt_steps)

    @scan_tqdm(opt_steps)
    def step(new_state, tup):
        _, key = tup
        new_state = impl_step(
            new_state,
            key,
            past_y,
            mask_history,
            cond_sde=cond_sde,
            optx_opt=optimizer,
            ts=ts,
            dt=dt,
        )
        # new_state = impl_full_scan(
        #     new_state,
        #     key,
        #     past_y,
        #     mask_history,
        #     cond_sde=cond_sde,
        #     optx_opt=optimizer,
        #     ts=ts,
        #     dt=dt,
        # )

        return new_state, new_state.design

    new_state, hist_design = jax.lax.scan(
        step, implicit_state, (jnp.arange(0, opt_steps), keys_opt)
    )
    return new_state, hist_design


def optimize_design_one_step(
    key_step: PRNGKeyArray,
    implicit_state: ImplicitState,
    past_y: Array,
    mask_history: Array,
    optimizer: GradientTransformation,
    ts: Array,
    dt: float,
    cond_sde: CondSDE,
    single_step_ub: int = 1.4, # original values
    inner_gradient_steps: int = 100, # inner_steps to take t > single_step_ub
):
    opt_steps = ts.shape[0] - 1
    keys_opt = jax.random.split(key_step, opt_steps)

    @scan_tqdm(opt_steps)
    def step(new_state, tup):
        _, y, y_next, key = tup
        new_state = impl_one_step(
            new_state,
            key,
            y,
            y_next,
            mask_history,
            cond_sde=cond_sde,
            optx_opt=optimizer,
            single_step_ub=single_step_ub,
            inner_gradient_steps=inner_gradient_steps,
        )
        thetas, _, cntrst_thetas, _, design, *_ = new_state
        return new_state, (thetas, cntrst_thetas, design)

    y = jax.tree.map(lambda x: x[:-1], past_y)
    y_next = jax.tree.map(lambda x: x[1:], past_y)
    new_state, hist = jax.lax.scan(
        step, implicit_state, (jnp.arange(0, opt_steps), y, y_next, keys_opt)
    )

    thetas, cntrst_thetas, design_hist = hist
    state = ImplicitState(
        thetas,
        new_state.weights,
        cntrst_thetas,
        new_state.weights_c,
        new_state.design,
        new_state.opt_state,
    )
    return state, design_hist


def init_trajectory(
    key_init,
    sde,
    nn_score,
    n_samples,
    n_samples_cntrst,
    tf,
    ts,
    dts,
    ground_truth_shape,
):
    """
    Unitialize full trajectory for conditional sampling using parallel update from Marion et al 2024
    """
    # Initialize samples
    init_samples = jax.random.normal(
        key_init, (n_samples + n_samples_cntrst, *ground_truth_shape)
    )
    tfs = jnp.zeros((n_samples + n_samples_cntrst,)) + tf

    # Denoise
    state_f = SDEState(position=init_samples, t=tfs)
    revert_sde = partial(sde.reverso, score=nn_score, dts=dts)

    # Make right shape
    keys = jax.random.split(key_init, n_samples + n_samples_cntrst)
    _, state_Ts = jax.vmap(revert_sde)(keys, state_f)
    state_Ts = jax.tree.map(
        lambda arr: einops.rearrange(arr, "n h ... -> h n ..."), state_Ts
    )

    # Init thetas
    thetas = state_Ts.position[:, :n_samples]
    cntrst_thetas = state_Ts.position[:, n_samples:]

    return thetas, cntrst_thetas


def init_start_time(
    key_init: PRNGKeyArray,
    n_samples: int,
    n_samples_cntrst: int,
    ground_truth_shape: Tuple[int, ...],
) -> Tuple[Array, Array]:
    """
    Initialize thetas for just the start time of the conditional sampling
    """
    key_t, key_c = jax.random.split(key_init)
    thetas = jax.random.normal(key_t, (n_samples, *ground_truth_shape))
    cntrst_thetas = jax.random.normal(key_c, (n_samples_cntrst, *ground_truth_shape))
    return thetas, cntrst_thetas


# @jax.jit
def main(
    key: PRNGKeyArray,
    num_meas: int,
    plotter_theta: Callable,
    plotter_contrastive: Callable,
    single_step_ub: float = 1.4,
    inner_gradient_steps: int = 100,
):
    print('Optimizing design')
    key_init, key_step = jax.random.split(key)

    sde, cond_sde, mask, ground_truth, tf, n_t, nn_score = initialize_experiment(
        key_init
    )
    dt = tf / (n_t - 1)
    n_samples = 150
    n_samples_cntrst = 151

    # Time initialization (kept outside the function)
    ts = jnp.linspace(0, tf, n_t)
    dts = jnp.diff(ts)

    # init design and measurement hist
    # design = jax.random.uniform(key_init, (2,), minval=0, maxval=28)
    design = jnp.array([0.1, 0.1])
    y = cond_sde.mask.measure(design, ground_truth)
    design = jax.random.uniform(key_init, (2,), minval=0, maxval=28)

    measurement_history = jnp.zeros((num_meas, *y.shape))
    measurement_history = measurement_history.at[0].set(y)

    # init optimizer
    optimizer = optax.chain(optax.adam(learning_rate=0.1), optax.scale(-1))
    opt_state = optimizer.init(design)

    # init thetas
    # thetas, cntrst_thetas = init_trajectory(key_init, sde, nn_score, n_samples, n_samples_cntrst, tf, ts, dts, ground_truth.shape)
    thetas, cntrst_thetas = init_start_time(
        key_init, n_samples, n_samples_cntrst, ground_truth.shape
    )
    weights_0 = jnp.zeros((n_samples,))
    weights_c_0 = jnp.zeros((n_samples_cntrst,))
    implicit_state = ImplicitState(
        thetas, weights_0, cntrst_thetas, weights_c_0, design, opt_state
    )

    # stock in joint_y all measurements
    joint_y = y
    mask_history = mask.make(design)
    fig, axs = plt.subplots(1, 3)
    ax1, ax2, ax3 = axs
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax1.set_title("Ground truth")
    ax2.set_title("Mesure")
    ax3.set_title("Mask")
    ax1.imshow(ground_truth, cmap="gray")
    ax2.imshow(joint_y, cmap="gray")
    ax3.imshow(mask_history, cmap="gray")
    # jax.experimental.io_callback(plot_results, None, opt_hist, ground_truth, joint_y, mask_history, thetas, cntrst_thetas)

    plt.show()
    # design_step = jax.jit(partial(optimize_design, optimizer=optimizer, ts=ts, dt=dt, cond_sde=cond_sde))
    design_step = partial(
        optimize_design_one_step, optimizer=optimizer, ts=ts, dt=dt, cond_sde=cond_sde, single_step_ub=single_step_ub, inner_gradient_steps=inner_gradient_steps
    )

    l2_hist = []
    ssim_hist = []
    
    for n_meas in range(num_meas):
        key_noise, key_opt, key_gen = jax.random.split(key_step, 3)
        # make noised path for measurements
        keys_noise = jax.random.split(key_noise, n_t)
        state_0 = SDEState(joint_y, jnp.zeros_like(y))
        past_y = jax.jit(jax.vmap(sde.path, in_axes=(0, None, 0)))(
            keys_noise, state_0, ts
        )
        past_y = SDEState(past_y.position[::-1], past_y.t)
        # past_y contains a noisy trajectory of observations [i.e. y0 = A_{xi}\theta + noise, y_t = noised(y0)]
        # plotter_line(past_y.position)

        # optimize design
        optimal_state, hist_implicit = design_step(
            key_opt, implicit_state, past_y, mask_history
        )
        opt_hist = hist_implicit

        # make new measurement
        new_measurement = cond_sde.mask.measure(optimal_state.design, ground_truth)
        measurement_history = measurement_history.at[n_meas].set(new_measurement)

        # add measured data to joint_y and update history of mask location
        joint_y = cond_sde.mask.restore(optimal_state.design, joint_y, new_measurement)
        mask_history = cond_sde.mask.restore(
            optimal_state.design, mask_history, cond_sde.mask.make(optimal_state.design)
        )
        # plt.imshow(mask_history, cmap="gray")

        print(joint_y[10, 20])

        # logging
        print(f"Design_start: {design} Design_end:{optimal_state.design}")

        # jax.experimental.io_callback(plot_results, None, opt_hist, ground_truth, joint_y, mask_history, optimal_state.thetas[-1, :], optimal_state.cntrst_thetas[-1, :])
        jax.experimental.io_callback(
            plotter_theta,
            None,
            opt_hist,
            ground_truth,
            joint_y,
            optimal_state.thetas[-1, :],
            optimal_state.weights,
            n_meas,
        )
        jax.experimental.io_callback(
            plotter_contrastive,
            None,
            opt_hist,
            ground_truth,
            joint_y,
            optimal_state.cntrst_thetas[-1, :],
            optimal_state.weights_c,
            n_meas,
        )

        print(jnp.max(joint_y))

        # reinitiazize implicit state
        design = jax.random.uniform(key_step, (2,), minval=0, maxval=28)
        opt_state = optimizer.init(design)
        key_t, key_c = jax.random.split(key_gen)
    
        key_step, _ = jax.random.split(key_step)

        thetas, cntrst_thetas = init_start_time(
            key_step, n_samples, n_samples_cntrst, ground_truth.shape
        )
        implicit_state = ImplicitState(
            thetas, weights_0, cntrst_thetas, weights_c_0, design, opt_state

        )
        
        l2 = metric_l2(ground_truth, optimal_state.thetas[-1, :], optimal_state.weights)
        ssim = metric_ssim(ground_truth, optimal_state.thetas[-1, :], optimal_state.weights)
        
        print('l2:', l2, 'ssim:', ssim)
        l2_hist.append(l2)
        ssim_hist.append(ssim)

    return ground_truth, (optimal_state.thetas[-1, :], optimal_state.weights), joint_y, l2_hist, ssim_hist

# @jax.jit
def main_random(key: PRNGKeyArray, num_meas: int, plotter_random: Callable):
    print("Randomised designs")
    key_init, key_step = jax.random.split(key)

    sde, cond_sde, mask, ground_truth, tf, n_t, nn_score = initialize_experiment(
        key_init
    )
    dt = tf / (n_t - 1)
    n_samples = 150
    n_samples_cntrst = 151

    # Time initialization (kept outside the function)
    ts = jnp.linspace(0, tf, n_t)
    dts = jnp.diff(ts)

    # init design and measurement hist
    # design = jax.random.uniform(key_init, (2,), minval=0, maxval=28)
    design = jnp.array([0.1, 0.1])
    y = cond_sde.mask.measure(design, ground_truth)
    design = jax.random.uniform(key_init, (2,), minval=0, maxval=28)

    measurement_history = jnp.zeros((num_meas, *y.shape))
    measurement_history = measurement_history.at[0].set(y)

    # init optimizer
    optimizer = optax.chain(optax.adam(learning_rate=0.9), optax.scale(-1))
    opt_state = optimizer.init(design)

    ts = jnp.linspace(0, tf, n_t)

    # init thetas
    # thetas, cntrst_thetas = init_trajectory(key_init, sde, nn_score, n_samples, n_samples_cntrst, tf, ts, dts, ground_truth.shape)
    thetas, cntrst_thetas = init_start_time(
        key_init, n_samples, n_samples_cntrst, ground_truth.shape
    )
    weights_0 = jnp.zeros((n_samples,))
    weights_c_0 = jnp.zeros((n_samples_cntrst,))
    implicit_state = ImplicitState(
        thetas, weights_0, cntrst_thetas, weights_c_0, design, opt_state
    )

    # stock in joint_y all measurements
    joint_y = y
    mask_history = mask.make(design)
    noiser = jax.jit(jax.vmap(sde.path, in_axes=(0, None, 0)))
    
    l2_hist = []
    ssim_hist = []
    
    for n_meas in range(num_meas):
        key_noise, key_opt, key_gen = jax.random.split(key_step, 3)
        # make noised path for measurements
        keys_noise = jax.random.split(key_noise, n_t)
        state_0 = SDEState(joint_y, jnp.zeros_like(y))
        past_y = noiser(keys_noise, state_0, ts)
        past_y = SDEState(past_y.position[::-1], past_y.t)
        # plotter_line(past_y.position)
        optimal_state = generate_cond_sampleV2(
            joint_y, mask_history, key_opt, cond_sde, ground_truth.shape, n_t, n_samples
        )[0]
        thetas, weights = optimal_state

        # random design
        design = jax.random.uniform(key_step, (2,), minval=0, maxval=28)

        # make new measurement
        new_measurement = cond_sde.mask.measure(design, ground_truth)
        measurement_history = measurement_history.at[n_meas].set(new_measurement)

        # add measured data to joint_y and update history of mask location
        joint_y = cond_sde.mask.restore(design, joint_y, new_measurement)
        mask_history = cond_sde.mask.restore(
            design, mask_history, cond_sde.mask.make(design)
        )
        # plt.imshow(mask_history, cmap="gray")

        jax.experimental.io_callback(
            plotter_random,
            None,
            ground_truth,
            joint_y,
            design,
            thetas.position,
            weights,
            n_meas,
        )

        key_step, _ = jax.random.split(key_step)

        l2 = metric_l2(ground_truth, thetas.position, weights)
        ssim_rand = metric_ssim(ground_truth, thetas.position, weights)
        print('l2:', l2, 'ssim:', ssim_rand)
        l2_hist.append(l2)
        ssim_hist.append(ssim_rand)

    return ground_truth, optimal_state, joint_y, l2_hist, ssim_hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rng_key", type=int, default=0)
    parser.add_argument("--random", type=bool, default=True)
    parser.add_argument("--num_meas", type=int, default=25)
    parser.add_argument("--prefix", type=str, default="") 
    parser.add_argument("--single_step_ub", type=float, default=1.4)
    parser.add_argument("--inner_gradient_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="gpu")

    args = parser.parse_args()

    # Change device
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    num_meas = args.num_meas
    key_int = args.rng_key
    random = args.random
    single_step_ub = args.single_step_ub
    inner_gradient_steps = args.inner_gradient_steps
    
    rng_key = jax.random.PRNGKey(key_int)
    dir_path = f"runs/{args.prefix}/{key_int}_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}"

    logging_path_theta = f"{dir_path}/theta"
    logging_path_contrastive = f"{dir_path}/contrastive"
    os.makedirs(logging_path_theta, exist_ok=True)
    os.makedirs(logging_path_contrastive, exist_ok=True)
    plotter_theta = partial(log_samples, logging_path=logging_path_theta, size=SIZE)
    plotter_contrastive = partial(
        log_samples, logging_path=logging_path_contrastive, size=SIZE
    )

    if random:
        logging_path_random = f"{dir_path}/random"
        os.makedirs(logging_path_random, exist_ok=True)
        plotter_r = partial(plotter_random, logging_path=logging_path_random, size=SIZE)
        ground_truth, state_random, y_random, l2_hist_random, ssim_hist_random = main_random(rng_key, num_meas, plotter_r)

    ground_truth, state, y, l2_hist, ssim_hist = main(rng_key, num_meas, plotter_theta, plotter_contrastive,
                                                     single_step_ub=single_step_ub, inner_gradient_steps=inner_gradient_steps)
    if random:
        plot_comparison(ground_truth, state_random, state, y_random, y, dir_path)
    
    # Save the metrics 
    save_results(l2_hist, ssim_hist, f"runs/{args.prefix}/", key_int)
    if random:
        save_results(l2_hist_random, ssim_hist_random, f"runs/{args.prefix}/", key_int, random=True)
