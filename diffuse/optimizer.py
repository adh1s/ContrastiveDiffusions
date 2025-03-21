from functools import partial
from typing import NamedTuple

import jax
import jax.experimental
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray
from optax import GradientTransformation

from diffuse.conditional import CondSDE
from diffuse.sde import SDEState
from diffuse.inference import (
    calculate_past_contribution_score,
    calculate_drift_expt_post,
    logpdf_change_expected,
    logpdf_change_y,
    particle_step,
    logprob_y,
)


class ImplicitState(NamedTuple):
    thetas: Array
    weights: Array
    cntrst_thetas: Array
    weights_c: Array
    design: Array
    opt_state: optax.OptState


def information_gain(theta: Array, cntrst_theta: Array, design: Array, cond_sde):
    r"""
    Information gain estimator
    Estimator \sum_i log p(y_i | theta_i, design) - \sum_j w_{ij} log p(y_i | theta_j, design)
    """
    # sample y ~ p(y | \theta, \design)
    y_ref = cond_sde.mask.measure(design, theta)
    # calculate log p(y | \theta, \design)
    logprob_ref = logprob_y(theta, y_ref, design, cond_sde)
    # calculate log p(y | \theta, \design) for all thetas
    logprob_target = jax.vmap(logprob_y, in_axes=(None, 0, None, None))(
        cntrst_theta, y_ref, design, cond_sde
    )

    logprob_means = jnp.mean(logprob_target, axis=0, keepdims=True)
    # take away the mean to avoid numerical instability
    log_weights = jax.lax.stop_gradient(logprob_target - logprob_means)
    _norm = jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    log_weights = log_weights - _norm

    weighted_logprobs = jnp.mean(log_weights + logprob_target, axis=1)

    return (logprob_ref - weighted_logprobs).mean(), y_ref


def update_joint(
    sde_state: SDEState,
    ys: Array,
    ys_next: Array,
    key: PRNGKeyArray,
    cond_sde: CondSDE,
    mask_history: Array,
    design: Array,
    dt: float,
):
    r"""
    simulate \theta according to conditional sde:
    \(
    \theta_{t+dt} = \[ -\beta(t) / 2 \theta_t - \beta(t) \nabla_\theta log p(y^t_past | \theta_t, \xi_past) - \beta(t) \nabla_\theta \log p(\theta_t) \]dt + \sqrt(\beta(t) )DWt
    \)
    """
    drift_past = calculate_past_contribution_score(
        cond_sde, sde_state, mask_history, ys
    )
    logpdf = partial(
        logpdf_change_y,
        y_next=ys_next,
        design=mask_history,
        cond_sde=cond_sde,
        dt=dt,
    )
    positions = particle_step(sde_state, key, drift_past, cond_sde, dt, logpdf)

    return positions


def update_expected_posterior(
    cntrst_sde_state: SDEState,
    ys: Array,
    ys_next: Array,
    y_measured: Array,
    key: PRNGKeyArray,
    cond_sde: CondSDE,
    mask_history: Array,
    design: Array,
    dt: float,
):
    r"""
    simulate \theta according to conditional sde for expected posterior:
    .. math::
        $$
        \theta_{t+dt} = \[ -\beta(t) / 2 \theta_t - \beta(t) \nabla_\theta log p(y_past^t | \theta_t, \xi_past) - \beta(t) \sum_1^N \nabla_\theta log p(y_i^t | \theta_t, \xi) / N - \beta(t) \nabla_\theta \log p(\theta_t) \]dt + \sqrt(\beta(t) )DWt
        $$
    """
    drift_past = calculate_past_contribution_score(
        cond_sde, cntrst_sde_state, mask_history, y_measured
    )
    # why is there no SMC for the old quantities?
    drift_y = calculate_drift_expt_post(cond_sde, cntrst_sde_state, design, ys)
    logpdf = partial(
        logpdf_change_expected,
        y_next=ys_next,
        design=mask_history,
        cond_sde=cond_sde,
        dt=dt,
    )
    positions, weights = particle_step(
        cntrst_sde_state, key, drift_y + drift_past, cond_sde, dt, logpdf
    )

    return positions, weights


def calculate_and_apply_gradient(
    thetas: Array,
    cntrst_thetas: Array,
    design: Array,
    cond_sde: CondSDE,
    optx_opt: GradientTransformation,
    opt_state: optax.OptState,
):
    grad_xi_score = jax.grad(information_gain, argnums=2, has_aux=True)
    # return the ys (which are calculated in information_gain)
    # could use more thetas (increase N from 1?) for a better estimate?
    grad_xi, ys = grad_xi_score(thetas[-1], cntrst_thetas, design, cond_sde)
    updates, new_opt_state = optx_opt.update(grad_xi, opt_state, design)
    new_design = optax.apply_updates(design, updates)

    return new_design, new_opt_state, ys


def impl_step(
    state: ImplicitState,
    rng_key: PRNGKeyArray,
    past_y: Array,
    mask_history: Array,
    cond_sde: CondSDE,
    optx_opt: GradientTransformation,
    ts: Array,
    dt: float,
):
    """
    Implicit step with parallel update
    """

    thetas, cntrst_thetas, design, opt_state = state
    sde_state = SDEState(thetas, ts)
    cntrst_sde_state = SDEState(cntrst_thetas, ts)

    key_theta, key_cntrst = jax.random.split(rng_key)

    def step_joint(sde_state, ys, ys_next, key):
        positions, weights = update_joint(
            sde_state, ys, ys_next, key, cond_sde, mask_history, design, dt
        )
        return positions, weights

    keys_time = jax.random.split(key_theta, ts.shape[0] - 1)
    sde_state = jax.tree_map(lambda x: x[1:], sde_state)
    position, weights = jax.vmap(step_joint)(
        sde_state, past_y.position[:-1], past_y.position[1:], keys_time
    )
    thetas = jnp.concatenate([thetas[:2], position[:-1]])

    ys = jax.vmap(cond_sde.mask.measure, in_axes=(None, 0))(design, thetas)

    def step_expected_posterior(cntrst_sde_state, ys, ys_next, y_measured, key):
        _, t = cntrst_sde_state
        positions, weights = update_expected_posterior(
            cntrst_sde_state,
            ys,
            ys_next,
            y_measured,
            key,
            cond_sde,
            mask_history,
            design,
            dt,
        )
        return positions, weights

    keys_time_c = jax.random.split(key_cntrst, ts.shape[0] - 1)
    cntrst_sde_state = jax.tree_map(lambda x: x[1:], cntrst_sde_state)
    position, weights_c = jax.vmap(step_expected_posterior)(
        cntrst_sde_state, ys[:-1], ys[1:], past_y.position[1:], keys_time_c
    )
    cntrst_thetas = jnp.concatenate([cntrst_thetas[:2], position[:-1]])

    # get EIG gradient estimator
    design, opt_state, ys = calculate_and_apply_gradient(
        thetas[-1], cntrst_thetas[-1], design, cond_sde, optx_opt, opt_state
    )

    return ImplicitState(thetas, weights, cntrst_thetas, weights_c, design, opt_state)


def impl_one_step(
    state: ImplicitState,
    rng_key: PRNGKeyArray,
    past_y: SDEState,
    past_y_next: SDEState,
    mask_history: Array, # used for computing the posterior p(\theta|D_{k-1})
    cond_sde: CondSDE,
    optx_opt: GradientTransformation, 
    single_step_ub,
    inner_gradient_steps, 
):
    """
    Implicit step with one step update
    Must use same optimization steps as time steps
    """
    dt = past_y_next.t - past_y.t
    thetas, weights, cntrst_thetas, weights_c, design, opt_state = state
    sde_state = SDEState(thetas, past_y.t)
    cntrst_sde_state = SDEState(cntrst_thetas, past_y.t)

    # update joint distribution
    key_joint, key_cntrst = jax.random.split(rng_key, 2)

    def step_joint(sde_state, ys, ys_next, key):
        _, t = sde_state
        positions, weights = update_joint(
            sde_state, ys, ys_next, key, cond_sde, mask_history, design, dt
        )

        return (SDEState(positions, t + dt), weights), (positions, weights)

    ((thetas, weights), _) = step_joint(
        sde_state, past_y.position, past_y_next.position, key_joint
    )

    ys = cond_sde.mask.measure(design, thetas.position)
    ys_next = cond_sde.path(rng_key, SDEState(ys, past_y.t), past_y_next.t).position
    # Above amounts to sampling p(\theta|d_{k-1})p(y_k|\theta, d_{k-1}) = p(\theta, y_k|d_{k-1}) 

    def step_expected_posterior(cntrst_sde_state, ys, ys_next, y_measured, key):
        _, t = cntrst_sde_state
        positions, weights = update_expected_posterior(
            cntrst_sde_state,
            ys,
            ys_next,
            y_measured,
            key,
            cond_sde,
            mask_history,
            design,
            dt,
        )

        return (SDEState(positions, t + dt), weights), (positions, weights)

    ((cntrst_thetas, weights_c), _) = step_expected_posterior(
        cntrst_sde_state, ys, ys_next, past_y.position, key_cntrst
    )

    # Get EIG gradient estimator
    n = 50
    best_idx = jnp.argsort(weights)[-n:][::-1]
    best_idx_c = jnp.argsort(weights_c)[-n:][::-1]

    def step(state, itr):
        design, opt_state = state
        design, opt_state, _ = calculate_and_apply_gradient(
            thetas.position,
            cntrst_thetas.position,
            design,
            cond_sde,
            optx_opt,
            opt_state,
        )
        # taking more than 1 step here feels like a really dodgy thing to do?
        design = optax.projections.projection_box(design, 0.0, 28.0)
        return (design, opt_state), None

    design, opt_state = jax.lax.cond(
        past_y.t > single_step_ub, 
        lambda _: jax.lax.scan(step, (design, opt_state), jnp.arange(inner_gradient_steps,))[0], # past_y.t > single_step_ub (inner_steps steps)
        lambda _: calculate_and_apply_gradient( # past_y.t <= single_step_ub  (1 step)
            thetas.position,
            cntrst_thetas.position,
            design,
            cond_sde,
            optx_opt,
            opt_state,
        )[:2],
        None,
    )

    return ImplicitState(
        thetas.position, weights, cntrst_thetas.position, weights_c, design, opt_state
    )


def impl_full_scan(
    state: ImplicitState,
    rng_key: PRNGKeyArray,
    past_y: Array,
    mask_history: Array,
    cond_sde: CondSDE,
    optx_opt: GradientTransformation,
    ts: Array,
    dt: float,
):
    """
    Implicit step with full scan
    """

    thetas, cntrst_thetas, design, opt_state = state
    n_particles = thetas.shape[0]
    n_cntrst_particles = cntrst_thetas.shape[0]
    sde_state = SDEState(thetas, 0.0)
    cntrst_sde_state = SDEState(cntrst_thetas, 0.0)

    # update joint distribution
    key_theta, key_cntrst, key_y = jax.random.split(rng_key, 3)

    def step_joint(state, itr):
        sde_state, weights = state
        _, t = sde_state
        ys, ys_next, key = itr
        positions, weights = update_joint(
            sde_state, ys, ys_next, key, cond_sde, mask_history, dt
        )

        return (SDEState(positions, t + dt), weights), (positions, weights)

    keys_time = jax.random.split(key_theta, ts.shape[0] - 1)

    ((thetas, _), weights), hist = jax.lax.scan(
        step_joint,
        (sde_state, jnp.zeros((n_particles,))),
        (past_y.position[:-1], past_y.position[1:], keys_time),
    )

    # get ys
    # ys output should be (n_t, n_particles, ...)
    # and time noise -> measurement
    thetas_hist, _ = hist
    thetas_hist = jnp.concatenate([thetas[None], thetas_hist])

    ys = jax.vmap(cond_sde.mask.measure, in_axes=(None, 0))(design, thetas_hist)

    # update expected posterior
    def step_expected_posterior(state, itr):
        cntrst_sde_state, weights = state
        _, t = cntrst_sde_state
        ys, ys_next, y_measured, key = itr
        positions, weights = update_expected_posterior(
            cntrst_sde_state,
            ys,
            ys_next,
            y_measured,
            key,
            cond_sde,
            mask_history,
            design,
            dt,
        )

        return (SDEState(positions, t + dt), weights), (positions, weights)

    keys_time_c = jax.random.split(key_cntrst, ts.shape[0] - 1)

    ((cntrst_thetas, _), weights), hist = jax.lax.scan(
        step_expected_posterior,
        (cntrst_sde_state, jnp.zeros((n_cntrst_particles,))),
        (ys[:-1], ys[1:], past_y.position[1:], keys_time_c),
    )

    # get EIG gradient estimator
    design, opt_state, ys = calculate_and_apply_gradient(
        thetas, cntrst_thetas, design, cond_sde, optx_opt, opt_state
    )

    return ImplicitState(thetas, cntrst_thetas, design, opt_state)
