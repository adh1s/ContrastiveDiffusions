from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, PRNGKeyArray
from functools import partial

from diffuse.sde import SDE, SDEState, euler_maryama_step
from diffuse.images import SquareMask


@register_pytree_node_class
class CondState(NamedTuple):
    x: jnp.ndarray  # Current Markov Chain State x_t
    y: jnp.ndarray  # Current Observation Path y_t
    xi: jnp.ndarray  # Measured position of y_t | x_t, xi_t
    t: float

    def tree_flatten(self):
        children = (self.x, self.y, self.xi, self.t)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@dataclass
class CondSDE(SDE):
    """
    cond_sde.mask.restore act as the matrix A^T
    """

    mask: SquareMask
    tf: float
    score: Callable[[Array, float], Array]

    def reverse_drift(self, state: SDEState) -> Array:
        x, t = state
        beta_t = self.beta(self.tf - t)
        s = self.score(x, self.tf - t)
        return 0.5 * beta_t * x + beta_t * s

    def reverse_diffusion(self, state: SDEState) -> Array:
        x, t = state
        return jnp.sqrt(self.beta(self.tf - t))

    def logpdf(self, obs: Array, state_p: CondState, dt: float):
        """
        y_{k-1} | y_{k}, x_k ~ N(.| y_k + rev_drift*dt, sqrt(dt)*rev_diff)
        Args:
            obs (Array): The observation y_{k-1}.
            state_p (CondState): The previous state containing:
                - x_p (Array): The particle state x_k.
                - y_p (Array): The observation y_k.
                - xi_p (Array): The measurement position.
                - t_p (float): The time step.
            dt (float): The time step size.

        Returns:
            float: The log probability density of the observation.
        """
        x_p, y_p, xi, t_p = state_p
        # mean = y_p + cond_reverse_drift(state_p, self) * dt
        mean = y_p + self.mask.measure(xi, cond_reverse_drift(state_p, self)) * dt
        std = jnp.sqrt(dt) * cond_reverse_diffusion(state_p, self)

        return jax.scipy.stats.norm.logpdf(obs, mean, std).sum()

    def cond_reverse_step(
        self, state: CondState, dt: float, key: PRNGKeyArray
    ) -> CondState:
        """
        x_{k-1} | x_k, y_k ~ N(.| x_k + rev_drift*dt, sqrt(dt)*rev_diff)
        """
        x, y, xi, t = state

        def revese_drift(state):
            x, t = state
            return cond_reverse_drift(CondState(x, y, xi, t), self)

        def reverse_diffusion(state):
            x, t = state
            return cond_reverse_diffusion(CondState(x, y, xi, t), self)

        x, _ = euler_maryama_step(
            SDEState(x, t), dt, key, revese_drift, reverse_diffusion
        )
        y = self.mask.measure(xi, x)
        return CondState(x, y, xi, t - dt)
    
@dataclass
class CondSDEImplicit(SDE):
    """
    Take in an general (differentiable) likelihood function
    """
    likelihood_fn: Callable
    tf: float
    score: Callable[[Array, float], Array]

    def reverse_drift(self, state: SDEState) -> Array:
        x, t = state
        beta_t = self.beta(self.tf - t)
        s = self.score(x, self.tf - t)
        return 0.5 * beta_t * x + beta_t * s

    def reverse_diffusion(self, state: SDEState) -> Array:
        x, t = state
        return jnp.sqrt(self.beta(self.tf - t))
    
    def reverso(
        self, key: PRNGKeyArray, state_tf: SDEState, score: Callable, dts: float, y, xi, guidance_scale=1.0
    ) -> SDEState:
        x_tf, tf = state_tf
        state_tf_0 = SDEState(x_tf, jnp.array([0.0]))

        def reverse_drift(likelihood_fn, state, y, xi):
            x, t = state
            beta_t = self.beta(tf - t)
            int_b = self.beta.integrate(tf - t, 0) 
            alpha_t = jnp.exp(-0.5 * int_b)
            s = score(x, tf - t)
            xi = jnp.array(xi)
            y = jnp.array(y)

            def guidance_likelihood(x): # Tweedie's formula
                x_0 = (1 / (jnp.sqrt(alpha_t)))*(x + (1-alpha_t)*score(x, tf - t))
                likelihoods = jax.vmap(lambda y, xi: likelihood_fn(y, x_0, xi))(y, xi)
                return jnp.sum(likelihoods)

            guidance_grad = jax.grad(guidance_likelihood)(x) * guidance_scale
            
            return 0.5 * beta_t * x + beta_t * (s + guidance_grad)

        def reverse_diffusion(state):
            x, t = state
            return jnp.sqrt(self.beta(tf - t))
        
        step = partial(
            euler_maryama_step_cond, drift=reverse_drift, diffusion=reverse_diffusion, likelihood_fn=self.likelihood_fn, y=y, xi=xi
        )

        def body_fun(state, tup):
            dt, key = tup
            next_state = step(state, dt, key)
            return next_state, next_state

        n_dt = dts.shape[0]
        keys = jax.random.split(key, n_dt)
        state_f, history = jax.lax.scan(body_fun, state_tf_0, (dts, keys))
        history = jax.tree_map(
            lambda arr, x: jnp.concatenate([arr[None], x]), state_tf_0, history
        )
        return state_f, history

def euler_maryama_step_cond(
    state: SDEState, dt: float, key: PRNGKeyArray, drift: Callable, diffusion: Callable, likelihood_fn: Callable, y: Array, xi: Array
) -> SDEState:
    dx = drift(likelihood_fn, state, y, xi) * dt + diffusion(state) * jax.random.normal(
        key, state.position.shape
    ) * jnp.sqrt(dt)
    return SDEState(state.position + dx, state.t + dt)

# TODO:
# Implement the design space in UFG (https://arxiv.org/abs/2409.15761)
# e.g. guidance_scale_schedule, corrector-steps, average over noise at t=0-

def impl_log_likelihood(encoder, weights, y, x, xi, b=0, a=1, m=0.0001, measurement_noise=0.5):
    xi = jnp.reshape(xi, (1, -1))
    x = jnp.reshape(x, (1, -1))

    xi_mu, _ = encoder.apply(weights, xi)
    x_mu, _ = encoder.apply(weights, x)

    dist = jnp.linalg.norm(x_mu - xi_mu, axis=-1, ord=2)
    mu = jnp.log(b + (a / (m + dist)))
    log_val = jax.scipy.stats.norm.logpdf(jnp.log(y), mu, measurement_noise)
    assert log_val.shape[0] == 1
    return log_val.sum()

# Note that x is \theta
def cond_reverse_drift(state: CondState, cond_sde: CondSDE) -> Array:
    x, y, xi, t = state
    drift_x = cond_sde.reverse_drift(SDEState(x, t))
    beta_t = cond_sde.beta(cond_sde.tf - t)
    meas_x = cond_sde.mask.measure(xi, x) # y^t = [A_{xi}]x
    alpha_t = jnp.exp(cond_sde.beta.integrate(0.0, t)) # does this need to be (tf - t)?
    drift_y = (
        beta_t * cond_sde.mask.restore(xi, jnp.zeros_like(x), y - meas_x) / alpha_t
    )
    return drift_x + drift_y

def cond_reverse_diffusion(state: CondState, cond_sde: CondSDE) -> Array:
    x, y, xi, t = state
    img = cond_sde.mask.restore(xi, x, y)
    return cond_sde.reverse_diffusion(SDEState(img, t))

def euler_maryama_step_array(
    state: SDEState, dt: float, key: PRNGKeyArray, drift: Array, diffusion: Array
) -> SDEState:
    dx = drift * dt + diffusion * jax.random.normal(
        key, state.position.shape
    ) * jnp.sqrt(dt)
    return SDEState(state.position + dx, state.t + dt)
