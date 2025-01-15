import jax.numpy as jnp
import optax
import equinox as eqx

from jax import random, vmap

def compute_loss_(model, feature, target, N, key):
    log_probs = model(feature, N, key)
    loss = - log_probs[target]
    return loss

def compute_loss(model, features, targets, N, key):
    keys = random.split(key, targets.shape[0])
    return jnp.mean(vmap(compute_loss_, in_axes=(None, 0, 0, None, 0))(model, features, targets, N, keys))

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def make_step_scan(carry, n, optim, N):
    model, features, targets, opt_state, key = carry
    key, key_ = random.split(key)
    loss, grads = compute_loss_and_grads(model, features[n], targets[n], N, key_)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, opt_state, key], loss

def compute_error(carry, n, N):
    model, features, targets, key = carry
    key, key_ = random.split(key)
    predicted_class = jnp.argmax(model(features[n], N, key_))
    is_correct = predicted_class == targets[n]
    return [model, features, targets, key], is_correct