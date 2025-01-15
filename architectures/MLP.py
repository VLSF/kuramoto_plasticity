import jax.numpy as jnp
import equinox as eqx

from jax import random
from jax.nn import relu, log_softmax

class MLP(eqx.Module):
    weights: list
    biases: list

    def __init__(self, key, N_features, N_layers):
        shapes = [N_features[0],] + [N_features[1],]*N_layers + [N_features[-1],]
        keys = random.split(key, (len(shapes)-1))
        self.weights = [random.normal(key, (s_out, s_in)) * 2 / jnp.sqrt(s_out + s_in) for s_in, s_out, key in zip(shapes[:-1], shapes[1:], keys)]
        self.biases = [jnp.zeros((s_out,)) for s_out, in zip(shapes[1:])]

    def __call__(self, u, N, key):
        v = self.weights[0] @ u + self.biases[0]
        for w, b in zip(self.weights[1:], self.biases[1:]):
            v = relu(v)
            v = w @ v + b
        return v

class MLP(MLP):
    def __call__(self, u, N, key):
        v = super().__call__(u, N, key)
        return log_softmax(v)