# random vector only generates for the first layer; readout is one relu layer; matrices in Kuramoto layers are not constrained

import jax.numpy as jnp
import equinox as eqx

from jax.lax import scan, dot_general
from jax import random
from jax.nn import relu, log_softmax

class Kuramoto_layer(eqx.Module):
    Omega: jnp.array
    J: jnp.array
    gamma: jnp.array

    def __init__(self, key, N_f, D):
        keys = random.split(key, 2)
        self.Omega = random.normal(keys[0], (N_f, D, D)) / jnp.sqrt(2 / (N_f*D + D))
        self.J = random.normal(keys[1], (N_f, N_f, D, D)) / jnp.sqrt(1 / (N_f*D))
        self.gamma = 1e-2*jnp.ones((1,))

    def Kuramoto_update(self, x, c):
        o = dot_general(self.Omega, x, (((2,), (1,)), ((0,), (0,))))
        p = c + dot_general(self.J, x, (((1, 3), (0, 1)), ((), ())))
        p = p - x*jnp.sum(p*x, axis=1, keepdims=True)
        res = x + self.gamma*(o + p)
        res = res / jnp.linalg.norm(res, axis=1, keepdims=True)
        return res

    def __call__(self, x, c, N):
        for _ in range(N):
            x = self.Kuramoto_update(x, c)
        return x

class relu_readout(eqx.Module):
    U: jnp.array
    b: jnp.array

    def __init__(self, key, N_f, D):
        self.U = random.normal(key, (N_f*D, N_f, D, D)) / jnp.sqrt(2 / (N_f*D + N_f*D*D))
        self.b = jnp.zeros((N_f, D))

    def __call__(self, x):
        res = dot_general(self.U, x, (((1, 3), (0, 1)), ((), ())))
        res = relu(jnp.linalg.norm(res, axis=1).reshape(x.shape) + self.b)
        return res

class AKOrN_v1(eqx.Module):
    Kuramoto_layers: list
    readout_layers: list
    encoder: list
    decoder: list

    def __init__(self, key, N_features, D, N_layers):
        keys = random.split(key, 4)
        self.Kuramoto_layers = [Kuramoto_layer(key, N_features[1], D) for key in random.split(keys[0], N_layers)]
        self.readout_layers = [relu_readout(key, N_features[1], D) for key in random.split(keys[1], N_layers)]
        self.encoder = [random.normal(keys[2], (N_features[1]*D, N_features[0])) * jnp.sqrt(2 / (N_features[1]*D + N_features[0])), jnp.zeros((N_features[1], D))]
        self.decoder = [random.normal(keys[3], (N_features[2], N_features[1]*D)) * jnp.sqrt(2 / (N_features[1]*D + N_features[2])), jnp.zeros((N_features[2], ))]

    def __call__(self, c, N, key):
        c = (self.encoder[0] @ c).reshape(self.encoder[1].shape) + self.encoder[1]
        x = random.normal(key, c.shape)
        x = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        x = self.Kuramoto_layers[0](x, c, N)
        c = self.readout_layers[0](x)
        for i in range(1, len(self.readout_layers)):  
            x = self.Kuramoto_layers[i](x, c, N)
            c = self.readout_layers[i](x)
        c = self.decoder[0] @ c.reshape(-1,) + self.decoder[1]
        return c

class AKOrN_v1_classification(AKOrN_v1):
    def __call__(self, x, N, key):
        y = super().__call__(x, N, key)
        return log_softmax(y)