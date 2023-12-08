from jax.nn import selu
from jax import numpy as jnp



def forward_pass(params, input):
    """
    Args
    ----------
    params : list
        Parameters of the network, with one list element per layer. 
        See notes below on network initialization.

    input : ndarray
        Array of shape (batch_size, n_features)
    
    Returns
    -------
    eta : ndarray 
        Array of shape (batch_size, n_targets)
    """
    activations = input
    
    # Hidden Layers
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = selu(outputs)
        
    #  Output Layers
    w_output, b_output = params[-1] 
    eta = jnp.dot(w_output, activations) + b_output 
    return eta


# ======

from jax import vmap as vmap

batched_forward_pass = vmap(forward_pass, in_axes=(None, 0))


from jax import random as jran


def get_random_layer_params(m, n, ran_key, scale=0.01):
    """Helper function to randomly initialize 
    weights and biases using the JAX-defined randoms."""
    w_key, b_key = jran.split(ran_key)
    ran_weights = scale * jran.normal(w_key, (n, m))
    ran_biases = scale * jran.normal(b_key, (n,)) 
    return ran_weights, ran_biases


def get_init_network_params(sizes, ran_key):
    """Initialize all layers for a fully-connected neural network."""
    keys = jran.split(ran_key, len(sizes))
    return [get_random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def get_network_layer_sizes(n_features, n_targets, n_layers, n_neurons_per_layer):
    dense_layer_sizes = [n_neurons_per_layer]*n_layers
    layer_sizes = [n_features, *dense_layer_sizes, n_targets]
    return layer_sizes

# =
SEED = 0
ran_key = jran.PRNGKey(SEED)

num_features, num_targets = 1000, 1
num_layers, num_neurons_per_layer = 1, 50

layer_sizes = get_network_layer_sizes(num_features, num_targets, num_layers, num_neurons_per_layer)

init_params = get_init_network_params(layer_sizes, ran_key)

# Verify functions perform as expected

ran_key, func_key = jran.split(ran_key)
random_feature_array = jran.uniform(func_key, minval=0, maxval=1, shape=(num_features, ))
single_pred = forward_pass(init_params, random_feature_array)
print(single_pred)

num_batches = 5
ran_key, batch_key = jran.split(ran_key)
random_batch_of_features = jran.uniform(func_key, minval=0, maxval=1, shape=(num_batches, num_features))
batch_preds = batched_forward_pass(init_params, random_batch_of_features)
