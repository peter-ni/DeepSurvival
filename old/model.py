import common_utils

target_jnp = common_utils.target_jnp
clin_jnp = common_utils.clin_jnp
sv_jnp = common_utils.sv_jnp
sv_conf_jnp = common_utils.sv_conf_jnp
cna_jnp = common_utils.cna_jnp

target_vars = common_utils.target_vars
clin_vars = common_utils.clin_vars
sv_vars = common_utils.sv_vars
sv_conf_vars = common_utils.sv_conf_vars
cna_vars = common_utils.cna_vars

from jax.nn import selu
from jax import numpy as jnp
from jax import random as jran
from jax import grad,jit, vmap



# Parameter Initialization Helper Functions

from common_utils import get_random_layer_params, get_init_network_params, get_network_layer_sizes

# Forward Pass
def forward_pass(params, input):
    activations = input
    
    # Hidden Layers
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = selu(outputs)
        
    #  Output Layer
    w_output, b_output = params[-1] 
    eta = jnp.dot(w_output, activations) + b_output 

    return eta

batched_forward_pass = vmap(forward_pass, in_axes = (None, 0))

# Loss Function
from common_utils import weibull_pdf, one_minus_weibull_cdf

# @jit
# def weibull_loss(params, base_haz_params, input, targets):
#     eta = batched_forward_pass(params, input)
#     lam = base_haz_params[0] * jnp.exp(- eta / base_haz_params[1])
#     k = base_haz_params[1]

#     dead = input[:,1] == 1
#     censored = input[:,1] == 0

#     likelihood = jnp.zeros_like(input[:,0])
#     likelihood[dead] = jnp.log(weibull_pdf(t = input[dead,0], lam = lam, k = k))
#     likelihood[censored] = jnp.log(one_minus_weibull_cdf(t = input[censored,0], lam = lam, k = k))
#     neg_log_likelihood = - jnp.sum(likelihood)
#     return neg_log_likelihood


def weibull_loss(params, base_haz_params, input, targets, batch_size, dead_idx, censor_idx):
    eta = batched_forward_pass(params, input)
    lam = base_haz_params[0] * jnp.exp(- eta / base_haz_params[1])
    k = base_haz_params[1]

    likelihood = jnp.zeros(batch_size, dtype = jnp.float32)
    clipping = 1e-6

    # Update the likelihood array for 'dead' cases
    likelihood = likelihood.at[dead_idx].set(jnp.log(weibull_pdf(t = targets[dead_idx,0], lam = lam[dead_idx], k = k) + clipping))
    # Update the likelihood array for 'censored' cases
    likelihood = likelihood.at[censor_idx].set(jnp.log(one_minus_weibull_cdf(t = targets[censor_idx,0], lam = lam[censor_idx], k = k) + clipping))

    neg_log_likelihood = - jnp.sum(likelihood)
    return neg_log_likelihood

grad_weibull_loss = grad(weibull_loss)


# Model Architecture
num_features, num_targets = 1658, 1
num_layers, num_neurons_per_layer = 1, 50
layer_sizes = get_network_layer_sizes(num_features, num_targets, num_layers, num_neurons_per_layer)
batch_size = 128

# Parameter Initialization
SEED = 0
rng_key = jran.PRNGKey(SEED)
init_params = get_init_network_params(layer_sizes, rng_key)
init_base_haz_params = [float(jran.uniform(rng_key, minval = 1, maxval = 1.5)), 1.0]




# Batch Sample Test
rng_key, batch_key = jran.split(rng_key)
rng_key, batch_target_key = jran.split(rng_key)
batch_sample = jran.uniform(batch_key, minval = 0, maxval = 1, shape = (batch_size, num_features))
batch_sample_time = jran.uniform(batch_key, minval = 0, maxval = 100, shape = (batch_size, 1))
batch_sample_status = jnp.array(jran.bernoulli(batch_key, p = 0.5, shape = (batch_size, 1)), dtype = jnp.float32)
batch_sample_targets = jnp.column_stack((batch_sample_time, batch_sample_status))
# batch_preds = batched_forward_pass(init_params, batch_sample)

batch_dead_idx = jnp.where(batch_sample_targets[:,1] == 1)[0]
batch_censor_idx = jnp.where(batch_sample_targets[:,1] == 0)[0]

# Loss Function Testing

batch_sample_loss = weibull_loss(params = init_params,
                                 base_haz_params = init_base_haz_params,
                                 input = batch_sample,
                                 targets = batch_sample_targets,
                                 batch_size = batch_size,
                                 dead_idx = batch_dead_idx,
                                 censor_idx = batch_censor_idx)

grad_weibull_loss = grad(weibull_loss, argnums = (0,1))

batch_sample_grad_loss = grad_weibull_loss(init_params,
                                           init_base_haz_params,
                                           batch_sample,
                                           batch_sample_targets,
                                           batch_size,
                                           batch_dead_idx,
                                           batch_censor_idx)

print(batch_sample_loss)




                  
# Training
# ===============================

# Optimizer
import optax
import functools

start_learning_rate = 1e-1
optimizer = optax.adam(learning_rate = start_learning_rate)
opt_state = optimizer.init((init_params, init_base_haz_params))


# Training Loop
num_epochs = 1

for __ in range(num_epochs):
    grads = grad_weibull_loss(init_params,
                              init_base_haz_params,
                              batch_sample,
                              batch_sample_targets,
                              batch_size,
                              batch_dead_idx,
                              batch_censor_idx)














