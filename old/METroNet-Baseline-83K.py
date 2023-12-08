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



# Dependencies
import numpy as np
import jax
import jax.numpy as jnp
import optax
import torch
import torch.utils.data as data


from flax import linen as nn
from flax.training import train_state
from jax import random





# Define baseline architecture
class baseline_arch(nn.Module):
    num_hidden : int = 50
    num_outputs : int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(50)(x)
        x = nn.relu(x)
        x = nn.Dense(features = self.num_outputs)(x)
        return x



rng_key = random.PRNGKey(0)
dataset = np.array(jnp.concatenate([target_jnp, clin_jnp, sv_jnp, cna_jnp], axis = 1))
input_shape = (dataset.shape[1] - 2,)



baseline_model = baseline_arch()
print(baseline_model.tabulate(rng_key, jnp.ones(input_shape)))





# Initialize Parameters and Apply to Model
rng, input_rng, init_rng = jax.random.split(rng_key, 3)
random_input = jax.random.normal(input_rng, (8, dataset.shape[1]))

model_params_init = baseline_model.init(init_rng, random_input)
baseline_model_init = baseline_model.apply(model_params_init, random_input)


# Data Loader
data_loader = data.DataLoader(dataset, batch_size = 128, shuffle = True)

# Loss Function
# output: subject-specific eta_i
# base_haz: population-level shape/scale parameters of baseline survival 

from common_utils import weibull_pdf, one_minus_weibull_cdf

def weibull_loss(output, base_haz, target, batch):
    dead = batch[:,1] == 1
    censored = batch[:,1] == 0
    lam = base_haz[0] * torch.exp(- output / base_haz[1])
    k = base_haz[1]

    likelihood = torch.zeros_like(batch[:,0])
    likelihood[dead] = torch.log(weibull_pdf(t = batch[dead,0], lam = lam, k = k))
    likelihood[censored] = torch.log(one_minus_weibull_cdf(t = batch[censored,0], lam = lam, k = k))
    neg_log_likelihood = -torch.sum(likelihood)
    return neg_log_likelihood
    

# Testing Loss Function
mock_batch = jnp.array([
    [5.0, 1],  # dead
    [4.0, 0],  # censored
    [6.0, 1],  # dead
    [7.0, 0]   # censored
])

base_haz_params = jnp.array([1.0, 2.0])




    



# Optimizer
optimizer = optax.sgd(learning_rate = 0.1)
model_state = train_state.TrainState.create(apply_fn = baseline_model.apply, params = model_params_init, tx = optimizer)


