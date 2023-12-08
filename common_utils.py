import numpy as np
import pandas as pd
import torch


cancer_pd = pd.read_csv('out_df.csv')



target_vars = [
    "TIME",
    "STATUS"
]

clin_vars = [
    "SEX",
    "RACE",
    "AGE_AT_SEQUENCING",
    "SAMPLE_TYPE",
    "SAMPLE_COVERAGE",
    "TUMOR_PURITY",
    "MSI_SCORE",
    "TMB_NONSYNONYMOUS",
    "TIME_SINCE_DX"
]




cna_vars = list(set(cancer_pd.columns) - set(target_vars) - set(clin_vars) - set(['PATIENT_ID']))


# Some functions for loss/likelihood calculations
# Same parametrization as Wikpiedia
def weibull_pdf(t, lam, k):
    out = torch.zeros(len(t))
    for i in range(len(t)):
        entry = (k/lam[i]) * (t[i]/lam[i])**(k-1) * torch.exp(-(t[i]/lam[i])**k)
        out[i] = torch.sum(entry)
    
    
    return out
    # return (k/lam) * (t/lam)**(k-1) * jnp.exp(-(t/lam)**k)

def one_minus_weibull_cdf(t, lam, k):
    out = torch.zeros(len(t))
    for i in range(len(t)):
        entry = torch.exp(-(t[i]/lam[i])**k)
        out[i] = torch.sum(entry)
    return out
    # return jnp.exp(-(t/lam)**k)




# Helper Functions for Parameter Initialization

def get_random_layer_params(m, n, rng_key, scale=0.01):
    """Helper function to randomly initialize 
    weights and biases using the JAX-defined randoms."""
    w_key, b_key = jran.split(rng_key)
    ran_weights = scale * jran.normal(w_key, (n, m))
    ran_biases = scale * jran.normal(b_key, (n,)) 
    return ran_weights, ran_biases


def get_init_network_params(sizes, rng_key):
    """Initialize all layers for a fully-connected neural network."""
    keys = jran.split(rng_key, len(sizes))
    return [get_random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def get_network_layer_sizes(n_features, n_targets, n_layers, n_neurons_per_layer):
    dense_layer_sizes = [n_neurons_per_layer]*n_layers
    layer_sizes = [n_features, *dense_layer_sizes, n_targets]
    return layer_sizes


# Gets survival times from eta, base hazard parameters
def getSurvTime(eta, base_haz_params):
    base_haz_params_trans = torch.nn.functional.softplus(base_haz_params)
    rho,  k = base_haz_params_trans[0], base_haz_params_trans[1]
    lam = rho * torch.exp(- eta / k)
    two = torch.Tensor([2])
    median_surv = lam * (torch.log(two))**(1/k)
    return median_surv

# Gets time-dependent AUC
from sklearn.metrics import roc_auc_score


def timeAUC(input, times):
    results = []
    defined = True
    input = input.sort_values(by = input.columns[2])
    for time in times:
        df_subset = input[input.iloc[:, 0] <= time]
        true_labels = df_subset.iloc[:,1]
        pred_scores = df_subset.iloc[:,2]
    
        if len(set(true_labels)) > 1:
            auc = roc_auc_score(true_labels, pred_scores)
        else:
            auc = None
            defined = False
        results.append(auc)
    return results









