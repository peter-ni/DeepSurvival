import jax as jp
import numpy as np
import pandas as pd
import common_utils


from jax import grad, jit, vmap
from jax import random



cancer_data_path = '/Users/peterni/Desktop/Projects/DeepSurvival/DeepSurvival/cleaned_datasets/cancer_data.csv'
cancer_df = pd.read_csv(cancer_data_path)


clinical_vars = [
    "SEX",
    "RACE",
    "AGE_AT_SEQUENCING",
    "SAMPLE_TYPE",
    "SAMPLE_COVERAGE",
    "TUMOR_PURITY",
    "MSI_SCORE",
    "TMB_NONSYNONYMOUS",
    "STATUS",
    "TIME",
    "TIME_SINCE_DX"
]

all_col_names = cancer_df.columns
exclude = set(clinical_vars + ["PATIENT_ID"])
cna_vars = [col for col in cancer_df.columns if col not in exclude]



# Parameter Initialization















