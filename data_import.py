import numpy as np
import pandas as pd

# ============
# File Parsing
# ============

cna_file_path = "./msk_met_2021/data_cna.txt"
clinical_file_path = "./msk_met_2021/data_clinical_patient.txt"
sample_file_path = "./msk_met_2021/data_clinical_sample.txt"


cna_data_full = pd.read_csv(cna_file_path, sep = '\t', header=None, low_memory = False).T
clinical_data_full = pd.read_csv(clinical_file_path, sep = '\t', low_memory = False, skiprows = 4)
sample_data_full = pd.read_csv(sample_file_path, sep = '\t', low_memory = False, skiprows = 4)
cna_data_full.iloc[0,0] = "PATIENT_ID"

# ================
# Output CSV files
# ================

# cna_data_full.to_csv("./msk_met_2021/data_cna_out.csv", index=False)
# clinical_data_full.to_csv("./msk_met_2021/data_clinical_patient_out.csv", index=False)
# sample_data_full.to_csv("./msk_met_2021/data_clinical_sample_out.csv", index=False)
# print("Done")