# DeepSurvival

Steps to Run:

1) Run data_import.py, which pulls data files out of msk_met2021 raw trial data
2) Run data_cleaner.R, which does some encoding and additional cleaning and outputs cleaned data out_df.csv
3) Run pytorch_model.py to run and save a DNN model. Note, the model files use functions from common_utils.py
4) Saved model parameters, time-dependent AUC, and train/test loss by epoch are in saved_models, predictions will be saved in the home directory.
5) Some plots can be generated in analysis.R

