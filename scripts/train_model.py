"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script runs the training process of the neural network model on the generated dataset.
"""

from config import FullConfig
from model_methods.model_construction import train_model


full_config = FullConfig.load()

import pandas as pd

tr_x = pd.read_pickle(full_config.dataset_config.output_folder + "x_training.pkl")
print("Training data: " + str(tr_x.shape))
tr_y = pd.read_pickle(full_config.dataset_config.output_folder + "y_mc_training.pkl")
print("Training labels: " + str(tr_y.shape))

vd_x = pd.read_pickle(full_config.dataset_config.output_folder + "x_validation.pkl")
print("Validation data: " + str(vd_x.shape))
vd_y = pd.read_pickle(full_config.dataset_config.output_folder + "y_mc_validation.pkl")
print("Validation labels: " + str(vd_y.shape))

ts_x = pd.read_pickle(full_config.dataset_config.output_folder + "x_test.pkl")
print("Test data: " + str(ts_x.shape))
ts_y = pd.read_pickle(full_config.dataset_config.output_folder + "y_mc_test.pkl")
print("Test data: " + str(ts_y.shape))

#train_model(full_config, specific_fold=1)
