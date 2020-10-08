"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script runs the training process of the neural network model on the generated dataset.
"""

import numpy as np
from config import FullConfig
from model_methods.model_construction import train_model, test_model


full_config = FullConfig.load()

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#train_model(full_config, specific_fold=1)
test_model(full_config, specific_fold=1)
