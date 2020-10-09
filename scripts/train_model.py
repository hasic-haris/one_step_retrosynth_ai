"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script runs the training process of the neural network model on the generated dataset.
"""

from config import FullConfig
from model_methods.model_construction import train_model, test_model


full_config = FullConfig.load()

#print("\nStep 1/2: Train the models with the specified hyper-parameters.\n")
#train_model(full_config, specific_folds=[1, 2])

print("\nStep 2/2: Test the trained models.\n")
test_model(full_config, specific_folds=[1, 2])
