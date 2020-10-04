"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script runs the training process of the neural network model on the generated dataset.
"""

from config import FullConfig

from neural_networks.tf_main import train_and_evaluate_model, qualitative_model_assessment


config = FullConfig.load()

#train_and_evaluate_model(**input_params)
