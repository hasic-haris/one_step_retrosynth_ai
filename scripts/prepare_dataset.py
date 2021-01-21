"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 2nd, 2020
Description: This script prepares all of the necessary datasets for the construction and the evaluation of the approach.
"""

from config import FullConfig

from data_methods.dataset_construction import generate_unique_compound_pools
from data_methods.dataset_construction import expand_reaction_dataset
from data_methods.dataset_construction import generate_dataset_splits
from data_methods.dataset_construction import generate_fingerprint_datasets
from data_methods.dataset_construction import create_model_training_datasets


full_config = FullConfig.load()

print("\nStep 1/5: Generate unique compound pools for the reactants and products separately.\n")
#generate_unique_compound_pools(full_config)

print("\nStep 2/5: Expand the original dataset with additional, useful information.\n")
# NOTE: This step might produce a lot of RDKit warnings. They can be ignored since I didn't manage to turn them off.
#expand_reaction_dataset(full_config)

print("\nStep 3/5: Split the dataset for n-fold cross validation sets for training, validation and testing.\n")
#generate_dataset_splits(full_config)

print("\nStep 4/5: Generate all specified representations for all of the constructed data splits.\n")
#generate_fingerprint_datasets(full_config)

print("\nStep 5/5: Generate the final dataset that will be used for model training.\n")
#create_model_training_datasets(full_config)
