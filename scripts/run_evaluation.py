"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script evaluates the full single-step retrosynthesis pipeline.
"""

from config import FullConfig
from one_step_retrosynthesis.reactant_retrieval import benchmark_reactant_candidate_retrieval


full_config = FullConfig.load()

# TODO:
# 1. Generate a pool of the Top-10 most frequent occurring molecules per each class.
# 2.

print("\nOptional Step: Run a benchmark for the reactant retrieval and scoring process.\n")
benchmark_reactant_candidate_retrieval(full_config)

#print("\nStep 1/3: Generate the dataset that will be used for the final evaluation of the approach.\n")
#create_final_evaluation_dataset(full_config)

#print("\nStep 2/3: Use the constructed model to classify the .\n")

#print("\nStep 3/3: Generate the final dataset that will be used in the final evaluation of the approach.\n")

# generate_pipeline_test_dataset(fold_ind=1, **input_params)
# qualitative_model_assessment(**input_params)
