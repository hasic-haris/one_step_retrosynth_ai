"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script evaluates the full single-step retrosynthesis pipeline.
"""

from config import FullConfig
from one_step_retrosynthesis.reactant_retrieval import benchmark_reactant_candidate_retrieval
from data_methods.dataset_construction import create_final_evaluation_dataset
from model_methods.model_construction import apply_model


full_config = FullConfig.load()

# TODO:
# 1. Generate a pool of the Top-10 most frequent occurring molecules per each class.
# 2. Generate a good dataset for the final evaluation. DONE
# 3. Apply the trained models on that dataset.
# 4. Generate a good visual summary representation of the actual results.

# print("\nOptional Step: Run a benchmark for the reactant retrieval and scoring process.\n")
# benchmark_reactant_candidate_retrieval(full_config)

print("\nStep 1/3: Generate the dataset that will be used for the final evaluation of the approach.\n")
#create_final_evaluation_dataset(full_config)

print("\nStep 2/3: Apply the constructed model to classify the non-filtered data from the test molecules.\n")
apply_model(full_config)


#print("\nStep 3/3: Check the reactant retrieval and scoring performance on these molecules.\n")

# generate_pipeline_test_dataset(fold_ind=1, **input_params)
# qualitative_model_assessment(**input_params)
