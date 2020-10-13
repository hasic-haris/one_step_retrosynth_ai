"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script evaluates the full single-step retrosynthesis pipeline.
"""

from config import FullConfig
from one_step_retrosynthesis.reactant_retrieval import benchmark_reactant_candidate_retrieval
from data_methods.dataset_construction import create_final_evaluation_dataset
from model_methods.model_construction import apply_model
from one_step_retrosynthesis.reactant_retrieval import complete_and_score_suggestions
from one_step_retrosynthesis.retrosynthetic_analysis import bulk_analyze_disconnection_suggestions


full_config = FullConfig.load()

# TODO:
# 1. Generate a pool of the Top-10 most frequent occurring molecules per each class.
# 2. Generate a good dataset for the final evaluation. DONE
# 3. Apply the trained models on that dataset. DONE
# 4. Create a bulk reactant retrieval and scoring for the generated model results.
# 5. Generate a good visual summary representation of the actual results.

# print("\nOptional Step: Run a benchmark for the reactant retrieval and scoring process.\n")
# benchmark_reactant_candidate_retrieval(full_config)

#print("\nStep 1/4: Generate the dataset that will be used for the final evaluation of the approach.\n")
#create_final_evaluation_dataset(full_config)

#print("\nStep 2/4: Apply the constructed model to classify the non-filtered data from the test molecules.\n")
#apply_model(full_config)

print("\nStep 3/4: Perform the reactant retrieval and scoring on the disconnection suggestions.\n")
complete_and_score_suggestions(full_config)

#print("\nStep 4/4: Check the reactant retrieval and scoring performance on these molecules.\n")
#bulk_analyze_disconnection_suggestions(full_config)
