"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script evaluates the full single-step retrosynthesis pipeline.
"""

from config import FullConfig

from reactant_retrieval_and_scoring.reactant_retrieval_and_scoring import benchmark_reactant_candidate_retrieval
from data_methods.dataset_construction import create_final_evaluation_dataset
from reactant_retrieval_and_scoring.reactant_retrieval_and_scoring import complete_and_score_suggestions
from single_step_retrosynthesis.single_step_retrosynthesis import bulk_analyze_disconnection_suggestions


full_config = FullConfig.load()

print("\nOptional Step: Run a benchmark for the reactant retrieval and scoring process.\n")
benchmark_reactant_candidate_retrieval(full_config)

print("\nStep 1/3: Generate the dataset that will be used for the final evaluation of the approach.\n")
create_final_evaluation_dataset(full_config)

print("\nStep 2/3: Perform the reactant retrieval and scoring on the disconnection suggestions.\n")
complete_and_score_suggestions(full_config)

print("\nStep 3/3: Check the reactant retrieval and scoring performance on these molecules.\n")
bulk_analyze_disconnection_suggestions(full_config)
