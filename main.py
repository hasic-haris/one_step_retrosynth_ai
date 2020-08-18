"""
Author:      Hasic Haris (Phd Student @ Ishida Lab, Department of Computer Science, Tokyo Institute of Technology)
Created on:  February 1st, 2020
Description: This is the main file. All test functions can be ran from here.
"""
from data_methods.dataset_construction import generate_unique_compound_pools
from data_methods.dataset_construction import expand_reaction_dataset
from data_methods.dataset_construction import generate_dataset_splits
from data_methods.dataset_construction import generate_fingerprint_datasets, create_final_fingerprint_datasets
from data_methods.dataset_construction import generate_pipeline_test_dataset

from neural_networks.tf_main import train_and_evaluate_model, qualitative_model_assessment

from reactant_retrieval.reactant_retrieval_and_scoring import benchmark_reactant_candidate_retrieval

from final_combined_pipeline.one_step_retrosynthetic_analysis import analyze_novel_molecule

# ----------------------------------------------------------- #
#   Stage 1: Prepare the input data for the planned models.   #
# ----------------------------------------------------------- #

# Preparation Step: Define all necessary input parameters to generate the input data for the network.
input_params = {
    # General dataset information.
    "source_dataset_path": "data_source/data_processed.csv",
    "output_folder": "D:/Datasets/one_step_retrosynth_ai_output/",

    # Dataset pre-processing settings.
    "num_folds": 5,
    "validation_split": 0.1,
    "random_seed": 101,
    "reaction_classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    # Fingerprint-related settings.
    "similarity_search_fp_config": {"type": "ecfp", "radius": 2, "bits": 1024},
    "input_data_fp_config": [{"type": "ecfp", "radius": 2, "bits": 1024, "folder_name": "ecfp_2_1024"},
                             {"type": "hsfp", "radius": 2, "bits": 1024, "ext": 2, "folder_name": "hsfp_2_2_1024"}],
    "best_fp_config": {"type": "hsfp", "radius": 2, "bits": 1024, "ext": 2, "folder_name": "hsfp_2_2_1024"}
}

# Step 1: Generate unique compound (RDKit Canonical SMILES) pools for the reactants and products separately.
# generate_unique_compound_pools(**input_params)

# Step 2: Generate the final dataset that will be used in the approach.
# expand_reaction_dataset(**input_params)

# Step 3: Split the dataset for n-fold cross validation sets for training, validation and testing.
# generate_dataset_splits(**input_params)

# Step 4: Generate specified representations for the constructed data splits.
# generate_fingerprint_datasets(**input_params)
# create_final_fingerprint_datasets(**input_params)

# --------------------------------------------------- #
#   Stage 2: Train and evaluate the planned models.   #
# --------------------------------------------------- #

# Step 1: Define all necessary model parameters to train the networks.
# train_and_evaluate_model(**input_params)

# ----------------------------------------------------------------- #
#   Stage 3: Evaluate the reactant retrieval and scoring process.   #
# ----------------------------------------------------------------- #

# benchmark_reactant_candidate_retrieval(**input_params)

# -------------------------------------------------- #
#   Stage 4: Evaluate the final combined pipeline.   #
# -------------------------------------------------- #

# generate_pipeline_test_dataset(fold_ind=1, **input_params)

# qualitative_model_assessment(**input_params)


