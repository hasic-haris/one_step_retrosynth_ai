"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script evaluates the full single-step retrosynthesis pipeline.
"""

from config import FullConfig

from retrosynthesis_methods.one_step_retrosynthetic_analysis import analyze_novel_molecule


config = FullConfig.load()

# generate_pipeline_test_dataset(fold_ind=1, **input_params)

# qualitative_model_assessment(**input_params)