"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script evaluates the reactant retrieval and scoring approach for a given compound database.
"""

from config import FullConfig

from reactant_retrieval.reactant_retrieval_and_scoring import benchmark_reactant_candidate_retrieval


config = FullConfig.load()

# benchmark_reactant_candidate_retrieval(**input_params)