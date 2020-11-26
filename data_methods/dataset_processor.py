"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  November 26th, 2020
Description: This file contains the improved, multiprocessing version of the functions for processing the input dataset.
"""

import os
import random

import multiprocessing as mp
import numpy as np
import pandas as pd

from collections import Counter
from itertools import repeat
from tqdm import tqdm

from chemistry_methods.reactions import parse_reaction_roles


class DatasetProcessor:
    @staticmethod
    def __generate_initial_pools(input_dataset: pd.DataFrame):
        """ Generates the initial Pandas dataframes to store the unique reactant and product compounds. """

        print("Generating unique reactant and product compound representations...", end="")

        with mp.Pool(mp.cpu_count()) as process_pool:
            parsed_smiles_tuples = process_pool.starmap(parse_reaction_roles, zip(input_dataset["rxn_smiles"].values,
                                                                                  repeat("canonical_smiles_no_maps")))
            parsed_mol_tuples = process_pool.starmap(parse_reaction_roles, zip(input_dataset["rxn_smiles"].values,
                                                                               repeat("mol_no_maps")))
            process_pool.close()
            process_pool.join()

        print("done.")

        unique_reactants_pool = pd.DataFrame({"canonical_smiles": [smi_tup[0] for smi_tup in parsed_smiles_tuples],
                                              "mol_object": [mol_tup[0] for mol_tup in parsed_mol_tuples],
                                              "reaction_class": [{cls} for cls in input_dataset["class"].values]})

        unique_products_pool = pd.DataFrame({"canonical_smiles": [smi_tup[2] for smi_tup in parsed_smiles_tuples],
                                             "mol_object": [mol_tup[2] for mol_tup in parsed_mol_tuples],
                                             "reaction_class": [{cls} for cls in input_dataset["class"].values]})

        return unique_reactants_pool, unique_products_pool

    @staticmethod
    def __merge_two_sets(set_a: set, set_b: set):
        return set_a.union(set_b)

    @staticmethod
    def generate_unique_compound_pools(args):
        """ Generates and stores unique (RDKit Canonical SMILES) chemical compound pools of the reactants and products
            for a chemical reaction dataset. The dataset needs to be .csv file and should contain at least a column
            named 'rxn_smiles' in which the values for the mapped reaction SMILES strings are stored. """

        # Read the raw original chemical reaction dataset.
        input_dataset = pd.read_csv(args.dataset_config.input_dataset)

        # Generate the initial unique canonical SMILES reactant and product pools.
        unique_reactants_pool, unique_products_pool = DatasetProcessingMethods.__generate_initial_pools(input_dataset)

        # Aggregate reaction classes for the same reactant compounds.
        aggregation_functions = {"mol_object": "first",
                                 "reaction_class": "first"}  # "reaction_class": DatasetProcessingMethods.__merge_two_sets}
        unique_reactants_pool = unique_reactants_pool.groupby(["canonical_smiles"]).agg(aggregation_functions)

        print(unique_reactants_pool)
