import os
import random

import multiprocessing as mp
import numpy as np
import pandas as pd

from collections import Counter
from itertools import repeat
from tqdm import tqdm
from typing import List, Tuple, Optional

from chemistry_methods.reactions import parse_reaction_roles


class DatasetProcessing:

    @staticmethod
    def __generate_initial_pools(input_dataset: pd.DataFrame) -> Tuple:

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

        input_dataset = pd.read_csv(args.dataset_config.input_dataset)

        unique_reactants_pool, unique_products_pool = DatasetProcessing.__generate_initial_pools(input_dataset)

        # Aggregate reaction classes for the same reactant compounds.
        #aggregation_functions = {"mol_object": "first",
        #                         "reaction_class": "first"}  # "reaction_class": DatasetProcessingMethods.__merge_two_sets}
        #unique_reactants_pool = unique_reactants_pool.groupby(["canonical_smiles"]).agg(aggregation_functions)

        print(unique_reactants_pool)
        print(unique_products_pool)
