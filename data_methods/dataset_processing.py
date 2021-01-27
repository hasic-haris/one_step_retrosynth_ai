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
    def extract_data_from_rxn_smiles(rxn_smiles: str):

        reactant_pool_smiles, product_pool_smiles, reactant_pool_mol, product_pool_mol = [], [], [], []

        # Extract and save the canonical SMILES from the reaction.
        reactants, _, products = parse_reaction_roles(rxn_smiles, as_what="canonical_smiles_no_maps")
        [reactant_pool_smiles.append(reactant) for reactant in reactants]
        [product_pool_smiles.append(product) for product in products]

        # Extract and save the RDKit Mol objects from the reaction.
        reactants, _, products = parse_reaction_roles(rxn_smiles, as_what="mol_no_maps")
        [reactant_pool_mol.append(reactant) for reactant in reactants]
        [product_pool_mol.append(product) for product in products]

        return reactant_pool_smiles, product_pool_smiles, reactant_pool_mol, product_pool_mol

    @staticmethod
    def generate_something(compound_dataset, rxn_smiles_column, use_multiprocessing=True):

        if use_multiprocessing:

            with mp.Pool(mp.cpu_count()) as process_pool:
                lol, lel = zip(*[(processed_entry[0], processed_entry[1]) for processed_entry in
                                 tqdm(process_pool.imap(DatasetProcessing.extract_data_from_rxn_smiles,
                                                        compound_dataset[rxn_smiles_column].values),
                                      total=len(compound_dataset.index),
                                      ascii=True, desc="Processing entries (Multiple cores)")])
                process_pool.close()
                process_pool.join()

        else:
            lol, lel = zip(*[(processed_entry[0], processed_entry[1]) for processed_entry in
                             [DatasetProcessing.extract_data_from_rxn_smiles(row) for row in
                              tqdm(compound_dataset[rxn_smiles_column].values,
                                   ascii=True, total=len(compound_dataset.index),
                                   desc="Processing entries (Single core)")]])

        print(lol[0:5])
        print(lel[0:5])
        exit(0)

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



        pass

    @staticmethod
    def generate_compound_pools_from_dataset(args):

        # Read the input chemical reaction dataset.
        input_dataset = pd.read_csv(args.data_config.input_dataset_file_path)

        # Iterate through the chemical reaction entries and generate unique canonical SMILES reactant and product pools.
        # Reagents are skipped in this research.
        DatasetProcessing.generate_something(input_dataset, "rxn_smiles",
                                             use_multiprocessing=args.data_config.use_multiprocessing)

        # 1. Read the dataset.
        # 2. Extract data from each reaction SMILES string
        # 3. Aggregate the rows based on the target molecule and reaction classes.
        # 4. Generate the ECFP4 for each entry.
