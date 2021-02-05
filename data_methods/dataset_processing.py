import os
import random

import multiprocessing as mp
import numpy as np
import pandas as pd

from collections import Counter
from itertools import repeat
from tqdm import tqdm
from typing import List, Tuple, Optional

from chemistry_methods.fingerprints import MolecularFingerprintUtils
from chemistry_methods.reaction_analysis import extract_info_from_reaction
from chemistry_methods.reactions import parse_reaction_roles

from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect


class DataProcessingWorkers:
    """ Description: TBD. """

    @staticmethod
    def set_num_cores(config_num_cores: int) -> int:
        """ Description: Internal helper to determine the number of available CPU cores. """

        if config_num_cores <= 1:
            print("The entered number of CPU cores cannot be negative. Switching to single core execution mode.")
            return 1
        else:
            if config_num_cores > mp.cpu_count():
                print("The entered number of available cores larger than the number of available cores. ({} > {})"
                      "Using all available {} cores.".format(config_num_cores, mp.cpu_count(), mp.cpu_count()))
                return mp.cpu_count()
            else:
                return config_num_cores

    @staticmethod
    def extract_pool_data(reaction_smiles_and_class: Tuple) -> Tuple:
        """ Description: MP-friendly worker function for the generation of initial compound pool data. """

        # Extract the Canonical SMILES and Mol objects of the reactants and products from a reaction SMILES string.
        reactants, _, products = parse_reaction_roles(reaction_smiles_and_class[0], as_what="canonical_tuple_no_maps")

        # Return the (Canonical SMILES, Mol) tuple and the reaction class for a single entry.
        return reactants, products, reaction_smiles_and_class[1]

    @staticmethod
    def generate_molecular_fingerprint(mol_and_config: Tuple) -> Tuple:
        """ Description: MP-friendly worker function for the generation of molecular fingerprints. """

        # Depending on the input, generate and return an ECFP or HSFP for a reaction SMILES string.
        if mol_and_config[1]["type"] == "ecfp":
            molecular_fingerprint = MolecularFingerprintUtils.construct_ecfp(mol_and_config[0],
                                                                             radius=mol_and_config[1]["radius"],
                                                                             bits=mol_and_config[1]["bits"])
        else:
            molecular_fingerprint = MolecularFingerprintUtils.construct_hsfp(mol_and_config[0],
                                                                             radius=mol_and_config[1]["radius"],
                                                                             bits=mol_and_config[1]["bits"],
                                                                             from_atoms=[])

        return molecular_fingerprint

    @staticmethod
    def extract_relevant_information(all_data_necessary: Tuple):
        """ Extracts the necessary information from a single mapped reaction SMILES string. """

        # Extract the canonical SMILES and RDKit Mol objects from the reaction SMILES string.
        reactant_smiles, _, product_smiles = parse_reaction_roles(all_data_necessary[0], as_what="canonical_smiles_no_maps")
        reactants, _, products = parse_reaction_roles(all_data_necessary[0], as_what="mol_no_maps")

        # Sort the reactants and products in descending order by number of atoms so the largest reactants is always first.
        reactants, reactant_smiles = zip(*sorted(zip(reactants, reactant_smiles), key=lambda k: len(k[0].GetAtoms()),
                                                 reverse=True))
        products, product_smiles = zip(*sorted(zip(products, product_smiles), key=lambda k: len(k[0].GetAtoms()),
                                               reverse=True))

        r_uq_mol_maps, rr_smiles, rr_smols, rr_smals, rr_fps, rnr_smiles, rnr_smols, rnr_smals, rnr_fps = \
            [], [], [], [], [], [], [], [], []
        p_uq_mol_maps, pr_smiles, pr_smols, pr_smals, pr_fps, pnr_smiles, pnr_smols, pnr_smals, pnr_fps = \
            [], [], [], [], [], [], [], [], []

        # Extract the reactive and non-reactive parts of the reactant and product molecules.
        reactant_frags, product_frags = extract_info_from_reaction(all_data_necessary[0])

        # Iterate through all of the reactants and aggregate the specified data.
        for r_ind, reactant in enumerate(reactants):
            r_uq_mol_maps.append(all_data_necessary[1].index(reactant_smiles[r_ind]))
            rr_smiles.append(reactant_frags[r_ind][0][0])
            rnr_smiles.append(reactant_frags[r_ind][1][0])
            rr_smols.append(reactant_frags[r_ind][0][2])
            rnr_smols.append(reactant_frags[r_ind][1][2])
            rr_smals.append(reactant_frags[r_ind][0][3])
            rnr_smals.append(reactant_frags[r_ind][1][3])
            rr_fps.append(
                MolecularFingerprintUtils.construct_ecfp(reactant_frags[r_ind][0][2], radius=all_data_necessary[3]["radius"], bits=all_data_necessary[3]["bits"]))
            rnr_fps.append(
                MolecularFingerprintUtils.construct_ecfp(reactant_frags[r_ind][1][2], radius=all_data_necessary[3]["radius"], bits=all_data_necessary[3]["bits"]))

        # Iterate through all of the products and aggregate the specified data.
        for p_ind, product in enumerate(products):
            p_uq_mol_maps.append(all_data_necessary[2].index(product_smiles[p_ind]))
            pr_smiles.extend(product_frags[p_ind][0][0])
            pnr_smiles.extend(product_frags[p_ind][1][0])
            pr_smols.extend(product_frags[p_ind][0][2])
            pnr_smols.extend(product_frags[p_ind][1][2])
            pr_smals.extend(product_frags[p_ind][0][3])
            pnr_smals.extend(product_frags[p_ind][1][3])

            for pf in product_frags[p_ind][0][2]:
                pr_fps.append(MolecularFingerprintUtils.construct_ecfp(pf, radius=all_data_necessary[3]["radius"], bits=all_data_necessary[3]["bits"]))
            for pf in product_frags[p_ind][1][2]:
                pnr_fps.append(MolecularFingerprintUtils.construct_ecfp(pf, radius=all_data_necessary[3]["radius"], bits=all_data_necessary[3]["bits"]))

        # Return the extracted information.
        return r_uq_mol_maps, rr_smiles, rr_smols, rr_smals, rr_fps, rnr_smiles, rnr_smols, rnr_smals, rnr_fps, \
               p_uq_mol_maps, pr_smiles, pr_smols, pr_smals, pr_fps, pnr_smiles, pnr_smols, pnr_smals, pnr_fps


class DatasetProcessing:
    """ Description: Group of methods for the pre-processing of the input reaction dataset. """

    def __init__(self, config_params):
        """ Description: TBD. """

        self.input_dataset_file_path = config_params.data_config.input_dataset_file_path
        self.reaction_smiles_column = config_params.data_config.reaction_smiles_column
        self.reaction_class_column = config_params.data_config.reaction_class_column
        self.output_folder_path = config_params.data_config.output_folder_path

        self.fp_config = config_params.descriptor_config.similarity_search

        self.use_num_cores = DataProcessingWorkers.set_num_cores(config_params.data_config.use_num_cores)

    def __generate_unique_compound_pools(self, compound_dataset):
        """ Description: TBD. """

        # Step 1: Extracting the canonical SMILES and reaction class for each reactant and product in the input dataset.
        if self.use_num_cores > 1:
            with mp.Pool(self.use_num_cores) as process_pool:
                reactants, products, reaction_class = \
                    zip(*[(processed_entry[0], processed_entry[1], processed_entry[2])
                          for processed_entry in
                          tqdm(process_pool.imap(DataProcessingWorkers.extract_pool_data, compound_dataset.values),
                               total=len(compound_dataset.index), ascii=True,
                               desc="Generating initial reactant and product data (Cores: {})".format(self.use_num_cores))])

                process_pool.close()
                process_pool.join()

        else:
            reactants, products, reaction_class = \
                zip(*[(processed_entry[0], processed_entry[1], processed_entry[2])
                      for processed_entry in [DataProcessingWorkers.extract_pool_data(reaction_smiles_and_class)
                                              for reaction_smiles_and_class in
                                              tqdm(compound_dataset.values,
                                                   ascii=True, total=len(compound_dataset.index),
                                                   desc="Generating initial reactant and product data (Single Core)")]])

        reactant_reaction_class = [reaction_class[list_ind] for list_ind, reactant_list in enumerate(reactants)
                                   for _ in reactant_list]
        product_reaction_class = [reaction_class[list_ind] for list_ind, product_list in enumerate(products)
                                   for _ in product_list]

        reactant_smiles = [reactant[0] for reactant_list in reactants for reactant in reactant_list]
        reactant_mols = [reactant[1] for reactant_list in reactants for reactant in reactant_list]

        product_smiles = [product[0] for product_list in products for product in product_list]
        product_mols = [product[1] for product_list in products for product in product_list]

        # Step 2: Aggregating the reaction classes for the same reactant and product canonical SMILES in the pools.
        unique_reactants_pool = pd.DataFrame({"canonical_smiles": reactant_smiles, "mol_object": reactant_mols,
                                              "reaction_class": reactant_reaction_class}).groupby("canonical_smiles")\
            .agg({"mol_object": lambda x: x.tolist()[0], "reaction_class": lambda x: set(x.tolist())})
        unique_reactants_pool.reset_index(inplace=True)

        unique_products_pool = pd.DataFrame({"canonical_smiles": product_smiles, "mol_object": product_mols,
                                              "reaction_class": product_reaction_class}).groupby("canonical_smiles") \
            .agg({"mol_object": lambda x: x.tolist()[0], "reaction_class": lambda x: set(x.tolist())})
        unique_products_pool.reset_index(inplace=True)

        # Step 3: When the duplicates have been filtered out for both pools, the mols and fingerprints are generated.
        if self.use_num_cores > 1:
            with mp.Pool(self.use_num_cores) as process_pool:
                reactant_ecfps = [processed_entry for processed_entry in
                                  tqdm(process_pool.imap(DataProcessingWorkers.generate_molecular_fingerprint,
                                                         [(x, self.fp_config) for x in unique_reactants_pool["mol_object"].values]),
                                       total=len(unique_reactants_pool.index), ascii=True,
                                       desc="Generating the reactant fingerprint data (Cores: {})".format(self.use_num_cores))]

                product_ecfps = [processed_entry for processed_entry in
                                  tqdm(process_pool.imap(DataProcessingWorkers.generate_molecular_fingerprint,
                                                         [(x, self.fp_config) for x in unique_products_pool["mol_object"].values]),
                                       total=len(unique_products_pool.index), ascii=True,
                                       desc="Generating the product fingerprint data (Cores: {})".format(self.use_num_cores))]

                process_pool.close()
                process_pool.join()

        else:
            reactant_ecfps = [processed_entry for processed_entry in
                              [DataProcessingWorkers.generate_molecular_fingerprint((x, self.fp_config))
                               for x in tqdm(unique_reactants_pool["mol_object"].values,
                                             ascii=True, total=len(unique_reactants_pool.index),
                                             desc="Generating the reactant fingerprint data (Single Core)")]]

            product_ecfps = [processed_entry for processed_entry in
                              [DataProcessingWorkers.generate_molecular_fingerprint((x, self.fp_config))
                               for x in tqdm(unique_reactants_pool["mol_object"].values,
                                             ascii=True, total=len(unique_reactants_pool.index),
                                             desc="Generating the product fingerprint data (Single Core)")]]

        print("Saving the processed compound data...", end="", flush=True)
        unique_reactants_pool.assign(ecfp_1024=reactant_ecfps).to_pickle(self.output_folder_path + "unique_reactants_pool.pkl")
        unique_products_pool.assign(ecfp_1024=product_ecfps).to_pickle(self.output_folder_path + "unique_products_pool.pkl")
        print("done.")

    def __expand_reaction_dataset(self, compound_dataset):

        # Read the raw chemical reaction dataset and rename the fetched columns.
        compound_dataset.columns = ["reaction_smiles", "reaction_class"]

        # Create new columns to store the id's of the unique reactant and product molecules.
        compound_dataset["reactants_uq_mol_maps"], compound_dataset["products_uq_mol_maps"] = None, None
        # Create new columns to store the SMILES strings of the reactive parts of reactant and product molecules.
        compound_dataset["reactants_reactive_smiles"], compound_dataset["products_reactive_smiles"] = None, None
        # Create new columns to store the SMILES Mol objects of the reactive parts of reactant and product molecules.
        compound_dataset["reactants_reactive_smols"], compound_dataset["products_reactive_smols"] = None, None
        # Create new columns to store the SMARTS Mol objects of the reactive parts of reactant and product molecules.
        compound_dataset["reactants_reactive_smals"], compound_dataset["products_reactive_smals"] = None, None
        # Create new columns to store the fingerprints of the reactive parts of reactant and product molecules.
        compound_dataset["reactants_reactive_fps"], compound_dataset["products_reactive_fps"] = None, None

        # Create new columns to store the SMILES strings of the non-reactive parts of reactant and product molecules.
        compound_dataset["reactants_non_reactive_smiles"], compound_dataset["products_non_reactive_smiles"] = None, None
        # Create new columns to store the SMILES Mol objects of the non-reactive parts of reactant and product molecules.
        compound_dataset["reactants_non_reactive_smols"], compound_dataset["products_non_reactive_smols"] = None, None
        # Create new columns to store the SMARTS Mol objects of the non-reactive parts of reactant and product molecules.
        compound_dataset["reactants_non_reactive_smals"], compound_dataset["products_non_reactive_smals"] = None, None
        # Create new columns to store the fingerprints of the non-reactive parts of reactant and product molecules.
        compound_dataset["reactants_non_reactive_fps"], compound_dataset["products_non_reactive_fps"] = None, None

        # Read the previously generated unique molecule pools.
        reactant_pool = pd.read_pickle(self.output_folder_path +
                                       "unique_reactants_pool.pkl")["canonical_smiles"].values.tolist()
        product_pool = pd.read_pickle(self.output_folder_path +
                                      "unique_products_pool.pkl")["canonical_smiles"].values.tolist()

        if self.use_num_cores > 1:
            with mp.Pool(self.use_num_cores) as process_pool:
                ruqmm, rrsm, rrso, rrsa, rrsf, rnsm, rnso, rnsa, rnsf, puqmm, prsm, prso, prsa, prsf, pnsm, pnso, pnsa, pnsf = \
                    zip(*[(processed_entry[0], processed_entry[1], processed_entry[2], processed_entry[3], processed_entry[4],
                           processed_entry[5], processed_entry[6], processed_entry[7], processed_entry[8], processed_entry[9],
                           processed_entry[10], processed_entry[11], processed_entry[12], processed_entry[13], processed_entry[14],
                           processed_entry[15], processed_entry[16], processed_entry[17])
                          for processed_entry in
                          tqdm(process_pool.imap(DataProcessingWorkers.extract_relevant_information,
                                                 [(x, reactant_pool, product_pool, self.fp_config) for x in
                                                  compound_dataset["reaction_smiles"].values]),
                               total=len(compound_dataset.index), ascii=True,
                               desc="Generating initial reactant and product data (Cores: {})".format(self.use_num_cores))])

                process_pool.close()
                process_pool.join()

        else:
            ruqmm, rrsm, rrso, rrsa, rrsf, rnsm, rnso, rnsa, rnsf, puqmm, prsm, prso, prsa, prsf, pnsm, pnso, pnsa, pnsf = \
                zip(*[(processed_entry[0], processed_entry[1], processed_entry[2], processed_entry[3], processed_entry[4],
                       processed_entry[5], processed_entry[6], processed_entry[7], processed_entry[8], processed_entry[9],
                       processed_entry[10], processed_entry[11], processed_entry[12], processed_entry[13], processed_entry[14],
                       processed_entry[15], processed_entry[16], processed_entry[17])
                      for processed_entry in [DataProcessingWorkers.extract_relevant_information(
                        (lel, reactant_pool, product_pool, self.fp_config)
                    ) for lel in
                                              tqdm(compound_dataset["reaction_smiles"].values,
                                                   ascii=True, total=len(compound_dataset.index),
                                                   desc="Generating initial reactant and product data (Single Core)")]])

            print(ruqmm[0:3])
            print(rrsm[0:3])
            print(rrso[0:3])
            print(rrsa[0:3])
            print(rrsf[0:3])
            print(rnsm[0:3])
            print(rnso[0:3])
            print(rnsa[0:3])
            print(rnsf[0:3])
            print(puqmm[0:3])
            print(prsm[0:3])
            print(prso[0:3])
            print(prsa[0:3])
            print(prsf[0:3])
            print(pnsm[0:3])
            print(pnso[0:3])
            print(pnsa[0:3])
            print(pnsf[0:3])

    def main(self):

        # Read the input chemical reaction dataset.
        input_dataset = pd.read_csv(self.input_dataset_file_path)[[self.reaction_smiles_column,
                                                                   self.reaction_class_column]]

        # STEP 1 -- DONE.
        # self.__generate_unique_compound_pools(input_dataset[[self.reaction_smiles_column,
        #                                                       self.reaction_class_column]])

        # STEP 2.
        self.__expand_reaction_dataset(input_dataset[[self.reaction_smiles_column, self.reaction_class_column]])
