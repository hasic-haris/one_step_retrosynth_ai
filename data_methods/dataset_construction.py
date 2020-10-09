"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 21st, 2020
Description: This file contains necessary functions for the generation and splitting of the raw original dataset.
"""

import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from chemistry_methods.reactions import parse_reaction_roles
from chemistry_methods.fingerprints import construct_ecfp, construct_hsfp
from chemistry_methods.reaction_analysis import extract_info_from_reaction
from chemistry_methods.reaction_cores import get_reaction_core_atoms, get_separated_cores
from chemistry_methods.molecules import get_atom_environment, get_bond_environment
from data_methods.helpers import get_n_most_frequent_rows, encode_one_hot


def generate_unique_compound_pools(args):
    """ Generates and stores unique (RDKit Canonical SMILES) chemical compound pools of the reactants and products for a
        chemical reaction dataset. The dataset needs to contain a column named 'rxn_smiles' in which the values for the
        mapped reaction SMILES strings are stored. """

    reactant_pool_smiles, product_pool_smiles, reactant_pool_mol, product_pool_mol = [], [], [], []
    reactant_reaction_class, product_reaction_class = [], []

    # Read the raw original chemical reaction dataset.
    raw_dataset = pd.read_csv(args.dataset_config.raw_dataset)

    # Iterate through the chemical reaction entries and generate unique canonical SMILES reactant and product pools.
    # Reagents are skipped in this research.
    for row_ind, row in tqdm(raw_dataset.iterrows(), total=len(raw_dataset.index),
                             desc="Generating unique reactant and product compound representations"):
        # Extract and save the canonical SMILES from the reaction.
        reactants, _, products = parse_reaction_roles(row["rxn_smiles"], as_what="canonical_smiles_no_maps")
        [reactant_pool_smiles.append(reactant) for reactant in reactants]
        [product_pool_smiles.append(product) for product in products]

        # Extract and save the RDKit Mol objects from the reaction.
        reactants, _, products = parse_reaction_roles(row["rxn_smiles"], as_what="mol_no_maps")
        [reactant_pool_mol.append(reactant) for reactant in reactants]
        [product_pool_mol.append(product) for product in products]

        # Save the reaction class of the entry.
        [reactant_reaction_class.append(row["class"]) for _ in reactants]
        [product_reaction_class.append(row["class"]) for _ in products]

    # Aggregate the saved reaction classes for the same reactant compounds.
    for reactant_ind, reactant in tqdm(enumerate(reactant_pool_smiles), total=len(reactant_pool_smiles),
                                       desc="Aggregating reaction class values for the reactant compounds"):
        if type(reactant_reaction_class[reactant_ind]) == set:
            continue

        same_reactant_rows = [r_ind for r_ind, r in enumerate(reactant_pool_smiles) if r == reactant]
        aggregated_class_values = [c for c_ind, c in enumerate(reactant_reaction_class) if c_ind in same_reactant_rows]

        for same_row_ind in same_reactant_rows:
            reactant_reaction_class[same_row_ind] = set(aggregated_class_values)

    # Aggregate the saved reaction classes for the same product compounds.
    for product_ind, product in tqdm(enumerate(product_pool_smiles), total=len(product_pool_smiles),
                                     desc="Aggregating reaction class values for the product compounds"):
        if type(product_reaction_class[product_ind]) == set:
            continue

        same_product_rows = [p_ind for p_ind, p in enumerate(product_pool_smiles) if p == product]
        aggregated_class_values = [c for c_ind, c in enumerate(product_reaction_class) if c_ind in same_product_rows]

        for same_row_ind in same_product_rows:
            product_reaction_class[same_row_ind] = set(aggregated_class_values)

    print("Filtering unique reactant and product compounds...", end="")

    # Filter out duplicate reactant molecules from the reactant and product sets.
    reactant_pool_smiles, reactants_uq_ind = np.unique(reactant_pool_smiles, return_index=True)
    product_pool_smiles, products_uq_ind = np.unique(product_pool_smiles, return_index=True)

    # Apply the unique indices to the list of RDKit Mol objects.
    reactant_pool_mol = np.array(reactant_pool_mol)[reactants_uq_ind].tolist()
    product_pool_mol = np.array(product_pool_mol)[products_uq_ind].tolist()

    # Apply the unique indices to the list of reaction classes.
    reactant_reaction_class = np.array(reactant_reaction_class)[reactants_uq_ind].tolist()
    product_reaction_class = np.array(product_reaction_class)[products_uq_ind].tolist()

    print("done.")

    # Pre-generate the reactant molecular fingerprint descriptors for similarity searching purpouses.
    ecfp_1024 = []

    for uqr_ind, uq_reactant in tqdm(enumerate(reactant_pool_smiles), total=len(reactant_pool_smiles),
                                     desc="Generating reactant compound fingerprints"):
        ecfp_1024.append(construct_ecfp(uq_reactant, radius=args.descriptor_config.similarity_search["radius"],
                                        bits=args.descriptor_config.similarity_search["bits"]))

    print("Saving the processed reactant compound data...", end="")

    # Store all of the generated reactant fingerprints in a .pkl file.
    pd.DataFrame({"mol_id": list(range(0, len(reactant_pool_smiles))), "canonical_smiles": reactant_pool_smiles,
                  "mol_object": reactant_pool_mol, "ecfp_1024": ecfp_1024, "reaction_class": reactant_reaction_class}).\
        to_pickle(args.dataset_config.output_folder + "unique_reactants_pool.pkl")

    print("done.")

    # Pre-generate the product molecular fingerprint descriptors for similarity searching purpouses.
    ecfp_1024 = []

    for uqp_ind, uq_product in tqdm(enumerate(product_pool_smiles), total=len(product_pool_smiles),
                                    desc="Generating product compound fingerprints"):
        ecfp_1024.append(construct_ecfp(uq_product, radius=args.descriptor_config.similarity_search["radius"],
                                        bits=args.descriptor_config.similarity_search["bits"]))

    print("Saving the processed product compound data...", end="")

    # Store all of the generated product fingerprints in a .pkl file.
    pd.DataFrame({"mol_id": list(range(0, len(product_pool_smiles))), "canonical_smiles": product_pool_smiles,
                  "mol_object": product_pool_mol, "ecfp_1024": ecfp_1024, "reaction_class": product_reaction_class}).\
        to_pickle(args.dataset_config.output_folder + "unique_products_pool.pkl")

    print("done.")


def extract_relevant_information(reaction_smiles, uq_reactant_mols_pool, uq_product_mols_pool, fp_params):
    """ Extracts the necessary information from a single mapped reaction SMILES string. """

    # Extract the canonical SMILES and RDKit Mol objects from the reaction SMILES string.
    reactant_smiles, _, product_smiles = parse_reaction_roles(reaction_smiles, as_what="canonical_smiles_no_maps")
    reactants, _, products = parse_reaction_roles(reaction_smiles, as_what="mol_no_maps")

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
    reactant_frags, product_frags = extract_info_from_reaction(reaction_smiles)

    # Iterate through all of the reactants and aggregate the specified data.
    for r_ind, reactant in enumerate(reactants):
        r_uq_mol_maps.append(uq_reactant_mols_pool.index(reactant_smiles[r_ind]))
        rr_smiles.append(reactant_frags[r_ind][0][0])
        rnr_smiles.append(reactant_frags[r_ind][1][0])
        rr_smols.append(reactant_frags[r_ind][0][2])
        rnr_smols.append(reactant_frags[r_ind][1][2])
        rr_smals.append(reactant_frags[r_ind][0][3])
        rnr_smals.append(reactant_frags[r_ind][1][3])
        rr_fps.append(construct_ecfp(reactant_frags[r_ind][0][2], radius=fp_params["radius"], bits=fp_params["bits"]))
        rnr_fps.append(construct_ecfp(reactant_frags[r_ind][1][2], radius=fp_params["radius"], bits=fp_params["bits"]))

    # Iterate through all of the products and aggregate the specified data.
    for p_ind, product in enumerate(products):
        p_uq_mol_maps.append(uq_product_mols_pool.index(product_smiles[p_ind]))
        pr_smiles.extend(product_frags[p_ind][0][0])
        pnr_smiles.extend(product_frags[p_ind][1][0])
        pr_smols.extend(product_frags[p_ind][0][2])
        pnr_smols.extend(product_frags[p_ind][1][2])
        pr_smals.extend(product_frags[p_ind][0][3])
        pnr_smals.extend(product_frags[p_ind][1][3])

        for pf in product_frags[p_ind][0][2]:
            pr_fps.append(construct_ecfp(pf, radius=fp_params["radius"], bits=fp_params["bits"]))
        for pf in product_frags[p_ind][1][2]:
            pnr_fps.append(construct_ecfp(pf, radius=fp_params["radius"], bits=fp_params["bits"]))

    # Return the extracted information.
    return r_uq_mol_maps, rr_smiles, rr_smols, rr_smals, rr_fps, rnr_smiles, rnr_smols, rnr_smals, rnr_fps,\
           p_uq_mol_maps, pr_smiles, pr_smols, pr_smals, pr_fps, pnr_smiles, pnr_smols, pnr_smals, pnr_fps


def expand_reaction_dataset(args):
    """ Standardizes and expands the original dataset with additional, useful information. The raw dataset needs to
        contain columns named 'id', 'rxn_smiles' and 'class' in which the values for the reaction identification, mapped
        reaction SMILES and reaction class are stored, respectively."""

    # Read the raw chemical reaction dataset and rename the fetched columns.
    raw_dataset = pd.read_csv(args.dataset_config.raw_dataset)[["id", "rxn_smiles", "class"]]
    raw_dataset.columns = ["patent_id", "reaction_smiles", "reaction_class"]

    # Create new columns to store the id's of the unique reactant and product molecules.
    raw_dataset["reactants_uq_mol_maps"], raw_dataset["products_uq_mol_maps"] = None, None
    # Create new columns to store the SMILES strings of the reactive parts of reactant and product molecules.
    raw_dataset["reactants_reactive_smiles"], raw_dataset["products_reactive_smiles"] = None, None
    # Create new columns to store the SMILES Mol objects of the reactive parts of reactant and product molecules.
    raw_dataset["reactants_reactive_smols"], raw_dataset["products_reactive_smols"] = None, None
    # Create new columns to store the SMARTS Mol objects of the reactive parts of reactant and product molecules.
    raw_dataset["reactants_reactive_smals"], raw_dataset["products_reactive_smals"] = None, None
    # Create new columns to store the fingerprints of the reactive parts of reactant and product molecules.
    raw_dataset["reactants_reactive_fps"], raw_dataset["products_reactive_fps"] = None, None

    # Create new columns to store the SMILES strings of the non-reactive parts of reactant and product molecules.
    raw_dataset["reactants_non_reactive_smiles"], raw_dataset["products_non_reactive_smiles"] = None, None
    # Create new columns to store the SMILES Mol objects of the non-reactive parts of reactant and product molecules.
    raw_dataset["reactants_non_reactive_smols"], raw_dataset["products_non_reactive_smols"] = None, None
    # Create new columns to store the SMARTS Mol objects of the non-reactive parts of reactant and product molecules.
    raw_dataset["reactants_non_reactive_smals"], raw_dataset["products_non_reactive_smals"] = None, None
    # Create new columns to store the fingerprints of the non-reactive parts of reactant and product molecules.
    raw_dataset["reactants_non_reactive_fps"], raw_dataset["products_non_reactive_fps"] = None, None

    # Read the previously generated unique molecule pools.
    reactant_pool = pd.read_pickle(args.dataset_config.output_folder +
                                   "unique_reactants_pool.pkl")["canonical_smiles"].values.tolist()
    product_pool = pd.read_pickle(args.dataset_config.output_folder +
                                  "unique_products_pool.pkl")["canonical_smiles"].values.tolist()

    # Iterate through all of the reactions and generate their unique molecule mapping for easier reactant retrieval in
    # the later stages of the approach.
    for row_ind, row in tqdm(raw_dataset.iterrows(), total=len(raw_dataset.index),
                             desc="Generating unique reactant and product compound representations"):

        # Extract the needed values from the reaction SMILES string.
        ruqmm, rrsm, rrso, rrsa, rrsf, rnsm, rnso, rnsa, rnsf, puqmm, prsm, prso, prsa, prsf, pnsm, pnso, pnsa, pnsf = \
            extract_relevant_information(row["reaction_smiles"], reactant_pool, product_pool,
                                         args.descriptor_config.similarity_search)

        # Assign the extracted values to the data frame.
        raw_dataset.at[row_ind, "reactants_uq_mol_maps"] = ruqmm

        raw_dataset.at[row_ind, "reactants_reactive_smiles"] = rrsm
        raw_dataset.at[row_ind, "reactants_reactive_smols"] = rrso
        raw_dataset.at[row_ind, "reactants_reactive_smals"] = rrsa
        raw_dataset.at[row_ind, "reactants_reactive_fps"] = rrsf

        raw_dataset.at[row_ind, "reactants_non_reactive_smiles"] = rnsm
        raw_dataset.at[row_ind, "reactants_non_reactive_smols"] = rnso
        raw_dataset.at[row_ind, "reactants_non_reactive_smals"] = rnsa
        raw_dataset.at[row_ind, "reactants_non_reactive_fps"] = rnsf

        raw_dataset.at[row_ind, "products_uq_mol_maps"] = puqmm

        raw_dataset.at[row_ind, "products_reactive_smiles"] = prsm
        raw_dataset.at[row_ind, "products_reactive_smols"] = prso
        raw_dataset.at[row_ind, "products_reactive_smals"] = prsa
        raw_dataset.at[row_ind, "products_reactive_fps"] = prsf

        raw_dataset.at[row_ind, "products_non_reactive_smiles"] = pnsm
        raw_dataset.at[row_ind, "products_non_reactive_smols"] = pnso
        raw_dataset.at[row_ind, "products_non_reactive_smals"] = pnsa
        raw_dataset.at[row_ind, "products_non_reactive_fps"] = pnsf

    print("Saving the generated compound data...", end="")

    # Save the final reaction dataset as in .pkl or .csv format.
    raw_dataset.to_pickle(args.dataset_config.output_folder + "final_training_dataset.pkl")

    print("done.")


def generate_dataset_splits(args):
    """ Generates training and test splits for the n-fold cross validation process in the ratio 80:20. """

    # Read the processed chemical reaction dataset.
    processed_dataset = pd.read_pickle(args.dataset_config.output_folder + "final_training_dataset.pkl")
    folds = [[] for _ in range(args.dataset_config.num_folds)]

    for cls in np.unique(processed_dataset["reaction_class"].values):
        # Select the subset of data with the respective class label.
        class_subset = processed_dataset.loc[processed_dataset["reaction_class"] == cls]

        # Shuffle this subset with a specified seed value.
        class_subset = class_subset.sample(frac=1, random_state=args.dataset_config.random_seed)

        # Split the subset into multiple folds and save the indices of the rows.
        for fold_index, current_fold in enumerate(np.array_split(class_subset.index.values,
                                                                 args.dataset_config.num_folds)):
            folds[fold_index].extend(current_fold.tolist())

    # Generate training and validation data and save all of the datasets.
    for fold_index, test_indices in enumerate(folds):
        print("Generating data for fold {}...".format(fold_index + 1), end="")

        # If a fold directory does nto exist for a specific fold, create it.
        directory_path = args.dataset_config.output_folder + "fold_{}/".format(fold_index + 1)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Split the remaining indices into training and validation sets.
        training_indices = set(processed_dataset.index.values).difference(test_indices)
        validation_indices = random.sample(training_indices,
                                           k=round(len(processed_dataset) * args.dataset_config.validation_split))
        training_indices = list(training_indices.difference(validation_indices))

        # Save all of the datasets for each respective fold.
        processed_dataset.iloc[training_indices, :].sort_values("reaction_class"). \
            to_pickle(directory_path + "training_data.pkl".format(fold_index + 1))
        processed_dataset.iloc[validation_indices, :].sort_values("reaction_class"). \
            to_pickle(directory_path + "validation_data.pkl".format(fold_index + 1))
        processed_dataset.iloc[test_indices, :].sort_values("reaction_class"). \
            to_pickle(directory_path + "test_data.pkl".format(fold_index + 1))

        print("done.")


def generate_fps_from_reaction_products(reaction_smiles, fp_data_configs):
    """ Generates specified fingerprints for the both reactive and non-reactive substructures of the reactant and
        product molecules that are the participating in the chemical reaction. """

    # Generate the RDKit Mol representations of the product molecules and generate the reaction cores.
    reactants, _, products = parse_reaction_roles(reaction_smiles, as_what="mol_no_maps")
    reaction_cores = get_reaction_core_atoms(reaction_smiles)

    # Separate the reaction cores if they consist out of multiple non-neighbouring parts.
    separated_cores = get_separated_cores(reaction_smiles, reaction_cores)

    # Define variables which will be used for storing the results.
    total_reactive_fps, total_non_reactive_fps = [], []

    # Iterate through the product molecules and generate fingerprints for all reactive and non-reactive substructures.
    for p_ind, product in enumerate(products):
        # Iterate through all of the dataset configurations.
        for fp_config in fp_data_configs:
            reactive_fps, non_reactive_fps = [], []
            # Generate fingerprints from the reactive substructures i.e. the reaction core(s).
            for core in separated_cores[1][p_ind]:
                # Generate reactive EC fingerprints and add them to the list.
                if fp_config["type"] == "ecfp":
                    reactive_fps.append(construct_ecfp(product, radius=fp_config["radius"], bits=fp_config["bits"],
                                                       from_atoms=core, output_type="np_array", as_type="np_float"))
                # Generate reactive HS fingerprints and add them to the list.
                else:
                    reactive_fps.append(construct_hsfp(product, radius=fp_config["radius"], bits=fp_config["bits"],
                                                       from_atoms=core, neighbourhood_ext=fp_config["ext"]))

            # Generate the extended environment of the reaction core.
            extended_core_env = get_atom_environment(reaction_cores[1][p_ind], product, degree=1)
            # Generate fingerprints from the non-reactive substructures i.e. non-reaction core substructures.
            for bond in product.GetBonds():
                # Generate the extended environment of the focus bond.
                extended_bond_env = get_bond_environment(bond, product, degree=1)

                # If the extended environment of the non-reactive substructure does not overlap with the extended
                # reaction core, generate a non-reactive fingerprint representation.
                if not extended_bond_env.intersection(extended_core_env):
                    # Generate non-reactive EC fingerprints and add them to the list.
                    if fp_config["type"] == "ecfp":
                        non_reactive_fps.append(
                            construct_ecfp(product, radius=fp_config["radius"], bits=fp_config["bits"],
                                           from_atoms=[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                                           output_type="np_array", as_type="np_float"))
                    # Generate non-reactive HS fingerprints and add them to the list.
                    else:
                        non_reactive_fps.append(
                            construct_hsfp(product, radius=fp_config["radius"], bits=fp_config["bits"],
                                           from_atoms=[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                                           neighbourhood_ext=fp_config["ext"]))

            # Append the generated fingerprints to the final list.
            total_reactive_fps.append(reactive_fps)
            total_non_reactive_fps.append(non_reactive_fps)

    # Return all of the generated fingerprints and labels.
    return total_reactive_fps, total_non_reactive_fps


def save_fingerprints_to_file(output_folder_path, fp_parameters, file_name, file_role, file_ext, fp_content, mode="w"):
    """ Generates a descriptive name and saves the file to the specified output location. """

    # Generate the name for the file.
    file_name = "{}_{}_{}_{}_{}.{}".format(fp_parameters["type"], fp_parameters["radius"], fp_parameters["bits"],
                                           file_name[0:-4], file_role, file_ext) \
        if fp_parameters["type"] == "ecfp" else "{}_{}_{}_{}_{}_{}.{}".format(fp_parameters["type"],
                                                                              fp_parameters["radius"],
                                                                              fp_parameters["ext"],
                                                                              fp_parameters["bits"], file_name[0:-4],
                                                                              file_role, file_ext)

    # Save the new file in the specified format.
    if mode == "w":
        if file_ext == "pkl":
            pd.DataFrame(fp_content).to_pickle(output_folder_path + "/" + file_name)
        elif file_ext == "csv":
            pd.DataFrame(fp_content).to_csv(output_folder_path + "/" + file_name)
        else:
            raise Exception("Extension needs to be either 'pkl' or 'csv'.")

    # Append the content to an already existing dataset.
    else:
        old_dataset = pd.read_pickle(output_folder_path + "/" + file_name) \
            if file_ext == "pkl" else pd.read_csv(output_folder_path + "/" + file_name)
        old_dataset = old_dataset.append(pd.DataFrame(fp_content))
        old_dataset.to_pickle(output_folder_path + "/" + file_name)


def generate_fingerprint_datasets(args):
    """ Generates fingerprint representations for all of the previously constructed data splits. """

    # Iterate through all of the generated 'n-fold' folders.
    for directory_name in os.listdir(args.dataset_config.output_folder):
        if "fold" in directory_name:
            # Create folder for the type of fingerprints dataset which is specified in the input parameters.
            fold_dir_path = args.dataset_config.output_folder + directory_name + "/"

            # Create folders for all of the fingerprint configurations.
            for fp_config in args.descriptor_config.model_training:
                if not os.path.exists(fold_dir_path + fp_config["folder_name"]):
                    os.makedirs(fold_dir_path + fp_config["folder_name"])

            # Read all of the dataset splits for the current fold.
            for file_name in os.listdir(fold_dir_path):
                if file_name.endswith(".pkl"):
                    current_dataset = pd.read_pickle(fold_dir_path + file_name)

                    reactive_fps = [[] for _ in range(0, len(args.descriptor_config.model_training))]
                    non_reactive_fps = [[] for _ in range(0, len(args.descriptor_config.model_training))]
                    mc_lab = []

                    # Iterate through all of the rows of each dataset.
                    for row_ind, row in tqdm(current_dataset.iterrows(), total=len(current_dataset.index),
                                             desc="Generating data for '{}' - '{}'".format(directory_name, file_name)):

                        # Fetch the reactive and non-reactive substructures from the products of this reaction.
                        r_fps, nr_fps = generate_fps_from_reaction_products(row["reaction_smiles"],
                                                                            args.descriptor_config.model_training)

                        # Generate multi-class labels because they are the same for every fingerprint.
                        mc_lab.extend(np.array([encode_one_hot(row["reaction_class"],
                                                               args.dataset_config.final_classes), ] * len(r_fps[0])))

                        # Iterate through all of the specified configurations.
                        for fpc_ind, fp_config in enumerate(args.descriptor_config.model_training):
                            # Append the reactive data and an equal amount of multi-class labels for the configuration.
                            reactive_fps[fpc_ind].extend(r_fps[fpc_ind])

                            # Append the non-reactive data for the configuration.
                            non_reactive_fps[fpc_ind].extend(nr_fps[fpc_ind])

                    # Save the reactive data and the labels, as well as the rest of the non-reactive data.
                    for fpc_ind, fp_config in enumerate(args.descriptor_config.model_training):
                        # Save the reactive data.
                        save_fingerprints_to_file(fold_dir_path + fp_config["folder_name"], fp_config, file_name, "r",
                                                  "pkl", reactive_fps[fpc_ind])

                        # Save the non-reactive data.
                        save_fingerprints_to_file(fold_dir_path + fp_config["folder_name"], fp_config, file_name, "nr",
                                                  "pkl", non_reactive_fps[fpc_ind])

                        # Save the binary and multi-class labels for the reactive parts of the data.
                        save_fingerprints_to_file(fold_dir_path + fp_config["folder_name"], fp_config, file_name, "bc",
                                                  "pkl", np.full((len(reactive_fps[fpc_ind]), 1), 1, np.float))
                        save_fingerprints_to_file(fold_dir_path + fp_config["folder_name"], fp_config, file_name, "mc",
                                                  "pkl", mc_lab)


def create_model_training_datasets(args):
    """ Aggregates the reactive and non-reactive parts to create the final input dataset for the network. """

    # Iterate through all of the generated 'n-fold' folders.
    for fold_dir in os.listdir(args.dataset_config.output_folder):
        if "fold" in fold_dir:
            fold_dir_path = args.dataset_config.output_folder + fold_dir + "/"

            # Iterate through all of the generated dataset variant folders in the current fold.
            for data_dir in os.listdir(fold_dir_path):
                if not data_dir.endswith(".pkl"):
                    data_dir_path = fold_dir_path + data_dir + "/"
                    print("Reading datasets from the '{}' folder.".format("/" + fold_dir + "/" + data_dir + "/"))

                    # Finally, iterate through all of the files in the current dataset variant folder and read the
                    # reactive and non-reactive parts.
                    for dataset_split in ["training", "validation", "test"]:
                        r_fp, nr_fp, r_bc, r_mc = None, None, None, None

                        for file_name in os.listdir(data_dir_path):
                            if dataset_split in file_name and "data_r" in file_name:
                                r_fp = pd.read_pickle(data_dir_path + file_name).values
                            if dataset_split in file_name and "data_nr" in file_name:
                                nr_fp = pd.read_pickle(data_dir_path + file_name).values
                            if dataset_split in file_name and "data_bc" in file_name:
                                r_bc = pd.read_pickle(data_dir_path + file_name).values
                            if dataset_split in file_name and "data_mc" in file_name:
                                r_mc = pd.read_pickle(data_dir_path + file_name).values

                        # Filter the negative samples to the amount of the highest populated positive class.
                        print("Filtering negative samples for the {} set...".format(dataset_split), end="")
                        nr_samples = sorted(Counter([np.argmax(r) for r in r_mc]).values(), reverse=True)[0]
                        nr_fp = nr_fp[get_n_most_frequent_rows(nr_fp, nr_samples)]

                        # Generate the labels for the saved non-reactive fingerprints.
                        nr_bc = np.full((len(nr_fp), 1), 0, np.float)
                        nr_mc = np.full((len(nr_fp), 11), 0, np.float)
                        nr_mc[:, 0] = 1.

                        print("done.")

                        # Aggregate the reactive and non-reactive fingerprints.
                        print("Aggregating and saving the data for the {} set...".format(dataset_split), end="")

                        x_fp = np.vstack((r_fp, nr_fp))
                        pd.to_pickle(pd.DataFrame(x_fp), data_dir_path + "x_{}.pkl".format(dataset_split))

                        print("done. Shape: {}".format(str(x_fp.shape)))

                        # Aggregate the reactive and non-reactive labels.
                        print("Aggregating and saving the labels for the {} set...".format(dataset_split), end="")

                        y_bc = np.vstack((r_bc, nr_bc))
                        pd.to_pickle(pd.DataFrame(y_bc), data_dir_path + "y_bc_{}.pkl".format(dataset_split))
                        y_mc = np.vstack((r_mc, nr_mc))
                        pd.to_pickle(pd.DataFrame(y_mc), data_dir_path + "y_mc_{}.pkl".format(dataset_split))

                        print("done. Shapes: {} and {}".format(str(y_mc.shape), str(y_bc.shape)))


def create_final_evaluation_dataset(args):

    fold_ind = 5
    fp_config = {"type": "hsfp", "radius": 2, "bits": 1024, "ext": 2, "folder_name": "hsfp_2_2_1024"}

    # Read the test dataset from the specified fold.
    test_dataset = pd.read_pickle(args.dataset_config.output_folder + "fold_{}/test_data.pkl".format(fold_ind))
    evaluation_data = []

    # Iterate through the d
    for row_ind, row in tqdm(test_dataset.iterrows(), total=len(test_dataset.index),
                             desc="Generating non-filtered version of the test dataset"):
        # Select only products from the reaction.
        _, _, products = parse_reaction_roles(row["reaction_smiles"], as_what="mol_no_maps")
        products_reaction_cores = get_reaction_core_atoms(row["reaction_smiles"])[1]

        for p_ind, product in enumerate(products):
            for bond in product.GetBonds():

                if fp_config["type"] == "ecfp":
                    bond_fp = construct_ecfp(product, radius=fp_config["radius"], bits=fp_config["bits"],
                                             from_atoms=[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                                             output_type="np_array", as_type="np_float")
                else:
                    bond_fp = construct_hsfp(product, radius=fp_config["radius"], bits=fp_config["bits"],
                                             from_atoms=[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                                             neighbourhood_ext=fp_config["ext"])

                if bond.GetBeginAtomIdx() in products_reaction_cores[p_ind] and \
                        bond.GetEndAtomIdx() in products_reaction_cores[p_ind]:
                    in_core = True
                else:
                    in_core = False

                evaluation_data.append([row["patent_id"], bond.GetIdx(), bond_fp, in_core, row["reaction_smiles"],
                                        row["reaction_class"], row["reactants_uq_mol_maps"]])

    data = pd.DataFrame(evaluation_data, columns=["patent_id", "bond_id", "bond_fp", "is_core", "reaction_smiles",
                                                  "reaction_class", "reactants_uq_mol_maps"])
    print(data.head(100))

    data.to_pickle(args.dataset_config.output_folder + "final_evaluation_dataset.pkl")
