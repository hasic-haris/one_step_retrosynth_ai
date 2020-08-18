"""
Author:      Hasic Haris (Phd Student @ Ishida Lab, Department of Computer Science, Tokyo Institute of Technology)
Created on:  February 21st, 2020
Description: This file contains functions for the generation and splitting of the source dataset.
"""
import pandas as pd
import numpy as np
import random
import os
import gc

from collections import Counter

from chem_methods.reactions import parse_reaction_roles
from chem_methods.reaction_cores import get_reaction_core_atoms, get_separated_cores
from chem_methods.fingerprints import construct_ecfp, construct_hsfp
from chem_methods.molecules import get_atom_environment, get_bond_environment
from data_methods.data_handling import get_n_most_frequent_rows, encode_one_hot
from chem_methods.reaction_analysis import extract_info_from_molecule, extract_info_from_reaction


# Done: 100%
def generate_unique_compound_pools(**kwargs):
    """Generate and store unique chemical compound (RDKit Canonical SMILES) pools of the reactants and products for a
    whole chemical reaction dataset. Specific details need to be specified as arguments."""

    # Define variables for unique reactant and product sets.
    reactant_pool, product_pool, reactant_pool_mol, product_pool_mol = [], [], [], []
    reactant_num_atoms, product_num_atoms, reactant_reaction_class, product_reaction_class = [], [], [], []

    # Iterate through all of the reactions and generate RDKit Canonical SMILES reactant and product sets. In this
    # dataset, reactions do not contain agents, so they are skipped when parsing the reaction roles.
    for row_ind, row in pd.read_csv(kwargs["source_dataset_path"]).iterrows():
        print("Generating canonical SMILES and Mol for row {}.".format(row_ind))

        # Extract the canonical SMILES from the reaction.
        reactants, _, products = parse_reaction_roles(row["rxn_smiles"], as_what="canonical_smiles_no_maps")

        [reactant_pool.append(reactant) for reactant in reactants]
        [product_pool.append(product) for product in products]

        # Extract the Mol objects from the reaction.
        reactants, _, products = parse_reaction_roles(row["rxn_smiles"], as_what="mol_no_maps")

        [reactant_pool_mol.append(reactant) for reactant in reactants]
        [reactant_num_atoms.append(len(reactant.GetAtoms())) for reactant in reactants]
        [reactant_reaction_class.append(row["class"]) for _ in reactants]

        [product_pool_mol.append(product) for product in products]
        [product_num_atoms.append(len(product.GetAtoms())) for product in products]
        [product_reaction_class.append(row["class"]) for _ in products]

    print("Aggregating the reaction class values for the reactants...")
    for rind, reactant in enumerate(reactant_pool):
        if type(reactant_reaction_class[rind]) == set:
            continue

        same_reactant_rows = [i for i, x in enumerate(reactant_pool) if x == reactant]
        class_values = [y for j, y in enumerate(reactant_reaction_class) if j in same_reactant_rows]

        for k in same_reactant_rows:
            reactant_reaction_class[k] = set(class_values)

    print("Aggregating the reaction class values for the products...")
    for pind, product in enumerate(product_pool):
        if type(product_reaction_class[pind]) == set:
            continue

        same_product_rows = [i for i, x in enumerate(product_pool) if x == product]
        class_values = [y for j, y in enumerate(product_reaction_class) if j in same_product_rows]

        for k in same_product_rows:
            product_reaction_class[k] = set(class_values)

    # Filter out duplicate reactant molecules from the reactant and product sets.
    reactant_pool, reactants_uq_ind = np.unique(reactant_pool, return_index=True)
    product_pool, products_uq_ind = np.unique(product_pool, return_index=True)

    # Apply the unique indices to the list of Mol objects.
    reactant_pool_mol = np.array(reactant_pool_mol)[reactants_uq_ind].tolist()
    reactant_num_atoms = np.array(reactant_num_atoms)[reactants_uq_ind].tolist()
    reactant_reaction_class = np.array(reactant_reaction_class)[reactants_uq_ind].tolist()

    product_pool_mol = np.array(product_pool_mol)[products_uq_ind].tolist()
    product_num_atoms = np.array(product_num_atoms)[products_uq_ind].tolist()
    product_reaction_class = np.array(product_reaction_class)[products_uq_ind].tolist()

    # In order to save time during reactant retrieval in the later stages, pre-generate the needed molecular
    # descriptors. In this case, these will be fingerprints for bit size 1024.
    ecfp_1024 = []

    # Iterate through all of the unique reactant molecules and pre-generate the planned fingerprints.
    for uqr_ind, uq_reactant in enumerate(reactant_pool):
        print("Generating reactant fingerprints for row {}.".format(uqr_ind))
        ecfp_1024.append(construct_ecfp(uq_reactant, radius=kwargs["similarity_search_fp_config"]["radius"], bits=1024))

    # Store all of the generated reactant fingerprints in a pickle file.
    pd.DataFrame({"id": list(range(0, len(reactant_pool))), "canonical_smiles": reactant_pool,
                  "mol_object": reactant_pool_mol, "ecfp_1024": ecfp_1024, "num_atoms": reactant_num_atoms,
                  "reaction_class": reactant_reaction_class}). \
        to_pickle(kwargs["output_folder"] + "unique_reactants_pool.pkl")

    ecfp_1024 = []

    # Iterate through all of the unique product molecules and pre-generate the planned fingerprints.
    for uqp_ind, uq_product in enumerate(product_pool):
        print("Generating product fingerprints for row {}.".format(uqp_ind))
        ecfp_1024.append(construct_ecfp(uq_product, radius=kwargs["fp_sim_specs"]["radius"], bits=1024))

    # Store all of the generated product fingerprints in a pickle file.
    pd.DataFrame({"id": list(range(0, len(product_pool))), "canonical_smiles": product_pool,
                  "mol_object": product_pool_mol, "ecfp_1024": ecfp_1024, "num_atoms": product_num_atoms,
                  "reaction_class": product_reaction_class}). \
        to_pickle(kwargs["output_folder"] + "unique_products_pool.pkl")


# Done: 100%
def extract_relevant_information(reaction_smiles, uq_reactant_mols_pool, uq_product_mols_pool, fp_params):
    """ Extract all possible information from a single reaction SMILES string. """

    # Extract the molecules and canonical SMILES from the reaction SMILES string.
    reactants, _, products = parse_reaction_roles(reaction_smiles, as_what="mol_no_maps")
    reactant_smiles, _, product_smiles = parse_reaction_roles(reaction_smiles, as_what="canonical_smiles_no_maps")

    # Sort the reactants and products in descending order by number of atoms so the largest reactants is always first.
    reactants, reactant_smiles = zip(*sorted(zip(reactants, reactant_smiles), key=lambda k: len(k[0].GetAtoms()),
                                             reverse=True))
    products, product_smiles = zip(*sorted(zip(products, product_smiles), key=lambda k: len(k[0].GetAtoms()),
                                           reverse=True))

    # Create variables for all the information that will be generated.
    r_uq_mol_maps, rr_smiles, rr_smols, rr_smals, rr_fps, rnr_smiles, rnr_smols, rnr_smals, rnr_fps = \
        [], [], [], [], [], [], [], [], []
    p_uq_mol_maps, pr_smiles, pr_smols, pr_smals, pr_fps, pnr_smiles, pnr_smols, pnr_smals, pnr_fps = \
        [], [], [], [], [], [], [], [], []

    # Find the reactive and non-reactive parts of the reactant and product molecules.
    reactant_frags, product_frags = extract_info_from_reaction(reaction_smiles)

    # Iterate through all of the reactants and aggregate the specified data.
    for rind, reactant in enumerate(reactants):
        r_uq_mol_maps.append(uq_reactant_mols_pool.index(reactant_smiles[rind]))
        rr_smiles.append(reactant_frags[rind][0][0])
        rnr_smiles.append(reactant_frags[rind][1][0])
        rr_smols.append(reactant_frags[rind][0][2])
        rnr_smols.append(reactant_frags[rind][1][2])
        rr_smals.append(reactant_frags[rind][0][3])
        rnr_smals.append(reactant_frags[rind][1][3])

        rr_fps.append(construct_ecfp(reactant_frags[rind][0][2], radius=fp_params["radius"], bits=fp_params["bits"]))
        rnr_fps.append(construct_ecfp(reactant_frags[rind][1][2], radius=fp_params["radius"], bits=fp_params["bits"]))

    # Iterate through all of the products and aggregate the specified data.
    for pind, product in enumerate(products):
        p_uq_mol_maps.append(uq_product_mols_pool.index(product_smiles[pind]))
        pr_smiles.extend(product_frags[pind][0][0])
        pnr_smiles.extend(product_frags[pind][1][0])
        pr_smols.extend(product_frags[pind][0][2])
        pnr_smols.extend(product_frags[pind][1][2])
        pr_smals.extend(product_frags[pind][0][3])
        pnr_smals.extend(product_frags[pind][1][3])

        for pf in product_frags[pind][0][2]:
            pr_fps.append(construct_ecfp(pf, radius=fp_params["radius"], bits=fp_params["bits"]))
        for pf in product_frags[pind][1][2]:
            pnr_fps.append(construct_ecfp(pf, radius=fp_params["radius"], bits=fp_params["bits"]))

    # Return the extracted information.
    return r_uq_mol_maps, rr_smiles, rr_smols, rr_smals, rr_fps, rnr_smiles, rnr_smols, rnr_smals, rnr_fps, \
           p_uq_mol_maps, pr_smiles, pr_smols, pr_smals, pr_fps, pnr_smiles, pnr_smols, pnr_smals, pnr_fps


# Done: 100%
def expand_reaction_dataset(**kwargs):
    """ Expand the original dataset with generated data that will be used later in the project. """

    # Read the source dataset and rename the fetched columns.
    source_dataset = pd.read_csv(kwargs["source_dataset_path"])[["id", "rxn_smiles", "class"]]
    source_dataset.columns = ["patent_id", "reaction_smiles", "reaction_class"]

    # Create new columns to store the id's of the unique reactant and product molecules.
    source_dataset["reactants_uq_mol_maps"], source_dataset["products_uq_mol_maps"] = None, None

    # Create new columns to store the SMILES strings of the reactive parts of reactant and product molecules.
    source_dataset["reactants_reactive_smiles"], source_dataset["products_reactive_smiles"] = None, None
    # Create new columns to store the SMILES 'Mol' objects of the reactive parts of reactant and product molecules.
    source_dataset["reactants_reactive_smols"], source_dataset["products_reactive_smols"] = None, None
    # Create new columns to store the SMARTS 'Mol' objects of the reactive parts of reactant and product molecules.
    source_dataset["reactants_reactive_smals"], source_dataset["products_reactive_smals"] = None, None
    # Create new columns to store the fingerprints of the reactive parts of reactant and product molecules.
    source_dataset["reactants_reactive_fps"], source_dataset["products_reactive_fps"] = None, None

    # Create new columns to store the SMILES strings of the non-reactive parts of reactant and product molecules.
    source_dataset["reactants_non_reactive_smiles"], source_dataset["products_non_reactive_smiles"] = None, None
    # Create new columns to store the SMILES 'Mol' objects of the non-reactive parts of reactant and product molecules.
    source_dataset["reactants_non_reactive_smols"], source_dataset["products_non_reactive_smols"] = None, None
    # Create new columns to store the SMARTS 'Mol' objects of the non-reactive parts of reactant and product molecules.
    source_dataset["reactants_non_reactive_smals"], source_dataset["products_non_reactive_smals"] = None, None
    # Create new columns to store the fingerprints of the non-reactive parts of reactant and product molecules.
    source_dataset["reactants_non_reactive_fps"], source_dataset["products_non_reactive_fps"] = None, None

    # Read the previously generated unique molecule pools.
    reactant_pool = pd.read_pickle(kwargs["output_folder"] +
                                   "unique_reactants_pool.pkl")["reactant_canonical_smiles"].values.tolist()
    product_pool = pd.read_pickle(kwargs["output_folder"] +
                                  "unique_products_pool.pkl")["product_canonical_smiles"].values.tolist()

    # Iterate through all of the reactions and generate their unique molecule mapping for easier reactant retrieval in
    # the later stages of the approach.
    for ind, row in source_dataset.iterrows():
        print("Generating unique molecules mapping for row {}.".format(ind))

        # Extract the needed values from the reaction smiles string.
        ruqmm, rrsm, rrso, rrsa, rrsf, rnsm, rnso, rnsa, rnsf, puqmm, prsm, prso, prsa, prsf, pnsm, pnso, pnsa, pnsf = \
            extract_relevant_information(row["reaction_smiles"], reactant_pool, product_pool,
                                         kwargs["similarity_search_fp_config"])

        # Assign the extracted values to the data frame.
        source_dataset.at[ind, "reactants_uq_mol_maps"] = ruqmm

        source_dataset.at[ind, "reactants_reactive_smiles"] = rrsm
        source_dataset.at[ind, "reactants_reactive_smols"] = rrso
        source_dataset.at[ind, "reactants_reactive_smals"] = rrsa
        source_dataset.at[ind, "reactants_reactive_fps"] = rrsf

        source_dataset.at[ind, "reactants_non_reactive_smiles"] = rnsm
        source_dataset.at[ind, "reactants_non_reactive_smols"] = rnso
        source_dataset.at[ind, "reactants_non_reactive_smals"] = rnsa
        source_dataset.at[ind, "reactants_non_reactive_fps"] = rnsf

        source_dataset.at[ind, "products_uq_mol_maps"] = puqmm

        source_dataset.at[ind, "products_reactive_smiles"] = prsm
        source_dataset.at[ind, "products_reactive_smols"] = prso
        source_dataset.at[ind, "products_reactive_smals"] = prsa
        source_dataset.at[ind, "products_reactive_fps"] = prsf

        source_dataset.at[ind, "products_non_reactive_smiles"] = pnsm
        source_dataset.at[ind, "products_non_reactive_smols"] = pnso
        source_dataset.at[ind, "products_non_reactive_smals"] = pnsa
        source_dataset.at[ind, "products_non_reactive_fps"] = pnsf

    # Save the final reaction dataset as in .pkl or .csv format.
    source_dataset.to_pickle(kwargs["output_folder"] + "final_reaction_dataset.pkl")
    source_dataset.to_csv(kwargs["output_folder"] + "final_reaction_dataset.csv", index=False)


# Done: 100%
def generate_dataset_splits(**kwargs):
    """ Split the dataset for n-fold cross validation sets for training, validation and testing (70:10:20). """

    # Read the source dataset.
    source_dataset = pd.read_pickle(kwargs["output_folder"] + "final_reaction_dataset.pkl")
    folds = [[] for _ in range(kwargs["num_folds"])]

    for cls in np.unique(source_dataset["reaction_class"].values):
        # Select the subset of data with the respective class label.
        class_subset = source_dataset.loc[source_dataset["reaction_class"] == cls]
        # Shuffle this subset with a specified seed value.
        class_subset = class_subset.sample(frac=1, random_state=kwargs["random_seed"])
        # Split the subset into multiple folds and save the indices of the rows.
        for fold_index, current_fold in enumerate(np.array_split(class_subset.index.values, kwargs["num_folds"])):
            folds[fold_index].extend(current_fold.tolist())

    # Generate training and validation data and save all of the datasets.
    for fold_index, test_indices in enumerate(folds):
        print("Generating data for fold {}...".format(fold_index + 1))

        # If a fold directory does nto exist for a specific fold, create it.
        directory_path = kwargs["output_folder"] + "fold_{}/".format(fold_index + 1)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Split the remaining indices into training and validation sets.
        training_indices = set(source_dataset.index.values).difference(test_indices)
        validation_indices = random.sample(training_indices, k=round(len(source_dataset) * kwargs["validation_split"]))
        training_indices = list(training_indices.difference(validation_indices))

        # Save all of the datasets for each respective fold.
        source_dataset.iloc[training_indices, :].sort_values("reaction_class"). \
            to_pickle(directory_path + "training_data.pkl".format(fold_index + 1))
        source_dataset.iloc[validation_indices, :].sort_values("reaction_class"). \
            to_pickle(directory_path + "validation_data.pkl".format(fold_index + 1))
        source_dataset.iloc[test_indices, :].sort_values("reaction_class"). \
            to_pickle(directory_path + "test_data.pkl".format(fold_index + 1))


# Done: 100%
def generate_fps_from_reaction_products(reaction_smiles, fp_data_configs):
    """ Generate specified fingerprints for the both reactive and non-reactive substructures of the reactant and product
    molecules that are the participating in the chemical reaction. """

    # Generate the RDKit 'Mol' representations of the product molecules and generate the reaction cores.
    reactants, _, products = parse_reaction_roles(reaction_smiles, as_what="mol_no_maps")
    reaction_cores = get_reaction_core_atoms(reaction_smiles)

    # Separate the reaction cores if they consist out of multiple non-neighbouring parts.
    separated_cores = get_separated_cores(reaction_smiles, reaction_cores)

    # Define variables which will be used for storing the results.
    total_reactive_fps, total_non_reactive_fps = [], []

    # Iterate through the product molecules and generate fingerprints for all reactive and non-reactive substructures.
    for pind, product in enumerate(products):
        # Iterate through all of the dataset configurations.
        for fp_config in fp_data_configs:
            reactive_fps, non_reactive_fps = [], []
            # Generate fingerprints from the reactive substructures i.e. the reaction core(s).
            for core in separated_cores[1][pind]:
                # Generate reactive EC fingerprints and add them to the list.
                if fp_config["type"] == "ecfp":
                    reactive_fps.append(construct_ecfp(product, radius=fp_config["radius"], bits=fp_config["bits"],
                                                       from_atoms=core, output_type="np_array", as_type="np_float"))
                # Generate reactive HS fingerprints and add them to the list.
                else:
                    reactive_fps.append(construct_hsfp(product, radius=fp_config["radius"], bits=fp_config["bits"],
                                                       from_atoms=core, nghb_size=fp_config["ext"]))

            # Generate the extended environment of the reaction core.
            extended_core_env = get_atom_environment(reaction_cores[1][pind], product, degree=1)
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
                                           nghb_size=fp_config["ext"]))

            # Append the generated fingerprints to the final list.
            total_reactive_fps.append(reactive_fps)
            total_non_reactive_fps.append(non_reactive_fps)

    # Return all of the generated fingerprints and labels.
    return total_reactive_fps, total_non_reactive_fps


# Done: 100%
def save_fingerprints_to_file(output_folder_path, fp_parameters, file_name, file_role, file_ext, fp_content, mode="w"):
    """ Generate a descriptive name and save the file to the specified output location. """

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


# Done: 100%
def process_non_reactive_fingerprints(folder_path, fp_params, file_name, file_role, file_extension, keep_pct=0.2):
    """ Pick the 20% of the most frequent non-reactive rows from each group and aggregate them. """

    # The first file needs an indication to also create the dataset.
    done_ctr = 0

    for item_name in os.listdir(folder_path):
        # Read only files which contain "_nr_" mark in their name.
        if file_name[0:-4] in item_name and file_role in item_name:
            print("Aggregating the '{}' file.".format(item_name))
            nr_fps = pd.read_pickle(folder_path + "/" + item_name).values
            # Convert the list of the non-reactive substructures to an np array for easy list indexing and select only
            # the n most frequent rows.
            nr_fps = nr_fps[get_n_most_frequent_rows(nr_fps, round(len(nr_fps) * keep_pct))].tolist()
            mode = "w" if done_ctr == 0 else "a"
            # Save the filtered non-reactive fingerprints.
            save_fingerprints_to_file(folder_path, fp_params, file_name, "nr", file_extension, nr_fps, mode=mode)
            # Delete the temporary file.
            print("Deleting the '{}' file.".format(item_name))
            os.remove(folder_path + "/" + item_name)
            done_ctr += 1


# Done: 100%
def generate_fingerprint_datasets(**kwargs):
    """ Generate fingerprint representations for all of the previously constructed data splits. """

    # Iterate through all of the generated 'n-fold' folders.
    for directory_name in os.listdir(kwargs["output_folder"]):
        if "fold" in directory_name:
            # Create folder for the type of fingerprints dataset which is specified in the input parameters.
            fold_dir_path = kwargs["output_folder"] + directory_name + "/"

            # Create folders for all of the fingerprint configurations.
            for fp_config in kwargs["fp_data_configs"]:
                if not os.path.exists(fold_dir_path + fp_config["folder_name"]):
                    os.makedirs(fold_dir_path + fp_config["folder_name"])

            # Read all of the dataset splits for the current fold.
            print("Reading files from the '{}' folder.".format(directory_name))

            for file_name in os.listdir(fold_dir_path):
                if file_name.endswith(".pkl"):
                    print("Generating data for the '{}' file.".format(file_name))
                    curr_dataset = pd.read_pickle(fold_dir_path + file_name)

                    reactive_fps = [[] for _ in range(0, len(kwargs["fp_data_configs"]))]
                    non_reactive_fps = [[] for _ in range(0, len(kwargs["fp_data_configs"]))]
                    mc_lab, row_ctr = [], 0

                    # Iterate through all of the rows of each dataset.
                    for row_ind, row in curr_dataset.iterrows():
                        print("Currently generating data for row {}/{}.".format(row_ctr, len(curr_dataset)))
                        # Fetch the reactive and non-reactive substructures from the products of this reaction.
                        r_fps, nr_fps = generate_fps_from_reaction_products(row["reaction_smiles"],
                                                                            kwargs["fp_data_configs"])

                        # Generate multi-class labels because they are the same for every fingerprint.
                        mc_lab.extend(np.array([encode_one_hot(row["reaction_class"], kwargs["reaction_classes"]), ]
                                               * len(r_fps[0])))

                        # Iterate through all of the specified configurations.
                        for fpc_ind, fp_config in enumerate(kwargs["fp_data_configs"]):
                            # Append the reactive data and an equal amount of multi-class labels for the configuration.
                            reactive_fps[fpc_ind].extend(r_fps[fpc_ind])
                            # Append the non-reactive data for the configuration.
                            non_reactive_fps[fpc_ind].extend(nr_fps[fpc_ind])

                            # Since the there are too many entries, save the non-reactive data cca. every 5000 rows.
                            if row_ctr != 0 and row_ctr % 5005 == 0:
                                save_fingerprints_to_file(fold_dir_path + fp_config["folder_name"], fp_config,
                                                          file_name, "nr_{}".format(row_ctr), "pkl",
                                                          non_reactive_fps[fpc_ind])
                                non_reactive_fps[fpc_ind] = []

                        row_ctr += 1

                    # Save the reactive data and the labels, as well as the rest of the non-reactive data.
                    for fpc_ind, fp_config in enumerate(kwargs["fp_data_configs"]):
                        # Save the reactive data.
                        save_fingerprints_to_file(fold_dir_path + fp_config["folder_name"], fp_config, file_name, "r",
                                                  "pkl", reactive_fps[fpc_ind])

                        # Save the rest of the non-reactive data.
                        save_fingerprints_to_file(fold_dir_path + fp_config["folder_name"], fp_config, file_name,
                                                  "nr_{}".format(len(curr_dataset)), "pkl", non_reactive_fps[fpc_ind])

                        # Save the binary and multi-class labels for the reactive parts of the data.
                        save_fingerprints_to_file(fold_dir_path + fp_config["folder_name"], fp_config, file_name,
                                                  "bc", "pkl", np.full((len(reactive_fps[fpc_ind]), 1), 1, np.float))
                        save_fingerprints_to_file(fold_dir_path + fp_config["folder_name"], fp_config, file_name,
                                                  "mc", "pkl", mc_lab)

                        gc.collect()

                    # Finally, filter out the top-n frequent non-reactive fingerprints for each configuration.
                    for fpc_ind, fp_config in enumerate(kwargs["fp_data_configs"]):
                        process_non_reactive_fingerprints(fold_dir_path + fp_config["folder_name"], fp_config,
                                                          file_name, "nr_", "pkl")


# Done: 100%
def create_final_fingerprint_datasets(**kwargs):
    """ Finally aggregate the reactive and non-reactive parts to create the final input dataset for the network. """

    # Iterate through all of the generated 'n-fold' folders.
    for fold_dir in os.listdir(kwargs["output_folder"]):
        if "fold" in fold_dir:
            fold_dir_path = kwargs["output_folder"] + fold_dir + "/"

            # Iterate through all of the generated dataset variant folders in the current fold.
            for data_dir in os.listdir(fold_dir_path):
                if not data_dir.endswith(".pkl"):
                    data_dir_path = fold_dir_path + data_dir + "/"
                    print("Reading files from the '{}' folder.".format(data_dir_path))

                    # Finally, iterate through all of the files in the current dataset variant folder and read the
                    # reactive and non-reactive parts.
                    for dataset_split in ["training", "validation", "test"]:
                        r_fp, nr_fp, r_bc, r_mc = None, None, None, None

                        for file_name in os.listdir(data_dir_path):
                            if dataset_split in file_name and "r" in file_name:
                                r_fp = pd.read_pickle(data_dir_path + file_name).values
                            if dataset_split in file_name and "nr" in file_name:
                                nr_fp = pd.read_pickle(data_dir_path + file_name).values
                            if dataset_split in file_name and "bc" in file_name:
                                r_bc = pd.read_pickle(data_dir_path + file_name).values
                            if dataset_split in file_name and "mc" in file_name:
                                r_mc = pd.read_pickle(data_dir_path + file_name).values

                        # Filter the negative samples to the amount of the highest populated positive class.
                        print("Filtering negative samples for the {} set...".format(dataset_split))
                        nr_samples = sorted(Counter([np.argmax(r) for r in r_mc]).values(), reverse=True)[0]
                        nr_fp = nr_fp[get_n_most_frequent_rows(nr_fp, nr_samples)]

                        # Generate the labels for the saved non-reactive fingerprints.
                        nr_bc = np.full((len(nr_fp), 1), 0, np.float)
                        nr_mc = np.full((len(nr_fp), 11), 0, np.float)
                        nr_mc[:, 0] = 1.

                        # Aggregate the reactive and non-reactive fingerprints.
                        print("Aggregating and saving the data for the {} set...".format(dataset_split))
                        x_fp = np.vstack((r_fp, nr_fp))
                        pd.to_pickle(pd.DataFrame(x_fp), data_dir_path + "x_{}.pkl".format(dataset_split))

                        # Aggregate the reactive and non-reactive labels.
                        print("Aggregating and saving the labels for the {} set...".format(dataset_split))
                        y_bc = np.vstack((r_bc, nr_bc))
                        pd.to_pickle(pd.DataFrame(y_bc), data_dir_path + "y_bc_{}.pkl".format(dataset_split))
                        y_mc = np.vstack((r_mc, nr_mc))
                        pd.to_pickle(pd.DataFrame(y_mc), data_dir_path + "y_mc_{}.pkl".format(dataset_split))


# Done: 100%
def save_pd_to_file(folder_path, pd_content, file_name, file_ext):
    if not os.path.exists(folder_path + file_name + file_ext):
        pd.to_pickle(pd_content, folder_path + file_name + file_ext)
    else:
        old_dataset = pd.read_pickle(folder_path + file_name + file_ext) \
            if file_ext == ".pkl" else pd.read_csv(folder_path + file_name)

        old_dataset = old_dataset.append(pd_content)

        old_dataset.to_pickle(folder_path + file_name + file_ext)


# Done: 33%
def generate_pipeline_test_dataset(fold_ind, **kwargs):
    """ Generate a dataset for the final pipeline. This dataset differs from others because all substructures are
    included and it also contains information about the correct core and correct reactants. """

    # Define the information that is going to be in the final dataset.
    dc_fps, dc_labels, ndc_fps, ndc_labels = np.array([]), np.array([]), np.array([]), np.array([])
    dc_ids, dc_frags, dc_maps, ndc_ids, ndc_frags, row_ctr = [], [], [], [], [], 0

    # Read the test dataset from the specified fold.
    curr_dataset = pd.read_pickle(kwargs["output_folder"] + "fold_{}/test_data.pkl".format(fold_ind))

    # Iterate through all of the rows of the dataset.
    for row_ind, row in curr_dataset.iterrows():
        print("Currently generating data for row {}/{}...".format(row_ctr, len(curr_dataset)))

        # Generate the RDKit 'Mol' representations of the product molecules and generate the reaction cores.
        _, _, products = parse_reaction_roles(row["reaction_smiles"], as_what="mol_no_maps")
        reaction_cores = get_reaction_core_atoms(row["reaction_smiles"])
        separated_cores = get_separated_cores(row["reaction_smiles"], reaction_cores)

        # Iterate through all target molecules present in the reaction.
        for p_ind, product in enumerate(products):
            # Iterate through all bonds of a single target molecule.
            for bond in product.GetBonds():
                # Construct a unique identification string for every entry.
                id_string = "r{}_m{}_b{}".format(row_ind, p_ind, bond.GetIdx())

                # Define the potential disconnection site which consists out of bond atoms.
                dc_site_atoms = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]

                # Construct the specified fingerprint for the potential disconnection site, according to specifications.
                if kwargs["best_fp_config"]["type"] == "ecfp":
                    bond_fp = construct_ecfp(product, radius=kwargs["best_fp_config"]["radius"],
                                             bits=kwargs["best_fp_config"]["bits"], from_atoms=dc_site_atoms,
                                             output_type="np_array", as_type="np_float")
                else:
                    bond_fp = construct_hsfp(product, radius=kwargs["best_fp_config"]["radius"],
                                             bits=kwargs["best_fp_config"]["bits"], from_atoms=dc_site_atoms,
                                             nghb_size=kwargs["best_fp_config"]["ext"])

                # Save all necessary information for the real disconnection sites.
                bond_fp_label, uq_maps = None, None
                for sep_core in separated_cores[1][p_ind]:
                    if (len(sep_core) == 1 and
                        (bond.GetBeginAtomIdx() in sep_core or bond.GetEndAtomIdx() in sep_core)) or \
                            (len(sep_core) > 1 and
                             (bond.GetBeginAtomIdx() in sep_core and bond.GetEndAtomIdx() in sep_core)):
                        dc_fps = bond_fp if len(dc_fps) == 0 else np.vstack((dc_fps, bond_fp))
                        bond_fp_label = encode_one_hot(row["reaction_class"], kwargs["reaction_classes"])
                        dc_labels = np.array(bond_fp_label) if len(dc_labels) == 0 \
                            else np.vstack((dc_labels, np.array(bond_fp_label)))
                        dc_ids.append(id_string)
                        dc_frags.append(extract_info_from_molecule(product, dc_site_atoms, role="product"))
                        dc_maps.append(row["reactants_uq_mol_maps"])

                # Save all necessary information for the non-disconnection sites.
                if bond_fp_label is None:
                    ndc_fps = bond_fp if len(ndc_fps) == 0 else np.vstack((ndc_fps, bond_fp))
                    bond_fp_label = encode_one_hot(0, kwargs["reaction_classes"])
                    ndc_labels = np.array(bond_fp_label) if len(ndc_labels) == 0 \
                        else np.vstack((ndc_labels, np.array(bond_fp_label)))
                    ndc_ids.append(id_string)
                    ndc_frags.append(extract_info_from_molecule(product, dc_site_atoms, role="product"))

        if row_ctr != 0 and row_ctr % 500 == 0:
            # Store the fingerprints and labels.
            save_pd_to_file(kwargs["output_folder"] + "pipeline_dataset/", pd.DataFrame(dc_fps), "x_dc", ".pkl")
            save_pd_to_file(kwargs["output_folder"] + "pipeline_dataset/", pd.DataFrame(dc_labels), "y_dc", ".pkl")
            save_pd_to_file(kwargs["output_folder"] + "pipeline_dataset/", pd.DataFrame(ndc_fps), "x_ndc", ".pkl")
            save_pd_to_file(kwargs["output_folder"] + "pipeline_dataset/", pd.DataFrame(ndc_labels), "y_ndc", ".pkl")

            # Store the generated additional information.
            save_pd_to_file(kwargs["output_folder"] + "pipeline_dataset/",
                            pd.DataFrame({"id": dc_ids, "mol_frags": dc_frags, "uq_maps": dc_maps}), "info_dc", ".pkl")
            save_pd_to_file(kwargs["output_folder"] + "pipeline_dataset/",
                            pd.DataFrame({"id": ndc_ids, "mol_frags": ndc_frags, "uq_maps": None}), "info_ndc", ".pkl")

            dc_fps, dc_labels, ndc_fps, ndc_labels = np.array([]), np.array([]), np.array([]), np.array([])
            dc_ids, dc_frags, dc_maps, ndc_ids, ndc_frags = [], [], [], [], []

        row_ctr += 1
