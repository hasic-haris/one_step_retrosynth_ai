"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 21st, 2020
Description: This file contains necessary functions for the retrieval and scoring of potential reactant candidates.
"""

import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

from chemistry_methods.fingerprints import bulk_dice_similarity, bulk_tanimoto_similarity, bulk_tversky_similarity
from model_methods.model_construction import apply_model


def fetch_similar_compounds(synthon_mol, synthon_fp, reactant_search_pool, recorded_rxn_classes, unique_class_groups,
                            use_class=False, cut_off=0.1, top_n=None, fetch_priority="similar",
                            similarity_metric="tanimoto", a=0.5, b=1.0):
    """ Returns the 'top_n' similar compounds from a specific 'search_pool' for a given pattern compound 'synthon_mol'.
        The given pattern compound should be a SMARTS generated RDKit Mol object in order to enable superstructure, but
        if it is not, the method will return structurally similar compounds anyway. """

    # Depending on the available reaction class information, narrow down the search pool.
    if use_class and type(recorded_rxn_classes) == int:
        reactant_search_pool = reactant_search_pool[reactant_search_pool.index.isin(unique_class_groups
                                                                                    [recorded_rxn_classes])]
    elif use_class and type(recorded_rxn_classes) == list:
        reactant_search_pool = reactant_search_pool[reactant_search_pool.index.isin(recorded_rxn_classes)]

    # Create subsets of the search pool representations for easier processing.
    id_search_pool = reactant_search_pool["mol_id"].values.tolist()
    mol_search_pool = reactant_search_pool["mol_object"].values.tolist()
    fp_search_pool = reactant_search_pool["ecfp_1024"].values.tolist()

    # Define and apply the specified similarity metric.
    if similarity_metric == "dice":
        similarity_values = bulk_dice_similarity(synthon_fp, fp_search_pool)
    elif similarity_metric == "tanimoto":
        similarity_values = bulk_tanimoto_similarity(synthon_fp, fp_search_pool)
    elif similarity_metric == "tversky":
        similarity_values = bulk_tversky_similarity(synthon_fp, fp_search_pool, a=a, b=b)
    else:
        raise Exception("Only the 'dice', 'tanimoto' and 'tversky' similarity metrics are currently implemented.")

    # Sort the similarity values and get the indices of the sorted array.
    sim_indices = sorted([(sim_ind, sim_val) for sim_ind, sim_val in enumerate(similarity_values)
                          if sim_val >= cut_off], key=lambda k: k[1], reverse=True)

    # If the cut-off values left the list of indices empty, it implies that the fragment is too small to find any
    # significant reactant match. Because of that, go through the list of frequent small molecule reactants from each
    # class and find the best fitting option.
    # if len(sim_indices) == 0:
    #    sim_indices =

    # Return the Top-N candidates based on the applied metric, regardless of substructure matching.
    if fetch_priority == "similar":
        return [(id_search_pool[sim_ind[0]], sim_ind[1]) for sim_ind in sim_indices][:top_n]

    # Return the Top-N candidates based on the applied metric, only if they're superstructures to the pattern molecule.
    elif fetch_priority == "superstructures":
        retrieved_candidates = []

        # Fill the list of candidates only with superstructure compounds until the given value of n is reached.
        for sim_ind in sim_indices:
            if len(retrieved_candidates) == top_n:
                break
            if mol_search_pool[sim_ind[0]].HasSubstructMatch(synthon_mol):
                retrieved_candidates.append((id_search_pool[sim_ind[0]], sim_ind[1]))

        # If there are not enough superstructure candidates, fill the rest of the list with similar compounds.
        if len(retrieved_candidates) < top_n:
            for sim_ind in sim_indices:
                if len(retrieved_candidates) == top_n:
                    break
                if not mol_search_pool[sim_ind[0]].HasSubstructMatch(synthon_mol):
                    retrieved_candidates.append((id_search_pool[sim_ind[0]], sim_ind[1]))

        # Return the final result.
        return retrieved_candidates[:top_n]

    # If the keywords is not equal to 'similar' or 'superstructures', raise an exception.
    else:
        raise Exception("Only the 'similar' and 'superstructures' variations are currently implemented.")


def score_reactant_combination(candidate_combination, scoring_fcn):
    """ Generates a score for a combination of reactant candidates according to the criteria. """

    # Extract only the reactant candidate compound ID's.
    reactant_ids = [combo[0] for combo in candidate_combination]

    # Score the reactant candidate combinations according to the specified criteria.
    if scoring_fcn == "similarity":
        combination_score = np.mean([combo[1] for combo in candidate_combination])
    else:
        combination_score = 0.0

    return reactant_ids, combination_score


def generate_candidate_combinations(reactants_candidates, top_n=5, scoring_fcn="similarity"):
    """ Generates combinations of potential reactant candidates. """

    # If only one reactant is suggested, just transform the data into the proper format.
    if len(reactants_candidates) == 1:
        return [([candidate[0]], candidate[1]) for candidate in reactants_candidates[0]][:top_n]

    # If the more than one reactant is suggested, generate all possible combinations.
    else:
        return sorted([score_reactant_combination(candidate_combination, scoring_fcn=scoring_fcn)
                       for candidate_combination in list(itertools.product(*reactants_candidates))],
                      key=lambda k: k[1], reverse=True)[:top_n]


def analyze_retrieved_elements(retrieved_element_dict, element_type="class"):
    """ Prints Top-N analysis of reactants retrieved based on a criteria specified in the input dictionary. """

    for rr_category in retrieved_element_dict.keys():
        if element_type == "position":
            print("\nPrinting the summary for candidates for synthons at position {}...".format(rr_category))
        elif element_type == "size":
            print("\nPrinting the summary for candidates for synthons with number of atoms {}...".format(rr_category))
        else:
            print("\nPrinting the summary for the combination of candidates from class {}...".format(rr_category))

        position_occurrence_ctr = Counter(retrieved_element_dict[rr_category])
        sum_of_entries = sum([x for x in position_occurrence_ctr.values()])
        total_entries = 0

        print("Total number of entries in this category: {}".format(sum(position_occurrence_ctr.values())))

        for poc in sorted(position_occurrence_ctr.keys()):
            if poc == -1:
                print("Not Found: {} ({}%)".format(position_occurrence_ctr[poc],
                                                   round((position_occurrence_ctr[poc] / sum_of_entries) * 100, 2)))
            else:
                total_entries += position_occurrence_ctr[poc]
                print("Top-{}: {} (cumm. {}) ({}%)".format(poc + 1, position_occurrence_ctr[poc], total_entries,
                                                           round((total_entries / sum_of_entries)*100, 2)))


def benchmark_reactant_candidate_retrieval(args):
    """ Tests the accuracy of the reactant retrieval approach based on fingerprint similarity on the full dataset. """

    # Read the needed datasets.
    final_training_dataset = pd.read_pickle(args.dataset_config.output_folder + "final_reaction_dataset.pkl")
    reactant_search_pool = pd.read_pickle(args.dataset_config.output_folder + "unique_reactants_pool.pkl")

    unique_class_groups = {}
    for reaction_class in args.dataset_config.final_classes:
        unique_class_groups.update({reaction_class: [ind for ind, x in enumerate(reactant_search_pool.values.tolist())
                                                     if reaction_class in x[4]]})

    # Generate dictionaries for the evaluation of individual reactants and reactant combinations.
    mol_size_dict = {"num_atoms=[0,5]": [], "num_atoms=(5,10]": [], "num_atoms=(10,25]": [], "num_atoms=(25,50]": [],
                     "num_atoms>50": []}
    mol_pos_dict = {1: [], 2: [], 3: []}
    mol_class_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    fetch_per_position = {0: 1, 1: 50, 2: 50}

    for row_ind, row in tqdm(final_training_dataset.iterrows(), total=len(final_training_dataset.index),
                             desc="Evaluating the reactant retrieval and scoring on the full final dataset"):
        # Sort the synthons per atom count and fetch all of the needed data.
        synthon_mols, synthon_fps, synthon_maps = zip(*sorted(zip(row["products_non_reactive_smals"],
                                                                  row["products_non_reactive_fps"],
                                                                  row["reactants_uq_mol_maps"]),
                                                              key=lambda k: len(k[0].GetAtoms()), reverse=True))
        suggested_combination = []

        # Go through the list of synthons from the product molecule and retrieve the reactant candidates.
        for synthon_ind, synthon_mol in enumerate(synthon_mols):
            reactants_candidates = fetch_similar_compounds(synthon_mol=synthon_mol,
                                                           synthon_fp=synthon_fps[synthon_ind],
                                                           reactant_search_pool=reactant_search_pool,
                                                           recorded_rxn_classes=row["reaction_class"],
                                                           unique_class_groups=unique_class_groups,
                                                           use_class=True,
                                                           cut_off=0.2,
                                                           top_n=50,
                                                           fetch_priority="superstructures")

            suggested_combination.append(reactants_candidates[:fetch_per_position[synthon_ind]])

            # Analyze the Top-N accuracy of the reactant retrieval based on the atom count.
            positions = [c[0] for c in reactants_candidates]
            position = -1
            if synthon_maps[synthon_ind] in positions:
                position = positions.index(synthon_maps[synthon_ind])

            if len(synthon_mol.GetAtoms()) <= 5:
                mol_size_dict["num_atoms=[0,5]"].append(position)
            elif len(synthon_mol.GetAtoms()) <= 10:
                mol_size_dict["num_atoms=(5,10]"].append(position)
            elif 10 < len(synthon_mol.GetAtoms()) <= 25:
                mol_size_dict["num_atoms=(10,25]"].append(position)
            elif 25 < len(synthon_mol.GetAtoms()) <= 50:
                mol_size_dict["num_atoms=(25,50]"].append(position)
            else:
                mol_size_dict["num_atoms>50"].append(position)

            mol_pos_dict[synthon_ind+1].append(position)

        # Find out the actual combination position.
        combination_positions = [c[0] for c in generate_candidate_combinations(suggested_combination, top_n=50)]
        combination_position = -1

        for c_ind, combination in enumerate(combination_positions):
            if set(combination) == set(synthon_maps):
                combination_position = c_ind
                break

        mol_class_dict[row["reaction_class"]].append(combination_position)

    # Analyze the retrieved reactants based on position in the reaction.
    analyze_retrieved_elements(mol_pos_dict, element_type="position")

    # Analyze the retrieved reactants based on the atom count of the target molecules.
    analyze_retrieved_elements(mol_size_dict, element_type="size")

    # Analyze the retrieved reactant combinations based on the reaction class.
    analyze_retrieved_elements(mol_class_dict, element_type="class")


def complete_and_score_suggestions(args):
    """ TBD. """

    # Use the model to predict the labels for each of the
    test_set_labels = apply_model(args)

    final_test_set = pd.read_pickle(args.evaluation_config.final_evaluation_dataset)
    final_test_set["predicted_class"] = list(test_set_labels)

    reactant_search_pool = pd.read_pickle(args.dataset_config.output_folder + "unique_reactants_pool.pkl")

    combo_position = []

    unique_class_groups = {}
    for reaction_class in args.dataset_config.final_classes:
        unique_class_groups.update({reaction_class: [ind for ind, x in enumerate(reactant_search_pool.values.tolist())
                                                     if reaction_class in x[4]]})

    fetch_per_position = {0: 1, 1: 50, 2: 50}

    for row_ind, row in tqdm(final_test_set.iterrows(), total=len(final_test_set.index),
                             desc="Retrieving and scoring reactant candidate combinations"):
        np.nonzero(row["reaction_class"])

        # Sort the synthons per atom count and fetch all of the needed data.
        synthon_mols, synthon_fps, synthon_maps = zip(*sorted(zip(row["non_reactive_smals"],
                                                                  row["non_reactive_fps"],
                                                                  row["reactants_uq_mol_maps"]),
                                                              key=lambda k: len(k[0].GetAtoms()), reverse=True))
        suggested_combination = []

        # Go through the list of synthons from the product molecule and retrieve the reactant candidates.
        for synthon_ind, synthon_mol in enumerate(synthon_mols):
            reactants_candidates = fetch_similar_compounds(synthon_mol=synthon_mol,
                                                           synthon_fp=synthon_fps[synthon_ind],
                                                           reactant_search_pool=reactant_search_pool,
                                                           recorded_rxn_classes=row["reaction_class"],
                                                           unique_class_groups=unique_class_groups,
                                                           use_class=True,
                                                           cut_off=0.2,
                                                           top_n=50,
                                                           fetch_priority="superstructures")

            suggested_combination.append(reactants_candidates[:fetch_per_position[synthon_ind]])

        # Find out the actual combination position.
        combination_positions = [c[0] for c in generate_candidate_combinations(suggested_combination, top_n=50)]
        combination_position = -1

        for c_ind, combination in enumerate(combination_positions):
            if set(combination) == set(synthon_maps):
                combination_position = c_ind
                break

        combo_position.append(combination_position)

    final_test_set["final_combo"] = combo_position

    print("Found: ")
    print(len(final_test_set[final_test_set["final_combo"] != -1]))
    print("Found at 1: ")
    print(len(final_test_set[final_test_set["final_combo"] == 1]))
    print("Found at 3: ")
    print(len(final_test_set[final_test_set["final_combo"] <= 3]))
    print("Found at 5: ")
    print(len(final_test_set[final_test_set["final_combo"] <= 5]))
    print("Found at 10: ")
    print(len(final_test_set[final_test_set["final_combo"] <= 10]))
    print("Found at 20: ")
    print(len(final_test_set[final_test_set["final_combo"] <= 20]))
    print("Found at 30: ")
    print(len(final_test_set[final_test_set["final_combo"] <= 30]))
    print("Found at 50: ")
    print(len(final_test_set[final_test_set["final_combo"] <= 50]))
    print("Not Found: ")
    print(len(final_test_set[final_test_set["final_combo"] == -1]))
