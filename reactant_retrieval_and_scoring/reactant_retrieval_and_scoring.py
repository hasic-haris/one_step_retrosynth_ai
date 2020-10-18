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


def complete_synthons(synthon_mol, synthon_fp, reactant_search_pool, reactant_ids_per_class=None, cut_off=0.1, top_n=10,
                      fetch_priority="similar", similarity_metric="tanimoto", a=0.5, b=1.0):
    """ Returns the 'top_n' similar compounds from a specific 'search_pool' for a given pattern compound 'synthon_mol'.
        The given pattern compound should be a SMARTS generated RDKit Mol object in order to enable superstructure, but
        if it is not, the method will return structurally similar compounds anyway. """

    # Depending on the available reaction class information, narrow down the search pool.
    if reactant_ids_per_class is not None:
        reactant_search_pool = reactant_search_pool[reactant_search_pool.index.isin(reactant_ids_per_class)]

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

        # If there are no superstructure candidates, fill the rest of the list with similar compounds.
        if len(retrieved_candidates) == 0:
            for sim_ind in sim_indices:
                if len(retrieved_candidates) == top_n:
                    break
                if not mol_search_pool[sim_ind[0]].HasSubstructMatch(synthon_mol):
                    retrieved_candidates.append((id_search_pool[sim_ind[0]], sim_ind[1]))

        # Return the final result.
        return retrieved_candidates

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

"""
def generate_candidate_combinations(reactants_candidates, top_n=50, scoring_fcn="similarity"):

    # If only one reactant is suggested, just transform the data into the proper format.
    if len(reactants_candidates) == 1:
        return [([candidate[0]], candidate[1]) for candidate in reactants_candidates[0]][:top_n]

    # If the more than one reactant is suggested, generate all possible combinations.
    else:
        return sorted([score_reactant_combination(candidate_combination, scoring_fcn=scoring_fcn)
                       for candidate_combination in list(itertools.product(*reactants_candidates))],
                      key=lambda k: k[1], reverse=True)[:top_n]
"""

def generate_candidate_combinations(reactants_candidates, top_n=50, scoring_fcn="similarity"):
    """ Generates combinations of potential reactant candidates. """

    # If only one reactant is suggested, just transform the data into the proper format.
    if len(reactants_candidates) == 1:
        return [{candidate[0]} for candidate in reactants_candidates[0]][:top_n]

    # If the more than one reactant is suggested, generate all possible combinations.
    else:
        full_scoring_list = sorted([score_reactant_combination(candidate_combination, scoring_fcn=scoring_fcn)
                                    for candidate_combination in list(itertools.product(*reactants_candidates))],
                                   key=lambda k: k[1], reverse=True)

        scores = dict()

        for scored_combination in full_scoring_list:
            if scored_combination[1] not in scores:
                scores[scored_combination[1]] = [set(scored_combination[0])]
            else:
                scores[scored_combination[1]].append(set(scored_combination[0]))

        return [scores[k] for k in sorted(scores.keys(), reverse=True)][:top_n]



def analyze_retrieved_reactants(retrieved_reactant_categories, analysis_type="general"):
    """ Prints Top-N analysis of reactants retrieved based on a criteria specified in the input dictionary. """

    for rr_category in retrieved_reactant_categories.keys():
        if analysis_type == "general":
            print("\nPrinting the general analysis summary for the retrieved reactants.")
        elif analysis_type == "class":
            print("\nPrinting the analysis summary for the retrieved reactants from class {}...".format(rr_category))
        elif analysis_type == "position":
            print("\nPrinting the analysis summary for retrieved reactants at position {}...".format(rr_category))
        else:
            print("\nPrinting the analysis summary for retrieved reactants with {} atoms...".format(rr_category))

        position_occurrence_ctr = Counter(retrieved_reactant_categories[rr_category])
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

    # Read the dataset containing all of the chemical reactions.
    final_reaction_dataset = pd.read_pickle(args.dataset_config.output_folder + "final_reaction_dataset.pkl")

    # Read the dataset containing the unique reactant candidate molecules.
    reactant_search_pool = pd.read_pickle(args.dataset_config.output_folder + "unique_reactants_pool.pkl")

    # Create bins of reactant ID's which participated in a reaction class.
    r_ids_per_class = {}
    for reaction_class in args.dataset_config.final_classes:
        if reaction_class > 0:
            r_ids_per_class.update({reaction_class: [ind for ind, x in enumerate(reactant_search_pool.values.tolist())
                                                     if reaction_class in x[4]]})

    # Define how many molecules per position are being fetched.
    fetch_per_position = {0: 5, 1: 20, 2: 20}

    # Generate dictionaries for the evaluation of individual reactants and reactant combinations.
    mol_dict = {1: []}
    mol_class_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    mol_pos_dict = {1: [], 2: [], 3: []}
    mol_size_dict = {"num_atoms=[0,5]": [], "num_atoms=(5,10]": [], "num_atoms=(10,25]": [], "num_atoms=(25,50]": [],
                     "num_atoms>50": []}

    for row_ind, row in tqdm(final_reaction_dataset.iterrows(), total=len(final_reaction_dataset.index), ascii=True,
                             desc="Evaluating the reactant retrieval and scoring on the full final dataset"):
        # Sort the synthons per atom count and fetch all of the needed data.
        synthon_mols, synthon_fps, synthon_maps = zip(*sorted(zip(row["products_non_reactive_smals"],
                                                                  row["products_non_reactive_fps"],
                                                                  row["reactants_uq_mol_maps"]),
                                                              key=lambda k: len(k[0].GetAtoms()), reverse=True))

        # Go through the list of synthons from the product molecule and retrieve the reactant candidates.
        suggested_combination = []

        for synthon_ind, synthon_mol in enumerate(synthon_mols):
            reactant_candidates = complete_synthons(synthon_mol=synthon_mol,
                                                    synthon_fp=synthon_fps[synthon_ind],
                                                    reactant_search_pool=reactant_search_pool,
                                                    reactant_ids_per_class=r_ids_per_class[row["reaction_class"]],
                                                    top_n=20,
                                                    fetch_priority="superstructures")

            suggested_combination.append(reactant_candidates[:fetch_per_position[synthon_ind]])

            # Analyze the Top-N accuracy of the reactant retrieval based on the atom count.
            positions = [rc[0] for rc in reactant_candidates]
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
        #combination_positions = [c[0] for c in generate_candidate_combinations(suggested_combination, top_n=50)]
        combination_positions = generate_candidate_combinations(suggested_combination, top_n=50)
        combination_position = -1

        for c_ind, combination in enumerate(combination_positions):
            #if set(combination) == set(synthon_maps):
            if set(synthon_maps) in combination:
                combination_position = c_ind
                break

        mol_class_dict[row["reaction_class"]].append(combination_position)
        mol_dict[1].append(combination_position)

    # Analyze the retrieved reactants.
    analyze_retrieved_reactants(mol_dict, analysis_type="general")

    # Analyze the retrieved reactant combinations based on the reaction class.
    analyze_retrieved_reactants(mol_class_dict, analysis_type="class")

    # Analyze the retrieved reactants based on position in the reaction.
    analyze_retrieved_reactants(mol_pos_dict, analysis_type="position")

    # Analyze the retrieved reactants based on the atom count of the target molecules.
    analyze_retrieved_reactants(mol_size_dict, analysis_type="size")


def complete_and_score_suggestions(args):
    """ TBD. """

    # Use the model to predict the labels for each of the extracted substructures.
    test_set_labels = apply_model(args)

    # Read the dataset containing all of the chemical reactions for testing.
    final_test_set = pd.read_pickle(args.evaluation_config.final_evaluation_dataset)

    final_test_set["predicted_classes"] = [np.nonzero(lab)[0] for lab in list(test_set_labels)]
    final_test_set["best_class"] = [0 if len(np.nonzero(lab)[0]) == 1 and np.nonzero(lab)[0][0] == 0
                                    else np.argmax(lab[1:])+1 for lab in list(test_set_labels)]

    final_test_set = final_test_set[final_test_set["in_core"]]

    # Read the dataset containing the unique reactant candidate molecules.
    reactant_search_pool = pd.read_pickle(args.dataset_config.output_folder + "unique_reactants_pool.pkl")

    # Create bins of reactant ID's which participated in a reaction class.
    # r_ids_per_class = {0: None}
    r_ids_per_class = {}
    for reaction_class in args.dataset_config.final_classes:
        if reaction_class > 0:
            r_ids_per_class.update({reaction_class: [ind for ind, x in enumerate(reactant_search_pool.values.tolist())
                                                     if reaction_class in x[4]]})

    # Define how many molecules per position are being fetched.
    fetch_per_position = {0: 5, 1: 20, 2: 20}

    # Generate dictionaries for the evaluation of individual reactants and reactant combinations.
    mol_dict = {1: []}
    mol_class_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    successful_pred, unsuccessful_pred = 0, 0
    final_combination_positions = []

    for row_ind, row in tqdm(final_test_set.iterrows(), total=len(final_test_set.index), ascii=True,
                             desc="Retrieving and scoring reactant candidate combinations"):

        # Count the amount of successful and unsuccessful predictions.
        if row["reaction_class"] in row["predicted_classes"]:
            successful_pred += 1
        else:
            unsuccessful_pred += 1

        # Sort the synthons per atom count and fetch all of the needed data.
        synthon_mols, synthon_fps, synthon_maps = zip(*sorted(zip(row["non_reactive_smals"],
                                                                  row["non_reactive_fps"],
                                                                  row["reactants_uq_mol_maps"]),
                                                              key=lambda k: len(k[0].GetAtoms()), reverse=True))

        # Go through the list of synthons from the product molecule and retrieve the reactant candidates.
        suggested_combination = []

        # Go through the list of synthons from the product molecule and retrieve the reactant candidates.
        for synthon_ind, synthon_mol in enumerate(synthon_mols):
            reactant_candidates = complete_synthons(synthon_mol=synthon_mol,
                                                    synthon_fp=synthon_fps[synthon_ind],
                                                    reactant_search_pool=reactant_search_pool,
                                                    # reactant_ids_per_class=None,
                                                    reactant_ids_per_class=list(itertools.chain.from_iterable(
                                                        [r_ids_per_class[k] for k in r_ids_per_class.keys()
                                                         if k in row["predicted_classes"]])),
                                                    top_n=20,
                                                    fetch_priority="superstructures")

            suggested_combination.append(reactant_candidates[:fetch_per_position[synthon_ind]])

        # Find out the actual combination position.
        #combination_positions = [c[0] for c in generate_candidate_combinations(suggested_combination, top_n=50)]
        combination_positions = generate_candidate_combinations(suggested_combination, top_n=50)
        combination_position = -1

        for c_ind, combination in enumerate(combination_positions):
            #if set(combination) == set(synthon_maps):
            if set(synthon_maps) in combination:
                combination_position = c_ind
                break

        mol_class_dict[row["reaction_class"]].append(combination_position)
        mol_dict[1].append(combination_position)
        final_combination_positions.append(combination_positions)

    # Analyze the retrieved reactants.
    # analyze_retrieved_reactants(mol_dict, analysis_type="general")

    # Analyze the retrieved reactant combinations based on the reaction class.
    # analyze_retrieved_reactants(mol_class_dict, analysis_type="class")

    print("Correctly classified: {}% ({}/{})".format(round(successful_pred * 100 / (successful_pred +
                                                                                    unsuccessful_pred), 2),
                                                     successful_pred, (successful_pred + unsuccessful_pred)))
    print("Incorrectly classified: {}% ({}/{})".format(round(unsuccessful_pred * 100 / (successful_pred +
                                                                                        unsuccessful_pred), 2),
                                                       unsuccessful_pred, (successful_pred + unsuccessful_pred)))

    final_test_set["combination_position"] = final_combination_positions
    final_test_set.to_pickle(args.dataset_config.output_folder + "final_test.pkl")

    """
    print("Found: ")
    print(len(final_test_set[final_test_set["combination_position"] != -1]))
    print("Found at 1: ")
    print(len(final_test_set[(final_test_set["combination_position"] != -1) & (final_test_set["combination_position"] == 1)]))
    print("Found at 3: ")
    print(len(final_test_set[(final_test_set["combination_position"] != -1) & (final_test_set["combination_position"] <= 3)]))
    print("Found at 5: ")
    print(len(final_test_set[(final_test_set["combination_position"] != -1) & (final_test_set["combination_position"] <= 5)]))
    print("Found at 10: ")
    print(len(final_test_set[(final_test_set["combination_position"] != -1) & (final_test_set["combination_position"] <= 10)]))
    print("Found at 20: ")
    print(len(final_test_set[(final_test_set["combination_position"] != -1) & (final_test_set["combination_position"] <= 20)]))
    print("Found at 30: ")
    print(len(final_test_set[(final_test_set["combination_position"] != -1) & (final_test_set["combination_position"] <= 30)]))
    print("Found at 50: ")
    print(len(final_test_set[(final_test_set["combination_position"] != -1) & (final_test_set["combination_position"] <= 50)]))
    print("Not Found: ")
    print(len(final_test_set[final_test_set["combination_position"] == -1]))
    """
