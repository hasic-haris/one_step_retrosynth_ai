"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 21st, 2020
Description: This file contains necessary functions for the retrieval and scoring of potential reactant candidates.
"""

import itertools
import pandas as pd
import numpy as np
from collections import Counter

from chemistry_methods.fingerprints import bulk_dice_similarity, bulk_tanimoto_similarity, bulk_tversky_similarity


def fetch_n_similar_compounds(synthon_ind, synthon_mols, synthon_fps, search_pool, reaction_class, uq_class_ids,
                              use_class=False, use_size=False, cut_off=0.1, top_n=None, fetch_priority="similar",
                              similarity_metric="tanimoto", a=0.5, b=1.0):
    """ Returns the 'top_n' similar compounds and for a given pattern compound ''. The given pattern compound should be
        a SMARTS generated RDKit Mol object in order to enable superstructure searching. """

    # Extract the RDKit Mol object and fingerprint of the pattern molecule.
    pattern_mol = synthon_mols[synthon_ind]
    pattern_mol_fp = synthon_fps[synthon_ind]

    # Depending on the available reaction class information, construct the search pools.
    if use_class:
        search_pool = search_pool[search_pool.index.isin(uq_class_ids[reaction_class])]

    # Further filter the pools according to the size of the pattern molecules.
    if len(pattern_mol.GetAtoms()) < 5 and use_size:
        id_pool = search_pool[search_pool["num_atoms"] <= 10]["id"].values.tolist()
        mol_pool = search_pool[search_pool["num_atoms"] <= 10]["mol_object"].values.tolist()
        fp_pool = search_pool[search_pool["num_atoms"] <= 10]["ecfp_1024"].values.tolist()
    else:
        id_pool = search_pool["id"].values.tolist()
        mol_pool = search_pool["mol_object"].values.tolist()
        fp_pool = search_pool["ecfp_1024"].values.tolist()

    # Define and apply the similarity metric.
    if similarity_metric == "dice":
        similarity_values = bulk_dice_similarity(pattern_mol_fp, fp_pool)
    elif similarity_metric == "tanimoto":
        similarity_values = bulk_tanimoto_similarity(pattern_mol_fp, fp_pool)
    elif similarity_metric == "tversky":
        similarity_values = bulk_tversky_similarity(pattern_mol_fp, fp_pool, a=a, b=b)
    else:
        raise Exception("Only the 'dice', 'tanimoto' and 'tversky' similarity metrics are currently implemented.")

    # Sort the values and get the indices of the sorted array.
    sim_indices = sorted([(sim_ind, sim_val) for sim_ind, sim_val in enumerate(similarity_values)
                          if sim_val >= cut_off], key=lambda k: k[1], reverse=True)

    # If the cut-off values left the list of indices empty, it implies that the fragment is too small to find any
    # significant reactant match. Because of that, go through the list of frequent small molecule reactants from each
    # class and find the best fitting option.
    if len(sim_indices) == 0:
        sim_indices = sorted([(sim_ind, sim_val) for sim_ind, sim_val in enumerate(similarity_values)],
                             key=lambda k: k[1], reverse=True)

    # Return the Top-N candidates based on the applied metric, regardless of substructure matching.
    if fetch_priority == "similar":
        return [(id_pool[sim_ind[0]], sim_ind[1]) for sim_ind in sim_indices][:top_n]

    # Return the Top-N candidates based on the applied metric, only if they're superstructures to the pattern molecule.
    elif fetch_priority == "superstructures":
        retrieved_candidates = []

        # Fill the list of candidates only with superstructure compounds until the given value of n is reached.
        for sim_ind in sim_indices:
            if len(retrieved_candidates) == top_n:
                break
            if mol_pool[sim_ind[0]].HasSubstructMatch(pattern_mol):
                retrieved_candidates.append((id_pool[sim_ind[0]], sim_ind[1]))

        # If there are not enough superstructure candidates, fill the rest of the list with similar compounds.
        if len(retrieved_candidates) < top_n:
            for sim_ind in sim_indices:
                if len(retrieved_candidates) == top_n:
                    break
                if not mol_pool[sim_ind[0]].HasSubstructMatch(pattern_mol):
                    retrieved_candidates.append((id_pool[sim_ind[0]], sim_ind[1]))

        # Return the final result.
        return retrieved_candidates[:top_n]

    # If the keywords is not equal to 'similar' or 'superstructures', raise an exception.
    else:
        raise Exception("Only the 'similar' and 'superstructures' variations are currently implemented.")


# Done: 100%
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


# Done: 100%
def score_reactant_combination(candidate_combination, scoring_fcn):
    """ Generate a score for a combination of reactant candidates according to the criteria. """

    # Isolate the candidate compound ID's.
    reactant_ids = [combo[0] for combo in candidate_combination]

    # Score the combinations according to a given criteria.
    if scoring_fcn == "similarity":
        combination_score = np.mean([combo[1] for combo in candidate_combination])
    else:
        combination_score = 0.0

    # Finally, return the pair (list_of_candidate_ids, combination_score)
    return reactant_ids, combination_score


# Done: 100%
def generate_candidate_combinations(reactants_candidates, top_n=5, scoring_fcn="similarity"):
    """ Generate combinations of potential reactant candidates. """

    # If the algorithm suggested one reactant, just transform the data into the proper format.
    if len(reactants_candidates) == 1:
        return [([candidate[0]], candidate[1]) for candidate in reactants_candidates[0]][:top_n]

    # If the algorithm suggested more than one reactant, generate all possible combinations.
    else:
        return sorted([score_reactant_combination(candidate_combination, scoring_fcn=scoring_fcn)
                       for candidate_combination in list(itertools.product(*reactants_candidates))],
                      key=lambda k: k[1], reverse=True)[:top_n]


# Done: 33%
def benchmark_reactant_candidate_retrieval(**kwargs):
    """ Tests the accuracy of the reactant retrieval approach based on fingerprint similarity on the full dataset. """

    # Read the generated full reaction dataset.
    full_dataset = pd.read_pickle(kwargs["output_folder"] + "final_reaction_dataset.pkl")
    search_pool = pd.read_pickle(kwargs["output_folder"] + "unique_reactants_pool.pkl")
    uq_class_groups = {}
    for reaction_class in range(1, 11):
        uq_class_groups.update({reaction_class: [ind for ind, x in enumerate(search_pool.values.tolist())
                                                 if reaction_class in x[5]]})

    # Generate dictionaries for the evaluation of individual reactants and reactant combinations.
    mol_size_dict = {"num_atoms=[0,5]": [], "num_atoms=(5,10]": [], "num_atoms=(10,25]": [], "num_atoms=(25,50]": [],
                     "num_atoms>50": []}
    mol_pos_dict = {1: [], 2: [], 3: []}
    mol_class_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    fetch_per_position = {0: 1, 1: 50, 2: 50}

    row_ctr = 0
    for row_ind, row in full_dataset.iterrows():
        # --------------------------------------------------------------------------------------------------------------
        # Step 1: Print out the information about the current processing status.
        # --------------------------------------------------------------------------------------------------------------
        # if row_ctr > 100:
        #    break

        print("Currently processing row {}...".format(row_ctr))
        row_ctr += 1

        # --------------------------------------------------------------------------------------------------------------
        # Step 2: Retrieve all reactant candidates.
        # --------------------------------------------------------------------------------------------------------------

        # Sort the synthons per atom count and fetch all needed data.
        synthon_smals, synthon_fps, synthon_maps = zip(*sorted(zip(row["products_non_reactive_smals"],
                                                                   row["products_non_reactive_fps"],
                                                                   row["reactants_uq_mol_maps"]),
                                                               key=lambda k: len(k[0].GetAtoms()), reverse=True))

        # print("Correct combination: {}".format(synthon_maps))
        suggested_combination = []

        # Go through the list of synthons from the product molecule and retrieve the reactant candidates.
        for synthon_ind, synthon in enumerate(synthon_smals):

            reactants_candidates = fetch_n_similar_compounds(synthon_ind, synthon_smals, synthon_fps, search_pool,
                                                             row["reaction_class"], uq_class_groups, use_class=True,
                                                             use_size=True, cut_off=0.2, top_n=50,
                                                             fetch_priority="superstructures")

            suggested_combination.append(reactants_candidates[:fetch_per_position[synthon_ind]])

            # Analysis.
            positions = [c[0] for c in reactants_candidates]
            position = -1
            if synthon_maps[synthon_ind] in positions:
                position = positions.index(synthon_maps[synthon_ind])

            if len(synthon.GetAtoms()) <= 5:
                mol_size_dict["num_atoms=[0,5]"].append(position)
            elif len(synthon.GetAtoms()) <= 10:
                mol_size_dict["num_atoms=(5,10]"].append(position)
            elif 10 < len(synthon.GetAtoms()) <= 25:
                mol_size_dict["num_atoms=(10,25]"].append(position)
            elif 25 < len(synthon.GetAtoms()) <= 50:
                mol_size_dict["num_atoms=(25,50]"].append(position)
            else:
                mol_size_dict["num_atoms>50"].append(position)

            mol_pos_dict[synthon_ind+1].append(position)

        # --------------------------------------------------------------------------------------------------------------
        # Step 3: Generate combinations out of the retrieved reactant candidates.
        # --------------------------------------------------------------------------------------------------------------

        # Analysis.
        combination_positions = [c[0] for c in generate_candidate_combinations(suggested_combination, top_n=50)]
        combination_position = -1

        for cind, combination in enumerate(combination_positions):
            if set(combination) == set(synthon_maps):
                combination_position = cind
                break

        mol_class_dict[row["reaction_class"]].append(combination_position)

    # Analyze the retrieved reactants based on position in the reaction.
    analyze_retrieved_elements(mol_pos_dict, element_type="position")
    # Analyze the retrieved reactants based on the atom count of the target molecules.
    analyze_retrieved_elements(mol_size_dict, element_type="size")
    # Analyze the retrieved reactant combinations based on the reaction class.
    analyze_retrieved_elements(mol_class_dict, element_type="class")


from chemistry_methods.fingerprints import construct_ecfp

def get_combinations_for_single_mol(synthon_mols, reaction_class, **kwargs):
    """ Tests the accuracy of the reactant retrieval approach based on fingerprint similarity on the full dataset. """

    # Read the generated full reaction dataset.
    # full_dataset = pd.read_pickle(kwargs["output_folder"] + "final_reaction_dataset.pkl")
    search_pool = pd.read_pickle(kwargs["output_folder"] + "unique_reactants_pool.pkl")

    uq_class_groups = {}
    for rc in range(1, 11):
        uq_class_groups.update({rc: [ind for ind, x in enumerate(search_pool.values.tolist()) if rc in x[5]]})
    fetch_per_position = {0: 3, 1: 50, 2: 50}

    synthon_mols = sorted(synthon_mols, key=lambda k: len(k.GetAtoms()), reverse=True)
    synthon_fps = [construct_ecfp(mol, 2, 1024) for mol in synthon_mols]

    suggested_combination = []

    # Go through the list of synthons from the product molecule and retrieve the reactant candidates.
    for synthon_ind, synthon in enumerate(synthon_mols):
        reactants_candidates = fetch_n_similar_compounds(synthon_ind, synthon_mols, synthon_fps, search_pool,
                                                         reaction_class, uq_class_groups, use_class=False,
                                                         use_size=True, cut_off=0.2, top_n=50,
                                                         fetch_priority="superstructures")

        suggested_combination.append(reactants_candidates[:fetch_per_position[synthon_ind]])

    # Analysis.
    return generate_candidate_combinations(suggested_combination, top_n=50)
