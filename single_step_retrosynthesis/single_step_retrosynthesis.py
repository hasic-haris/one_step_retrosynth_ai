"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  March 31st, 2020
Description: This file contains the necessary function for the single-step retrosynthesis analysis of a target molecule.
"""

import pandas as pd
from collections import Counter


def bulk_analyze_disconnection_suggestions(args):
    """ Returns potential disconnections in the target molecule including the chemical reaction class, real reactant
    molecules and the probability of the reaction. """

    final_data = pd.read_pickle(args.dataset_config.output_folder + "final_training_dataset.pkl")
    print(final_data.head(5))
    print(final_data.columns)

    ctr = []
    for _, row in final_data.iterrows():
        print(row["reactants_uq_mol_maps"])
        ctr.append(len(row["reactants_uq_mol_maps"]))

    print(Counter(ctr))

    exit(0)

    top_n, length = [], []
    patent_ids = list(set(final_data["patent_id"].values))

    for pid in patent_ids:
        min_pos = 666
        for _, row in final_data[final_data["patent_id"] == pid].iterrows():
            if len(row["combination_position"]) > 0:
                for cc_ind, cc in enumerate(row["combination_position"]):
                    if type(cc) == set and set(row["reactants_uq_mol_maps"]) == cc:
                        if cc_ind + 1 < min_pos:
                            min_pos = cc_ind + 1
                            length.append(1)
                    elif type(cc) == list and set(row["reactants_uq_mol_maps"]) in cc:
                        if cc_ind + 1 < min_pos:
                            min_pos = cc_ind + 1
                            length.append(len(cc))

                if min_pos == 1:
                    break

        top_n.append(min_pos)

    top_n = dict(Counter(top_n))
    agg_top_n = 0

    for key in sorted(top_n.keys()):
        agg_top_n += top_n[key]
        if key in [1, 3, 5, 10, 20, 30, 50]:
            print("Top-{}: {} ({:2.2f}%)".format(key, agg_top_n, (agg_top_n / sum(top_n.values())) * 100))
