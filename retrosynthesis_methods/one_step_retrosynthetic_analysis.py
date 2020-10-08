"""
Author:      Hasic Haris (Phd Student @ Ishida Lab, Department of Computer Science, Tokyo Institute of Technology)
Created on:  March 31st, 2020
Description: This file contains the main function for the one-step retro-synthetic analysis of a target molecule.
"""
import numpy as np

from rdkit.Chem import AllChem

from chemistry_methods.fingerprints import construct_hsfp
from chemistry_methods.reaction_analysis import extract_info_from_molecule
from model_methods.tf_model_construction import test_model
from model_methods.tf_main import generate_model_hp_config
from retrosynthesis_methods.reactant_retrieval_and_scoring import get_combinations_for_single_mol



# Done: 33%
def analyze_novel_molecule(mol, **kwargs):
    """ Returns potential disconnections in the target molecule including the chemical reaction class, real reactant
    molecules and the probability of the reaction. """

    # Check if the input molecule is given in SMILES or in the RDKit 'Mol' format.
    if isinstance(mol, str):
        # Generate the RDKit 'Mol' object from the input SMILES string.
        mol = AllChem.MolFromSmiles(mol)
        # Sanitize the molecule.
        AllChem.SanitizeMol(mol)

    substructure_fps = np.array([construct_hsfp(mol,
                                                kwargs["best_fp_config"]["radius"],
                                                kwargs["best_fp_config"]["bits"],
                                                from_atoms=[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                                                nghb_size=kwargs["best_fp_config"]["ext"]) for bond in mol.GetBonds()])
    substructure_labels = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in mol.GetBonds()])
    bond_info = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]

    _, predicted_labels, _, _, _, _ = test_model(
        generate_model_hp_config(input_fp_data=kwargs["best_fp_config"]["folder_name"], oversample=True, **kwargs),
        log_folder=kwargs["best_fp_config"]["folder_name"],
        x_test=substructure_fps,
        y_test=substructure_labels)

    for labels_ind, labels in enumerate(np.round(predicted_labels, 3)):
        if len([l_ind for l_ind, l in enumerate(labels) if l > 0.]) == 1 and \
                [l_ind for l_ind, l in enumerate(labels) if l > 0.][0] == 0:
            continue

        print("{} - {}".format(labels_ind, [l_ind for l_ind, l in enumerate(labels) if l > 0.]))

        synthons = extract_info_from_molecule(mol, bond_info[labels_ind], role="product")[1]
        reactant_combos = get_combinations_for_single_mol(synthons[3], np.argmax([l_ind for l_ind, l in enumerate(labels) if l > 0.]), **kwargs)

        print(synthons)
        print(reactant_combos)
