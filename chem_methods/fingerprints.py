"""
Author:      Hasic Haris (Phd Student @ Ishida Lab, Department of Computer Science, Tokyo Institute of Technology)
Created on:  January 11th, 2020
Description: This file contains functions for the construction and similarity checking of molecular fingerprints.
"""
import numpy as np

from rdkit.Chem import AllChem, DataStructs
from rdkit.DataStructs import cDataStructs

from chem_methods.molecules import get_atom_environment


# ----------------------------------------------------------------------------------------------------------------------
# Generation of Extended Connectivity Fingerprints and Hot Spot Fingerprints.
# ----------------------------------------------------------------------------------------------------------------------

# Done: 100%
def np_array_to_binary_vector(np_arr):
    """ Converts a NumPy array to the RDKit ExplicitBitVector type. """
    binary_vector = DataStructs.ExplicitBitVect(len(np_arr))
    binary_vector.SetBitsFromList(np.where(np_arr)[0].tolist())

    return binary_vector


# Done: 100 %
def construct_ecfp(mol, radius, bits, from_atoms=None, output_type="bit_vector", as_type="np_int"):
    """ Returns the Extended Connectivity Fingerprint (ECFP) of the whole molecule (default) or just specific atoms. """

    # Check if the input molecule is given in SMILES or in the RDKit 'Mol' format.
    if isinstance(mol, str):
        # Generate the RDKit 'Mol' object from the input SMILES string.
        mol = AllChem.MolFromSmiles(mol)
        # Sanitize the molecule.
        AllChem.SanitizeMol(mol)

    # Generate the ECFP based on the input parameters.
    if from_atoms is not None:
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits, fromAtoms=from_atoms)
    else:
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits)

    # If it is specified by the output type parameter, convert the result to an NumPy array.
    if output_type == "np_array":
        result_ecfp = np.array([])
        cDataStructs.ConvertToNumpyArray(ecfp, result_ecfp)
        ecfp = result_ecfp.astype(np.int) if as_type == "np_int" else result_ecfp.astype(np.float)

    # Return the constructed ECFP object.
    return ecfp


# Done: 100%
def construct_hsfp(mol, radius, bits, from_atoms, nghb_size=-1):
    """ Returns the Hot Spot Fingerprints (HSFP) in reference to a specified focus atom group as a NumPy array. """

    # Check if the input molecule is given in SMILES or in the RDKit 'Mol' format.
    if isinstance(mol, str):
        # Generate the RDKit 'Mol' object from the input SMILES string.
        mol = AllChem.MolFromSmiles(mol)
        # Sanitize the molecule.
        AllChem.SanitizeMol(mol)

    # Fetch the respective distance matrix.
    distance_matrix = AllChem.GetDistanceMatrix(mol)

    # Set the weight factor for the generation of the HSFP.
    weight_factor = np.max(distance_matrix) if nghb_size == -1 else nghb_size

    # Generate the base of the HSFP, which is basically the ECFP of the core atoms.
    core_fp = construct_ecfp(mol, radius=radius, bits=bits, from_atoms=from_atoms, output_type="np_array")
    hsfp = np.array(core_fp)

    # Iterate through and add other layers to the HSFP.
    for i in range(0, int(weight_factor)):
        # Generate a fingerprint for the distance increment, and calculate the bitwise difference.
        atom_environment = get_atom_environment(from_atoms, mol, degree=i+1)
        env_fp = construct_ecfp(mol, radius=radius, bits=bits, from_atoms=atom_environment, output_type="np_array")
        diff = np.bitwise_xor(core_fp, env_fp)

        # Add the weighted difference vector to the resulting HSFP vector.
        core_fp = core_fp + diff
        hsfp = hsfp + diff * 1/(i+2)

    # Return the resulting HSFP rounded to three decimals.
    return np.round(hsfp, 3)


# ----------------------------------------------------------------------------------------------------------------------
# Calculating fingerprint similarity scores.
# ----------------------------------------------------------------------------------------------------------------------

# Done: 100%
def tanimoto_similarity(ecfp1, ecfp2):
    """ Returns the Tanimoto similarity value between two fingerprints. """
    return DataStructs.TanimotoSimilarity(ecfp1, ecfp2)


# Done: 100%
def bulk_tanimoto_similarity(ecfp, ecfp_pool):
    """ Returns the Tanimoto similarity values between a single fingerprint and a pool of fingerprints. """
    return DataStructs.BulkTanimotoSimilarity(ecfp, ecfp_pool)


# Done: 100%
def dice_similarity(ecfp1, ecfp2):
    """ Returns the Dice similarity value between two fingerprints. """
    return DataStructs.DiceSimilarity(ecfp1, ecfp2)


# Done: 100%
def bulk_dice_similarity(ecfp, ecfp_pool):
    """ Returns the Dice similarity values between a single fingerprint and a pool of fingerprints. """
    return DataStructs.BulkDiceSimilarity(ecfp, ecfp_pool)


# Done: 100%
def tversky_similarity(ecfp1, ecfp2, a=0.5, b=1.0):
    """ Returns the Tversky similarity value between two fingerprints using the parameter values a and b. """
    return DataStructs.TverskySimilarity(ecfp1, ecfp2, a, b)


# Done: 100%
def bulk_tversky_similarity(ecfp, ecfp_pool, a=0.5, b=1.0):
    """ Returns the Tversky similarity values between a single fingerprint and a pool of fingerprints using the
    parameter values a and b. """
    return DataStructs.BulkTverskySimilarity(ecfp, ecfp_pool, a, b)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
