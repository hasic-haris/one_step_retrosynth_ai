"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  January 11th, 2020
Description: This file contains necessary functions for constructing and similarity checking of molecular fingerprints.
"""

import numpy as np
from rdkit.Chem import AllChem, DataStructs
from rdkit.DataStructs import cDataStructs

from chemistry_methods.molecules import get_atom_environment


def np_array_to_binary_vector(np_arr):
    """ Converts a NumPy array to the RDKit ExplicitBitVector type. """

    binary_vector = DataStructs.ExplicitBitVect(len(np_arr))
    binary_vector.SetBitsFromList(np.where(np_arr)[0].tolist())

    return binary_vector


def construct_ecfp(mol, radius, bits, from_atoms=None, output_type="bit_vector", as_type="np_int"):
    """ Returns the Extended Connectivity Fingerprint (ECFP) representation of the whole molecule (default) or just from
        specific atoms, if it is specified through the 'from_atoms' parameter. The type of the whole fingerprint and the
        individual bits can be adjusted with the parameters 'output_type' and 'as_type', respectively. """

    # Check if the input molecule is given in the SMILES or in the RDKit Mol format.
    if isinstance(mol, str):
        # Generate the RDKit Mol object from the input SMILES string.
        mol = AllChem.MolFromSmiles(mol)
        # Sanitize the molecule.
        AllChem.SanitizeMol(mol)

    # Generate the ECFP based on the specified input parameters.
    if from_atoms is not None:
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits, fromAtoms=from_atoms)
    else:
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits)

    # If it is specified by the output type parameter, convert the result to an NumPy array.
    if output_type == "np_array":
        result_ecfp = np.array([])
        cDataStructs.ConvertToNumpyArray(ecfp, result_ecfp)
        ecfp = result_ecfp.astype(np.int) if as_type == "np_int" else result_ecfp.astype(np.float)

    # Return the constructed fingerprint.
    return ecfp


def construct_hsfp(mol, radius, bits, from_atoms, neighbourhood_ext=None):
    """ Returns the Hot Spot Fingerprint (HSFP) representation of the whole molecule (default) or just from specific
        atoms, if it is specified through the 'from_atoms' parameter, as a NumPy array. If the parameter 'from_atoms'
        includes all of the atoms of the molecule, the constructed fingerprint is equivalent to the ECFP representation
        of the whole molecule. The 'neighbourhood_ext' parameter controls how many of the neighbourhood atoms are added
        to the focus atoms specified by the parameter 'from_atoms'. """

    # Check if the input molecule is given in SMILES or in the RDKit Mol format.
    if isinstance(mol, str):
        # Generate the RDKit Mol object from the input SMILES string.
        mol = AllChem.MolFromSmiles(mol)
        # Sanitize the molecule.
        AllChem.SanitizeMol(mol)

    # Fetch the respective distance matrix.
    distance_matrix = AllChem.GetDistanceMatrix(mol)

    # Set the weight factor for the generation of the HSFP.
    weight_factor = np.max(distance_matrix) if neighbourhood_ext is None else neighbourhood_ext

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


def tanimoto_similarity(ecfp1, ecfp2):
    """ Returns the Tanimoto similarity value between two fingerprints. """

    return DataStructs.TanimotoSimilarity(ecfp1, ecfp2)


def bulk_tanimoto_similarity(ecfp, ecfp_pool):
    """ Returns the Tanimoto similarity values between a single fingerprint and a pool of fingerprints. """

    return DataStructs.BulkTanimotoSimilarity(ecfp, ecfp_pool)


def dice_similarity(ecfp1, ecfp2):
    """ Returns the Dice similarity value between two fingerprints. """

    return DataStructs.DiceSimilarity(ecfp1, ecfp2)


def bulk_dice_similarity(ecfp, ecfp_pool):
    """ Returns the Dice similarity values between a single fingerprint and a pool of fingerprints. """

    return DataStructs.BulkDiceSimilarity(ecfp, ecfp_pool)


def tversky_similarity(ecfp1, ecfp2, a=0.5, b=1.0):
    """ Returns the Tversky similarity value between two fingerprints using the parameter values a and b. """

    return DataStructs.TverskySimilarity(ecfp1, ecfp2, a, b)


def bulk_tversky_similarity(ecfp, ecfp_pool, a=0.5, b=1.0):
    """ Returns the Tversky similarity values between a single fingerprint and a pool of fingerprints using the
        parameter values a and b. """

    return DataStructs.BulkTverskySimilarity(ecfp, ecfp_pool, a, b)
