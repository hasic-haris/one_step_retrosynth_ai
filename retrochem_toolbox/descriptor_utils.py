"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  January 11th, 2020
Edited on:   January 31st, 2021
"""

import numpy as np

from typing import List, Tuple, Union

from rdkit.Chem.AllChem import Mol
from rdkit.Chem.AllChem import GetDistanceMatrix, GetMorganFingerprintAsBitVect
from rdkit.DataStructs import DiceSimilarity, BulkDiceSimilarity
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from rdkit.DataStructs import TverskySimilarity, BulkTverskySimilarity
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray, ExplicitBitVect

from .compound_utils import CompoundConversionUtils, CompoundStructureUtils


class MolecularFingerprintsUtils:
    """ Description: Group of methods for handling the construction and similarity of molecular fingerprints."""

    @staticmethod
    def np_array_to_binary_vector(np_arr: np.ndarray) -> ExplicitBitVect:
        """ Description: Convert a NumPy array to the RDKit ExplicitBitVector type. """

        binary_vector = ExplicitBitVect(len(np_arr))
        binary_vector.SetBitsFromList(np.where(np_arr)[0].tolist())

        return binary_vector

    @staticmethod
    def construct_ecfp(compound: Union[str, Mol], radius: int, bits: int, from_atoms=None, output_type="bit_vector",
                       as_type="np_int", return_bit_info=False,
                       verbose=False) -> Union[Tuple, ExplicitBitVect, np.ndarray, None]:
        """ Description: Return the Extended Connectivity Fingerprint (ECFP) representation of the whole molecule or
                         just from specific atoms, if it is specified through the 'from_atoms' parameter. The type of
                         the fingerprint and the individual bits can be adjusted with the parameters 'output_type' and
                         'as_type', respectively. """

        if isinstance(compound, str):
            compound = CompoundConversionUtils.string_to_mol(compound)

        try:
            bit_info = {}

            if from_atoms is not None:
                ecfp = GetMorganFingerprintAsBitVect(compound, radius=radius, nBits=bits, fromAtoms=from_atoms,
                                                     bitInfo=bit_info)
            else:
                ecfp = GetMorganFingerprintAsBitVect(compound, radius=radius, nBits=bits, bitInfo=bit_info)

            # If it is specified by the output type parameter, convert the result to an NumPy array.
            if output_type == "np_array":
                result_ecfp = np.array([])
                ConvertToNumpyArray(ecfp, result_ecfp)
                ecfp = result_ecfp.astype(np.int) if as_type == "np_int" else result_ecfp.astype(np.float)

            if return_bit_info:
                return ecfp, bit_info
            else:
                return ecfp

        except Exception as exc_msg:
            if verbose:
                print("Exception occurred during the generation of the ECFP. Detailed message: {}".format(exc_msg))

            return None

    @staticmethod
    def construct_hsfp(compound: Union[str, Mol], radius: int, bits: int, from_atoms, neighbourhood_ext=None,
                       verbose=False) -> Union[np.ndarray, None]:
        """ Description: Return the Hot Spot Fingerprint (HSFP) representation of the whole molecule (default) or just
                         from specific atoms, if it is specified through the 'from_atoms' parameter, as a NumPy array.
                         If the parameter 'from_atoms' includes all of the atoms of the molecule, the constructed
                         fingerprint is equivalent to the ECFP representation of the whole molecule.
                         The 'neighbourhood_ext' parameter controls how many of the neighbourhood atoms are added to the
                         focus atoms specified by the parameter 'from_atoms'. """

        if isinstance(compound, str):
            compound = CompoundConversionUtils.string_to_mol(compound)

        try:
            # Fetch the respective distance matrix.
            distance_matrix = GetDistanceMatrix(compound)

            # Set the weight factor for the generation of the HSFP.
            weight_factor = np.max(distance_matrix) if neighbourhood_ext is None else neighbourhood_ext

            # Generate the base of the HSFP, which is basically the ECFP of the core atoms.
            core_fp = MolecularFingerprintsUtils.construct_ecfp(compound, radius=radius, bits=bits,
                                                                from_atoms=from_atoms, output_type="np_array")
            hsfp = np.array(core_fp)

            # Iterate through and add other layers to the HSFP.
            for i in range(0, int(weight_factor)):
                # Generate a fingerprint for the distance increment, and calculate the bitwise difference.
                atom_environment = CompoundStructureUtils.get_atom_environment(compound, from_atoms, n_degree=i+1)
                env_fp = MolecularFingerprintsUtils.construct_ecfp(compound, radius=radius, bits=bits,
                                                                   from_atoms=atom_environment, output_type="np_array")
                diff = np.bitwise_xor(core_fp, env_fp)

                # Add the weighted difference vector to the resulting HSFP vector.
                core_fp = core_fp + diff
                hsfp = hsfp + diff * 1/(i+2)

            # Return the resulting HSFP rounded to three decimals.
            return np.round(hsfp, 3)

        except Exception as exc_msg:
            if verbose:
                print("Exception occurred during the generation of the HSFP. Detailed message: {}".format(exc_msg))

            return None

    @staticmethod
    def tanimoto_similarity(ecfp_a: ExplicitBitVect, ecfp_b: ExplicitBitVect) -> float:
        """ Description: Return the Tanimoto similarity value between two ECFP fingerprints. """

        return TanimotoSimilarity(ecfp_a, ecfp_b)

    @staticmethod
    def bulk_tanimoto_similarity(ecfp: ExplicitBitVect, ecfp_pool: List[ExplicitBitVect]) -> List:
        """ Description: Return the Tanimoto similarity values between a single and a pool of ECFP fingerprints. """

        return BulkTanimotoSimilarity(ecfp, ecfp_pool)

    @staticmethod
    def dice_similarity(ecfp_a: ExplicitBitVect, ecfp_b: ExplicitBitVect) -> float:
        """ Description: Returns the Dice similarity value between two ECFP fingerprints. """

        return DiceSimilarity(ecfp_a, ecfp_b)

    @staticmethod
    def bulk_dice_similarity(ecfp: ExplicitBitVect, ecfp_pool: List[ExplicitBitVect]) -> List:
        """ Description: Returns the Dice similarity values between a single and a pool of ECFP fingerprints. """

        return BulkDiceSimilarity(ecfp, ecfp_pool)

    @staticmethod
    def tversky_similarity(ecfp_a: ExplicitBitVect, ecfp_b: ExplicitBitVect, a=0.5, b=1.0) -> float:
        """ Description: Returns the Tversky similarity value between two ECFP fingerprints. """

        return TverskySimilarity(ecfp_a, ecfp_b, a, b)

    @staticmethod
    def bulk_tversky_similarity(ecfp: ExplicitBitVect, ecfp_pool: List[ExplicitBitVect], a=0.5, b=1.0) -> List:
        """ Description: Returns the Tversky similarity values between a single and a pool of ECFP fingerprints. """

        return BulkTverskySimilarity(ecfp, ecfp_pool, a, b)
