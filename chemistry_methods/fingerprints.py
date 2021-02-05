import numpy as np
from rdkit.Chem import AllChem, DataStructs
from rdkit.DataStructs import cDataStructs

from chemistry_methods.molecules import get_atom_environment


class MolecularFingerprintUtils:
    """ Description: A group of methods for constructing and similarity checking of molecular fingerprints."""

    @staticmethod
    def np_array_to_binary_vector(np_arr):
        """ Description: Converts a NumPy array to the RDKit ExplicitBitVector type. """

        binary_vector = DataStructs.ExplicitBitVect(len(np_arr))
        binary_vector.SetBitsFromList(np.where(np_arr)[0].tolist())

        return binary_vector

    @staticmethod
    def construct_ecfp(mol, radius, bits, from_atoms=None, output_type="bit_vector", as_type="np_int", verbose=False):
        """ Description: Returns the Extended Connectivity Fingerprint (ECFP) representation of the whole molecule or
            just from specific atoms, if it is specified through the 'from_atoms' parameter. The type of the fingerprint
            and the individual bits can be adjusted with the parameters 'output_type' and 'as_type', respectively. """

        # Check if the input molecule is given in the SMILES or in the RDKit Mol format.
        if isinstance(mol, str):
            # Generate the RDKit Mol object from the input SMILES string.
            mol = AllChem.MolFromSmiles(mol)
            # Sanitize the molecule.
            AllChem.SanitizeMol(mol)

        try:
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

        # If it fails for any reason, display the error message if indicated and return None.
        except Exception as exception:
            if verbose:
                print("The ECFP generation of has failed. Detailed message: {}".format(exception))

            return None

    @staticmethod
    def construct_hsfp(mol, radius, bits, from_atoms, neighbourhood_ext=None, verbose=False):
        """ Description: Returns the Hot Spot Fingerprint (HSFP) representation of the whole molecule (default) or just
            from specific atoms, if it is specified through the 'from_atoms' parameter, as a NumPy array. If the
            parameter 'from_atoms' includes all of the atoms of the molecule, the constructed fingerprint is equivalent
            to the ECFP representation of the whole molecule. The 'neighbourhood_ext' parameter controls how many of the
            neighbourhood atoms are added to the focus atoms specified by the parameter 'from_atoms'. """

        # Check if the input molecule is given in SMILES or in the RDKit Mol format.
        if isinstance(mol, str):
            # Generate the RDKit Mol object from the input SMILES string.
            mol = AllChem.MolFromSmiles(mol)
            # Sanitize the molecule.
            AllChem.SanitizeMol(mol)

        try:
            # Fetch the respective distance matrix.
            distance_matrix = AllChem.GetDistanceMatrix(mol)

            # Set the weight factor for the generation of the HSFP.
            weight_factor = np.max(distance_matrix) if neighbourhood_ext is None else neighbourhood_ext

            # Generate the base of the HSFP, which is basically the ECFP of the core atoms.
            core_fp = MolecularFingerprintUtils.construct_ecfp(mol, radius=radius, bits=bits, from_atoms=from_atoms,
                                                               output_type="np_array")
            hsfp = np.array(core_fp)

            # Iterate through and add other layers to the HSFP.
            for i in range(0, int(weight_factor)):
                # Generate a fingerprint for the distance increment, and calculate the bitwise difference.
                atom_environment = get_atom_environment(from_atoms, mol, degree=i+1)
                env_fp = MolecularFingerprintUtils.construct_ecfp(mol, radius=radius, bits=bits,
                                                                  from_atoms=atom_environment, output_type="np_array")
                diff = np.bitwise_xor(core_fp, env_fp)

                # Add the weighted difference vector to the resulting HSFP vector.
                core_fp = core_fp + diff
                hsfp = hsfp + diff * 1/(i+2)

            # Return the resulting HSFP rounded to three decimals.
            return np.round(hsfp, 3)

        # If it fails for any reason, display the error message if indicated and return None.
        except Exception as exception:
            if verbose:
                print("The HSFP generation of has failed. Detailed message: {}".format(exception))

            return None

    @staticmethod
    def tanimoto_similarity(ecfp1, ecfp2):
        """ Description: Returns the Tanimoto similarity value between two fingerprints. """

        return DataStructs.TanimotoSimilarity(ecfp1, ecfp2)

    @staticmethod
    def bulk_tanimoto_similarity(ecfp, ecfp_pool):
        """ Description: Returns the Tanimoto similarity values between a single and a pool of fingerprints. """

        return DataStructs.BulkTanimotoSimilarity(ecfp, ecfp_pool)

    @staticmethod
    def dice_similarity(ecfp1, ecfp2):
        """ Description: Returns the Dice similarity value between two fingerprints. """

        return DataStructs.DiceSimilarity(ecfp1, ecfp2)

    @staticmethod
    def bulk_dice_similarity(ecfp, ecfp_pool):
        """ Description: Returns the Dice similarity values between a single and a pool of fingerprints. """

        return DataStructs.BulkDiceSimilarity(ecfp, ecfp_pool)

    @staticmethod
    def tversky_similarity(ecfp1, ecfp2, a=0.5, b=1.0):
        """ Description: Returns the Tversky similarity value between two fingerprints using the parameters a and b. """

        return DataStructs.TverskySimilarity(ecfp1, ecfp2, a, b)

    @staticmethod
    def bulk_tversky_similarity(ecfp, ecfp_pool, a=0.5, b=1.0):
        """ Description: Returns the Tversky similarity values between a single and a pool of fingerprints using the
            parameters a and b. """

        return DataStructs.BulkTverskySimilarity(ecfp, ecfp_pool, a, b)
