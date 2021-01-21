from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from typing import Union, Tuple, List
import numpy as np
from rdkit.Chem import AllChem, DataStructs
from rdkit.DataStructs import cDataStructs
from rdkit.Chem import SaltRemover


class CompoundConversion:
    """
    AUTHOR: Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
    DATE: December 29th, 2020
    DESCRIPTION: This class encapsulates useful methods for handling chemical compounds.
    """

    @staticmethod
    def smiles_to_mol(compound_smiles: str, verbose=True) -> Union[AllChem.Mol, None]:
        """ Convert a chemical compound SMILES string to a RDKit Mol object. """

        # Create a placeholder for the RDKit Mol object.
        rdkit_mol = None

        # Try to convert the compound SMILES string to a RDKit Mol object and sanitize the generated structure.
        try:
            rdkit_mol = AllChem.MolFromSmiles(compound_smiles)
            AllChem.SanitizeMol(rdkit_mol)

            return rdkit_mol

        # If an exception occurs for any reason, print the message if indicated, and return None as the final result.
        except Exception as exc_msg:
            if verbose:
                if rdkit_mol is None:
                    print("Exception occurred during the conversion process ", end="")
                else:
                    print("Exception occurred during sanitization process ", end="")

                print("'{}'. Detailed message:\n{}".format(compound_smiles, exc_msg))

            return None

    @staticmethod
    def smiles_to_canonical_smiles(compound_smiles: str, verbose=True) -> Union[str, None]:
        """ Convert a chemical compound SMILES string to a canonical SMILES string. """

        # Try to convert the input SMILES string to a canonical SMILES string using RDKit.
        try:
            return AllChem.MolToSmiles(CompoundConversion.smiles_to_mol(compound_smiles), canonical=True)

        # If an exception occurs for any reason, print the message if indicated, and return None as the final result.
        except Exception as exc_msg:
            if verbose:
                print("Exception occurred during the conversion process. Detailed message: {}".format(exc_msg))

            return None

    @staticmethod
    def mol_to_canonical_smiles(rdkit_mol: AllChem.Mol, verbose=True) -> Union[str, None]:
        """ Convert a chemical compound RDKit Mol object to a canonical SMILES string. """

        # Try to convert the input RDKit Mol object to a canonical SMILES string using RDKit.
        try:
            return AllChem.MolToSmiles(rdkit_mol, canonical=True)

        # If an exception occurs for any reason, print the message if indicated, and return None as the final result.
        except Exception as exc_msg:
            if verbose:
                print("Exception occurred during the conversion process. Detailed message: {}".format(exc_msg))

            return None


class CompoundNormalization:
    """ Description: Group of methods for handling the correctness of molecular structures. """

    @staticmethod
    def remove_salts(smiles: str, salt_list_file_path=None, apply_ad_hoc_stripper=False) -> str:
        """ Description: Remove salts from a SMILES string using the RDKit salt stripper. """

        try:
            # Apply the default RDKit salt stripper first.
            salt_remover = SaltRemover.SaltRemover()
            no_salt_smiles = CompoundConversion.mol_to_canonical_smiles(
                salt_remover.StripMol(CompoundConversion.smiles_to_mol(smiles)))

            # Apply the RDKit salt stripper if a user defined list of salts is specified.
            if salt_list_file_path is not None:
                salt_remover = SaltRemover.SaltRemover(defnFilename=salt_list_file_path)
                no_salt_smiles = CompoundConversion.mol_to_canonical_smiles(
                    salt_remover.StripMol(CompoundConversion.smiles_to_mol(no_salt_smiles)))

            # If there are some salts left behind, apply the 'ad hoc' salt stripper based on the symbol '.'.
            # NOTE: This is risky and it should only be applied if the SMILES string is one molecule.
            if apply_ad_hoc_stripper:
                no_salt_smiles = CompoundConversion.smiles_to_canonical_smiles(
                    sorted(no_salt_smiles.split("."), key=len, reverse=True)[0])

            return no_salt_smiles

        # If an exception occurs for any reason, print the message and return None.
        except Exception as ex:
            print(
                "Exception occured during stripping of the salts from the molecule '{}'. Detailed exception message:\n{}".format(
                    smiles, ex))
            return None

    @staticmethod
    def normalize_structure(smiles: str) -> str:
        """ Description: Use RDKit to normalize the specified molecule and return it as canonical SMILES. """

        try:
            mol = rdMolStandardize.Normalize(CompoundConversion.smiles_to_mol(smiles))

            return CompoundConversion.mol_to_canonical_smiles(mol)

        # If an exception occurs for any reason, print the message and return None.
        except Exception as ex:
            print(
                "Exception occured during stripping of the salts from the molecule '{}'. Detailed exception message:\n{}".format(
                    smiles, ex))
            return None


class CompoundRepresentations:
    """
    AUTHOR: Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
    DATE: December 29th, 2020
    DESCRIPTION: This class encapsulates useful methods for handling chemical compounds.
    """

    @staticmethod
    def __np_array_to_binary_vector(np_arr):
        """ Converts a NumPy array to a RDKit ExplicitBitVector type. """

        binary_vector = DataStructs.ExplicitBitVect(len(np_arr))
        binary_vector.SetBitsFromList(np.where(np_arr)[0].tolist())

        return binary_vector

    @staticmethod
    def generate_ecfp(mol, radius, bits, from_atoms=None, output_type="bit_vector", as_type="np_int"):
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

    @staticmethod
    def mol_to_ecfp(rdkit_mol: AllChem.Mol, radius: int, bits: int, output_type="bit_vector") -> Union[str, None]:
        """ Convert a chemical compound RDKit Mol object to a Extended-Connectivity FingerPrint or ECFP. """

        # Try to convert the input RDKit Mol object to a ECFP representation.
        try:
            ecfp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, radius=radius, nBits=bits)

            # If it is specified by the output type parameter, convert the result to an NumPy array.
            if output_type == "np_array":
                result_ecfp = np.array([])
                cDataStructs.ConvertToNumpyArray(ecfp, result_ecfp)
                ecfp = result_ecfp.astype(np.int)

            return ecfp

        # If an exception occurs for any reason, print the message and return None.
        except Exception as ex:
            print("Exception occured during the generation of the fingerprint. Detailed message: {}".format(ex))
            return None