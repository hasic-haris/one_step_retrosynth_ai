import re
import numpy as np

from indigo import *
from rdkit.Chem import AllChem, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.DataStructs import cDataStructs
from typing import List, Tuple

from rdchiral.template_extractor import extract_from_reaction


class ConversionUtils:
    """
    Description:
        Group of methods for handling the conversion of formats using RDKit.
    Configuration:
        No configuration files needed.
    """

    @staticmethod
    def smiles_to_mol(smiles: str, verbose=False) -> AllChem.Mol:
        """
        Description:
            Convert a SMILES string to a RDKit Mol object.
            Returns None if either the conversion or the sanitization of the SMILES string fail.
        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (AllChem.Mol): An RDKit Mol object representing the given SMILES string.
        """

        mol = None

        # Try to convert the SMILES string into a RDKit Mol object and sanitize it.
        try:
            mol = AllChem.MolFromSmiles(smiles)
            AllChem.SanitizeMol(mol)

            return mol

        # If an exception occurs for any reason, print the error message if indicated, and return None.
        except Exception as ex:
            if verbose:
                if mol is None:
                    print("Exception occured during the conversion process of ", end="")
                else:
                    print("Exception occured during the sanitization of ", end="")

                print("'{}'. Detailed exception message:\n{}".format(smiles, ex))

            return None

    @staticmethod
    def smarts_to_mol(smarts: str, verbose=False) -> AllChem.Mol:
        """
        Description:
            Convert a SMARTS string to a RDKit Mol object.
            Returns None if either the conversion or the sanitization of the SMARTS string fail.
        Input:
            smarts (str): A SMARTS string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (AllChem.Mol): An RDKit Mol object representing the given SMARTS string.
        """

        smarts_mol = None

        # Try to convert the SMARTS string into a RDKit Mol object and sanitize it.
        try:
            smarts_mol = AllChem.MolFromSmarts(smarts)
            AllChem.SanitizeMol(smarts_mol)

            return smarts_mol

        # If an exception occurs for any reason, print the error message if indicated, and return None.
        except Exception as ex:
            if verbose:
                if smarts_mol is None:
                    print("Exception occured during the conversion process of ", end="")
                else:
                    print("Exception occured during the sanitization of ", end="")

                print("'{}'. Detailed exception message:\n{}".format(smarts, ex))

            return None

    @staticmethod
    def smiles_to_canonical_smiles(smiles: str, verbose=False) -> str:
        """
        Description:
            Convert a SMILES string to a Canonical SMILES string.
            Returns None if either the conversion to the Canonical SMILES string fails.
        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A Canonical SMILES string representing the given chemical structure.
        """

        try:
            return AllChem.MolToSmiles(ConversionUtils.smiles_to_mol(smiles), canonical=True)

        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as ex:
            if verbose:
                print(
                    "Exception occured during the conversion of '{}' to Canonical SMILES. Detailed message: {}".format(
                        smiles, ex))

            return None

    @staticmethod
    def mol_to_canonical_smiles(mol: AllChem.Mol, verbose=False) -> str:
        """
        Description:
            Convert a RDKit Mol object to a Canonical SMILES string.
            Returns None if either the conversion to the Canonical SMILES string fails.
        Input:
            mol (AllChem.Mol): An RDKit Mol object representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A Canonical SMILES string representing the given chemical structure.
        """

        try:
            return AllChem.MolToSmiles(mol, canonical=True)

        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as ex:
            if verbose:
                print(
                    "Exception occured during the conversion of the RDKit Mol object to Canonical SMILES. Detailed message: {}".format(
                        ex))

            return None

    @staticmethod
    def mol_to_ecfp(mol: AllChem.Mol, radius: int, bits: int, output_type="bit_vector"):
        """ Description: Convert a RDKit Mol object to a Extended-connectivity fingerprint or ECFP. """

        try:
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits)

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

    @staticmethod
    def rxn_smiles_to_rxn_roles(rxn_smiles: str, remove_mapping=False) -> Tuple[List["str"]]:
        """ Description: Parse the reaction roles from the reaction SMILES string. """

        try:
            # Split the reaction SMILES string by the '>' symbol to obtain the reactants and products.
            # NOTE: In some cases, there can be additional characters on the product side separated by a whitespace.
            # For this reason the product side string is always additionally split by the whitespace and the only
            # the first element is considered.
            reactant_smiles = [r_smi for r_smi in rxn_smiles.split(">")[0].split(".") if r_smi != ""]
            agent_smiles = [a_smi for a_smi in rxn_smiles.split(">")[1].split(".") if a_smi != ""]
            product_smiles = [p_smi for p_smi in rxn_smiles.split(">")[2].split(" ")[0].split(".") if p_smi != ""]

            if remove_mapping:
                reactant_smiles = [ConversionUtils.smiles_to_canonical_smiles(re.sub(r":[-+]?[0-9]+", "", r_smi)) for
                                   r_smi in reactant_smiles]
                agent_smiles = [ConversionUtils.smiles_to_canonical_smiles(re.sub(r":[-+]?[0-9]+", "", a_smi)) for a_smi
                                in agent_smiles]
                product_smiles = [ConversionUtils.smiles_to_canonical_smiles(re.sub(r":[-+]?[0-9]+", "", p_smi)) for
                                  p_smi in product_smiles]

            return reactant_smiles, agent_smiles, product_smiles

        # If an exception occurs for any reason, print the message and return None.
        except Exception as ex:
            print("Exception occured during the parsing of the reaction roles. Detailed message: {}".format(ex))
            return None, None, None


class ReactionMapper:
    """ Description: Python multiprocessing-friendly wrapper for the Epam Indigo chemical reaction mapping API.
        Author:      Haris Hasic, Phd Student @ Tokyo Institute of Technology
        Created on:  December 8th, 2020 """

    @staticmethod
    def atom_map_reaction(rxn_smiles: str, timeout_period: int, existing_mapping="discard") -> str:
        """ Description: Atom map a reaction SMILES string using the Epam Indigo reaction atom mapper API. Any existing
                         mapping will be handled according to the value of the parameter 'existing_mapping'. Because it
                         can be a time consuming process, a timeout occurs after 'timeout_period' ms. """

        try:
            # Instantiate the Indigo class object and set the timeout period.
            indigo_mapper = Indigo()
            indigo_mapper.setOption("aam-timeout", timeout_period)

            # Return the atom mapping of the reaction SMILES string.
            rxn = indigo_mapper.loadReaction(rxn_smiles)
            rxn.automap(existing_mapping)

            return rxn.smiles()

        # If an exception occurs for any reason, print the message and return None.
        except Exception as ex:
            print("Exception occured during atom mapping of the reaction SMILES. Detailed message: {}".format(ex))
            return None

    @staticmethod
    def extract_reaction_template(rxn_smiles: str) -> str:
        """ Description:  Extract a reaction template from a SMILES string using the RDChiral library.
            Authors Note: This function relies on the GetSubstructMatches function from RDKit and if the
                          reaction contains many large molecules, the process can take a lot of time. """

        try:
            # Parse the reaction roles from the reaction SMILES.
            reactant_smiles, _, product_smiles = ConversionUtils.rxn_smiles_to_rxn_roles(rxn_smiles)

            reactant_side = ".".join(reactant_smiles)
            product_side = ".".join(product_smiles)

            # Extract the templates from the reaction SMILES using RDChiral.
            reaction_template = extract_from_reaction(
                {"reactants": reactant_side, "products": product_side, "_id": "0"})

            # Return the reaction SMARTS result if the processing finished correctly.
            if reaction_template is not None and "reaction_smarts" in reaction_template.keys():
                return reaction_template["reaction_smarts"]
            else:
                return None

        # If an exception occurs for any reason, print the message and return None.
        except Exception as ex:
            print(
                "Exception occured during the reaction rule template extraction from the reaction SMILES. Detailed message: {}".format(
                    ex))
            return None


class StructureUtils:
    """
    Description:
        Group of methods for handling the correctness of molecular structures.
    Configuration:
        No configuration files needed.
    """

    @staticmethod
    def remove_salts(smiles: str, salt_list_file_path=None, apply_ad_hoc_stripper=False, verbose=False) -> str:
        """
        Description:
            Remove salts from a SMILES string using the RDKit salt stripper.
            Returns None if the RDKit salt removal process fails.
        Input:
            smiles (str): A SMILES string representing a chemical structure.
            salt_list_file_path (str): A path string to a user-defined list of salt SMILES in .txt format.
            apply_ad_hoc_stripper (bool): A bool value indicating if the ad-hoc salt stripper will be applied.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A Canonical SMILES string representing the given chemical structure without salts.
        """

        try:
            # Apply the default RDKit salt stripper first.
            salt_remover = SaltRemover.SaltRemover()
            no_salt_smiles = ConversionUtils.mol_to_canonical_smiles(
                salt_remover.StripMol(ConversionUtils.smiles_to_mol(smiles)))

            # Apply the RDKit salt stripper if a user defined list of salts is specified.
            if salt_list_file_path is not None:
                salt_remover = SaltRemover.SaltRemover(defnFilename=salt_list_file_path)
                no_salt_smiles = ConversionUtils.mol_to_canonical_smiles(
                    salt_remover.StripMol(ConversionUtils.smiles_to_mol(no_salt_smiles)))

            # If there are some salts left behind, apply the 'ad hoc' salt stripper based on the symbol '.'.
            # NOTE: This is risky and it should only be applied if the SMILES string is one molecule, not on reaction SMILES.
            if apply_ad_hoc_stripper:
                no_salt_smiles = ConversionUtils.smiles_to_canonical_smiles(
                    sorted(no_salt_smiles.split("."), key=len, reverse=True)[0])

            return no_salt_smiles

        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as ex:
            if verbose:
                print(
                    "Exception occured during stripping of the salts from '{}'. Detailed exception message:\n{}".format(
                        smiles, ex))

            return None

    @staticmethod
    def normalize_structure(smiles: str, verbose=False) -> str:
        """
        Description:
            Use RDKit to normalize the specified molecule and return it as canonical SMILES.
            Returns None if the RDKit normalization process fails.
        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.
        Output:
            (str): A Canonical SMILES string representing the normalized chemical structure.
        """

        try:
            mol = rdMolStandardize.Normalize(ConversionUtils.smiles_to_mol(smiles))

            return ConversionUtils.mol_to_canonical_smiles(mol)

        # If an exception occurs for any reason, print the message if indicated, and return None.
        except Exception as ex:
            if verbose:
                print("Exception occurred during the normalization of '{}'. Detailed exception message:\n{}".format(
                    smiles, ex))

            return None
