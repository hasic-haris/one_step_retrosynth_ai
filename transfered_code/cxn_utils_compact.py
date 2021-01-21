"""
    DESCRIPTION: Compacted version of the ChemAxon re-implementation containing only useful functionalities.
"""

import re

from typing import Union, Tuple, List
from rdkit.Chem import AllChem

from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from rdchiral.main import rdchiralRun, rdchiralRunText
from rdchiral.template_extractor import extract_from_reaction


class PyReactor:

    @staticmethod
    def forward_apply(reactant_mols: Union[Tuple[str], List[str], Tuple[AllChem.Mol], List[AllChem.Mol]],
                      rxn_template: str, input_format="smiles", return_type="mol") -> List[list]:
        """ Apply a specified reaction rule template on a single or group of reactant molecules. """

        if len(reactant_mols) == 0:
            raise Exception("The given group of reactant molecules is empty.")

        if isinstance(reactant_mols[0], str):
            try:
                reactant_mols = [CxnUtils.string_to_molecule(reactant_mol, input_format=input_format) for reactant_mol
                                 in reactant_mols]
            except:
                raise Exception(
                    "Unable to convert all of the input reactant molecules ('{}'). Please check their validity.".format(
                        input_format))

        # Create and Sanitize the RDKit ChemicalReaction object.
        rxn = AllChem.ReactionFromSmarts(rxn_template)
        AllChem.SanitizeRxn(rxn)

        # Remove any potential duplicate suggestions.
        products_suggestions = PyReactor.__remove_duplicate_reactants(rxn.RunReactants(reactant_mols))

        if return_type == "str":
            return products_suggestions
        else:
            # Convert the SMILES strings into RDKit Mol objects and check their chemical validity.
            return PyReactor.__check_suggested_mol_validity(products_suggestions)

    @staticmethod
    def reverse_apply(product_mols: Union[Tuple[str], List[str], Tuple[AllChem.Mol], List[AllChem.Mol]],
                      rxn_template: str, input_format="smiles", return_type="mol") -> List[list]:
        """ Apply a specified reaction rule template backwards on a single or group of product molecules. """

        if len(product_mols) == 0:
            raise Exception("The given group of product molecules is empty.")

        if isinstance(product_mols[0], str):
            try:
                product_mols = [CxnUtils.string_to_molecule(product_mol, input_format=input_format) for product_mol in
                                product_mols]
            except:
                raise Exception(
                    "Unable to convert all of the input product molecules ('{}'). Please check their validity.".format(
                        input_format))

        # Split the reaction rule template SMARTS string into reactants and products substrings.
        reactants_substr, _, products_substr = rxn_template.split(">")

        # Generate a reverse reaction rule template SMARTS string.
        rxn_template = ">>".join([products_substr, reactants_substr])

        # Create and Sanitize the RDKit ChemicalReaction object.
        rxn = AllChem.ReactionFromSmarts(rxn_template)
        AllChem.SanitizeRxn(rxn)

        # Remove any potential duplicate suggestions.
        reactants_suggestions = PyReactor.__remove_duplicate_reactants(rxn.RunReactants(product_mols))

        if return_type == "str":
            return reactants_suggestions
        else:
            # Convert the SMILES strings into RDKit Mol objects and check their chemical validity.
            return PyReactor.__check_suggested_mol_validity(reactants_suggestions)

    @staticmethod
    def __remove_duplicate_reactants(mol_combinations: List[Tuple[AllChem.Mol]]) -> List[Tuple[str]]:
        """ Remove duplicate molecule combinations from a given list."""

        return list(set(
            [tuple([AllChem.MolToSmiles(mol, canonical=True) for mol in mol_combination]) for mol_combination in
             mol_combinations]))

    @staticmethod
    def __check_suggested_mol_validity(mol_combinations: List[Tuple[str]]) -> List[Tuple[AllChem.Mol]]:
        """ Convert the contents of molecule combinations from SMILES to RDKit Mol format, and check their chemical validity. """

        correct_mol_combinations = []

        for mol_combination in mol_combinations:
            try:
                correct_mol_combinations.append(
                    tuple([CxnUtils.string_to_molecule(mol, input_format="smiles") for mol in mol_combination]))
            except:
                continue


class CxnUtils:

    def __init__(self, rollout_list):
        self.rollout_list = self.__get_reaction_list(rollout_list)

    @staticmethod
    def string_to_molecule(input_mol: str, input_format="smiles") -> AllChem.Mol:
        """ Converts a string molecule representation to a RDKit Mol object. """

        # Converting the input string to the RDKit Mol object.
        mol = None
        try:
            if input_format == "smiles":
                mol = AllChem.MolFromSmiles(input_mol)
            elif input_format == "smarts":
                mol = AllChem.MolFromSmarts(input_mol)
            else:
                raise Exception("Choose one of the currently supported formats: 'smiles', or 'smarts'.")

            # Sanitizing the generated RDKit Mol object.
            AllChem.SanitizeMol(mol)

            return mol

        except Exception as ex:
            if mol is None:
                print("Exception occured during the conversion process for the molecule ", end="")
            else:
                print("Exception occured during sanitization of the molecule ", end="")
                mol = None

            print("'{}'. Detailed exception message:\n{}".format(input_mol, ex))

    @staticmethod
    def react_product_to_reactants(product_mols: Union[Tuple[str], List[str], Tuple[AllChem.Mol], List[AllChem.Mol]],
                                   rxn_template: str, input_format="smiles", return_type="str") -> List[list]:
        """ Applies a given reaction rule template to specific product molecule(s). """

        try:
            return PyReactor.reverse_apply(product_mols, rxn_template, input_format, return_type)
        except Exception as ex:
            raise Exception(
                "The reverse application of the reaction rule template was unsuccessful. Detailed exception message:\n{}".format(
                    ex))

    def is_terminal(self, input_mols: List[str]) -> bool:
        """ Checks if a list of reactants is found in the list of starting materials. """

        for mol in input_mols:
            for rxn in self.rollout_list:
                try:
                    # If a reaction can be applied on a molecule from the specified list, return False.
                    if len(self.react_product_to_reactants((mol,), rxn)) != 0:
                        return False
                except:
                    pass
        return True

    @staticmethod
    def __get_reaction_list(file_path: str, sep="\n") -> List[str]:
        """ Reads a textual file containing a list of specified SMARTS reaction rules. """

        try:
            return open(file_path, "r").read().split(sep)
        except Exception as ex:
            print(
                "Exception occured during the openning of the reaction rules SMARTS file. Detailed exception message:\n{}".format(
                    ex))

    @staticmethod
    def __get_mol_list(input_mols: List[str], input_format="smiles") -> List[AllChem.Mol]:
        """ Converts a list of molecules in string notation into a list of RDKit Mol objects. """

        try:
            return [CxnUtils.string_to_molecule(mol, input_format=input_format) for mol in input_mols]
        except Exception(ex):
            print(
                "Exception occured during the conversion of the molecules given in the specified list. Detailed exception message:\n{}".format(
                    ex))
