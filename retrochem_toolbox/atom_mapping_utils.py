"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  December 20th, 2020
Edited on:   January 31st, 2021
"""

import re

from indigo import Indigo
from typing import Union

from rdkit.Chem.AllChem import Mol
# noinspection PyUnresolvedReferences
from rdchiral.template_extractor import extract_from_reaction

from reaction_utils import ReactionConversionUtils


# noinspection PyArgumentList
class AtomMappingUtils:
    """ Description: Group of methods for the handling of chemical compound atom mapping. """

    @staticmethod
    def remove_compound_mapping(compound: Union[str, Mol]) -> Union[str, Mol]:
        """ Description: Remove any existing mapping from a chemical compound. """

        if isinstance(compound, str):
            return re.sub(r":[-+]?[0-9]+", "", compound)

        else:
            for atom in compound.GetAtoms():
                atom.SetAtomMapNum(0)

            return compound

    @staticmethod
    def atom_map_reaction_indigo(reaction_smiles_string: str, timeout_period: int, existing_mapping="discard",
                                 verbose=False) -> Union[str, None]:
        """ Description: Atom map a reaction SMILES string using the Epam Indigo reaction atom mapper API. """

        try:
            indigo_mapper = Indigo()
            indigo_mapper.setOption("aam-timeout", timeout_period)

            reaction_object = indigo_mapper.loadReaction(reaction_smiles_string)
            reaction_object.automap(existing_mapping)

            return reaction_object.smiles()

        except Exception as exc_msg:
            if verbose:
                print("Exception occurred during atom mapping of the reaction SMILES string '{}'. "
                      "Detailed message: {}".format(reaction_smiles_string, exc_msg))

            return None

    @staticmethod
    def extract_reaction_template(reaction_smiles_string: str, verbose=False) -> Union[str, None]:
        """ Description: Extract a reaction template from an atom mapped reaction SMILES string using RDChiral. """

        try:
            reactants, _, products = ReactionConversionUtils.parse_roles_from_reaction_smiles(reaction_smiles_string)

            reaction_template = extract_from_reaction({"reactants": ".".join(reactants),
                                                       "products": ".".join(products),
                                                       "_id": "0"})

            if reaction_template is not None and "reaction_smarts" in reaction_template.keys():
                return reaction_template["reaction_smarts"]
            else:
                return None

        except Exception as exc_msg:
            if verbose:
                print("Exception occurred during reaction template extraction from the reaction SMILES string '{}'. "
                      "Detailed message: {}".format(reaction_smiles_string, exc_msg))

            return None
