import re
from rdkit.Chem import AllChem
from typing import Union, Tuple, List
from indigo import Indigo

from chemical_compounds import CompoundConversion


class ChemicalReactions:
    """
    AUTHOR: Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
    DATE: December 29th, 2020
    DESCRIPTION: This class encapsulates useful methods for handling chemical reactions.
    """

    @staticmethod
    def parse_reaction_roles(rxn_smiles: str, as_what="smiles") -> Tuple[List, List, List]:
        """ Convert a reaction SMILES string to lists of reactants, reagents and products. """

        # Split the reaction SMILES string by the '>' symbol to obtain the reactants and products. In cases of extended
        # reaction SMILES strings, there can be additional characters on the product side separated by a whitespace. For
        # this reason, the product side string is always additionally split by a whitespace symbol and only the first
        # element is considered to ensure correct parsing for every reaction SMILES variation.
        reactants = rxn_smiles.split(">")[0].split(".")
        agents = rxn_smiles.split(">")[1].split(".")
        products = rxn_smiles.split(">")[2].split(" ")[0].split(".")

        # Return the original reaction role sub-strings including the reaction atom mappings.
        if as_what == "smiles":
            return [r_smiles for r_smiles in reactants if r_smiles != ""],\
                   [a_smiles for a_smiles in agents if a_smiles != ""],\
                   [p_smiles for p_smiles in products if p_smiles != ""]

        # Return the original reaction role sub-strings excluding the reaction atom mappings.
        elif as_what == "smiles_no_maps":
            return [re.sub(r":[-+]?[0-9]+", "", r_smiles) for r_smiles in reactants if r_smiles != ""],\
                   [re.sub(r":[-+]?[0-9]+", "", a_smiles) for a_smiles in agents if a_smiles != ""],\
                   [re.sub(r":[-+]?[0-9]+", "", p_smiles) for p_smiles in products if p_smiles != ""]







        # Parsings that include initial conversion to RDKit Mol objects and need to include sanitization: mol, mol_no_maps,
        # canonical_smiles and canonical_smiles_no_maps.
        elif as_what in ["mol", "mol_no_maps", "canonical_smiles", "canonical_smiles_no_maps"]:
            reactants, reagents, products = [], [], []

            # Iterate through all of the reactants.
            for reactant in rxn_roles[0].split("."):
                if reactant != "" and as_what in ["mol", "canonical_smiles"]:
                    mol_maps = AllChem.MolFromSmiles(reactant)
                    AllChem.SanitizeMol(mol_maps)

                    if as_what == "mol":
                        reactants.append(mol_maps)
                        continue

                    if as_what == "canonical_smiles":
                        reactants.append(AllChem.MolToSmiles(mol_maps, canonical=True))
                        continue

                elif reactant != "" and as_what in ["mol_no_maps", "canonical_smiles_no_maps"]:
                    mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", reactant))
                    AllChem.SanitizeMol(mol_no_maps)

                    if as_what == "mol_no_maps":
                        reactants.append(mol_no_maps)
                        continue

                    if as_what == "canonical_smiles_no_maps":
                        reactants.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
                        continue

            # Iterate through all of the reagents.
            for reagent in rxn_roles[1].split("."):
                if reagent != "" and as_what in ["mol", "canonical_smiles"]:
                    mol_maps = AllChem.MolFromSmiles(reagent)
                    AllChem.SanitizeMol(mol_maps)

                    if as_what == "mol":
                        reagents.append(mol_maps)
                        continue

                    if as_what == "canonical_smiles":
                        reagents.append(AllChem.MolToSmiles(mol_maps, canonical=True))
                        continue

                elif reagent != "" and as_what in ["mol_no_maps", "canonical_smiles_no_maps"]:
                    mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", reagent))
                    AllChem.SanitizeMol(mol_no_maps)

                    if as_what == "mol_no_maps":
                        reagents.append(mol_no_maps)
                        continue

                    if as_what == "canonical_smiles_no_maps":
                        reagents.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
                        continue

            # Iterate through all of the reactants.
            for product in rxn_roles[2].split(" ")[0].split("."):
                if product != "" and as_what in ["mol", "canonical_smiles"]:
                    mol_maps = AllChem.MolFromSmiles(product)
                    AllChem.SanitizeMol(mol_maps)

                    if as_what == "mol":
                        products.append(mol_maps)
                        continue

                    if as_what == "canonical_smiles":
                        products.append(AllChem.MolToSmiles(mol_maps, canonical=True))
                        continue

                elif product != "" and as_what in ["mol_no_maps", "canonical_smiles_no_maps"]:
                    mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", product))
                    AllChem.SanitizeMol(mol_no_maps)

                    if as_what == "mol_no_maps":
                        products.append(mol_no_maps)
                        continue

                    if as_what == "canonical_smiles_no_maps":
                        products.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
                        continue

            return reactants, reagents, products

        # Raise exception for any other keyword.
        else:
            raise Exception("Unknown parsing type. Select one of the following: "
                            "'smiles', 'smiles_no_maps', 'atom_maps', 'mapping_numbers', 'mol', 'mol_no_maps', "
                            "'canonical_smiles', 'canonical_smiles_no_maps'.")


class ReactionMapper:
    """
    AUTHOR: Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
    DATE: December 29th, 2020
    DESCRIPTION: This class encapsulates useful methods for handling the mapping of chemical reactions.
    """

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
            reactant_smiles, _, product_smiles = CompoundConversion.rxn_smiles_to_rxn_roles(rxn_smiles)

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


print("lol")
