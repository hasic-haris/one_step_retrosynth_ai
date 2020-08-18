"""
Author:      Hasic Haris (Phd Student @ Ishida Lab, Department of Computer Science, Tokyo Institute of Technology)
Created on:  January, 10th, 2020.
Description: This file contains functions for handling chemical reactions and their properties.
"""
import re

from rdkit.Chem import AllChem


# Done: 100%
def parse_reaction_roles(rsmiles, as_what="smiles"):
    """ Convert a reaction SMILES string to lists of reactants, agents and products in various data formats. """

    # Split the reaction SMILES string by the '>' symbol to obtain the reactants and products.
    roles = rsmiles.split(">")

    # NOTE: In some cases, there can be additional characters on the product side separated by a whitespace. For this
    # reason the product side string is always additionally split by the whitespace and the only the first element is
    # considered.

    # Parse the original SMILES strings from the reaction string including the reaction atom mappings.
    if as_what == "smiles":
        return [x for x in roles[0].split(".") if x != ""],\
               [x for x in roles[1].split(".") if x != ""],\
               [x for x in roles[2].split(" ")[0].split(".") if x != ""]

    # Parse the original SMILES strings from the reaction string excluding the reaction atom mappings.
    elif as_what == "smiles_no_maps":
        return [re.sub(r":[-+]?[0-9]+", "", x) for x in roles[0].split(".") if x != ""],\
               [re.sub(r":[-+]?[0-9]+", "", x) for x in roles[1].split(".") if x != ""],\
               [re.sub(r":[-+]?[0-9]+", "", x) for x in roles[2].split(" ")[0].split(".") if x != ""]

    # Parse the lists of atom map numbers from the reactions SMILES string.
    elif as_what == "atom_maps":
        return [[int(x[1:]) for x in re.findall(r":[-+]?[0-9]+", y)] for y in roles[0].split(".") if y != ""],\
               [[int(x[1:]) for x in re.findall(r":[-+]?[0-9]+", y)] for y in roles[1].split(".") if y != ""],\
               [[int(x[1:]) for x in re.findall(r":[-+]?[0-9]+", y)]
                for y in roles[2].split(" ")[0].split(".") if y != ""]

    # Parse the lists of number of mapped atoms per reactant and per product. Agents usually do not contain any mapping,
    # but they are included here for the sake of consistency.
    elif as_what == "mapping_numbers":
        return [len([el for el in roles[0].split(".") if el != ""]),
                len([el for el in roles[1].split(".") if el != ""]),
                len([el for el in roles[2].split(" ")[0].split(".") if el != ""])]

    # Parsings that include initial conversion to RDKit 'Mol' objects and need to include sanitization: mol,
    # mol_no_maps, canonical_smiles and canonical_smiles_no_maps.
    elif as_what in ["mol", "mol_no_maps", "canonical_smiles", "canonical_smiles_no_maps"]:
        reactants, agents, products = [], [], []

        # Iterate through all of the reactants.
        for reactant in roles[0].split("."):
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

        # Iterate through all of the agents.
        for agent in roles[1].split("."):
            if agent != "" and as_what in ["mol", "canonical_smiles"]:
                mol_maps = AllChem.MolFromSmiles(agent)
                AllChem.SanitizeMol(mol_maps)

                if as_what == "mol":
                    agents.append(mol_maps)
                    continue
                if as_what == "canonical_smiles":
                    agents.append(AllChem.MolToSmiles(mol_maps, canonical=True))
                    continue

            elif agent != "" and as_what in ["mol_no_maps", "canonical_smiles_no_maps"]:
                mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", agent))
                AllChem.SanitizeMol(mol_no_maps)

                if as_what == "mol_no_maps":
                    agents.append(mol_no_maps)
                    continue
                if as_what == "canonical_smiles_no_maps":
                    agents.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
                    continue

        # Iterate through all of the reactants.
        for product in roles[2].split(" ")[0].split("."):
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

        return reactants, agents, products

    # Raise exception for any other keyword.
    else:
        raise Exception("Unknown parsing type. Select one of the following: "
                        "'smiles', 'smiles_no_maps', 'canonical_smiles', 'canonical_smiles_no_maps', "
                        "'mol', 'atom_maps', 'mapping_numbers'.")
