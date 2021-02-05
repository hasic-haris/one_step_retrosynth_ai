"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  January 10th, 2020
Edited on:   January 31st, 2020
"""

import re

from copy import deepcopy
from typing import List, Tuple, Union

from rdkit.Chem.AllChem import Atom, Mol, RWMol, ChemicalReaction
from rdkit.Chem.AllChem import ReactionFromSmarts, ReactionToSmiles, ReactionToSmarts, SanitizeRxn

from retrochem_toolbox.compound_utils import CompoundConversionUtils
from data_methods.helpers import merge_common


class ReactionConversionUtils:
    """ Description: Group of methods for the handling of chemical reaction conversions. """

    @staticmethod
    def string_to_reaction(reaction_string: str, str_format="smiles", verbose=True) -> Union[ChemicalReaction, None]:
        """ Description: Convert a chemical reaction string representation to a ChemicalReaction object.
                         In the case of reactions, it is usually better to rely on individual compound parsing. """

        reaction_object = None

        try:
            if str_format in ["smiles", "smarts"]:
                reaction_object = ReactionFromSmarts(reaction_string)
            else:
                raise Exception("Supported string formats are 'smiles' and 'smarts'. Got: '{}'.".format(str_format))

            SanitizeRxn(reaction_object)

            return reaction_object

        except Exception as exc_msg:
            if verbose:
                if reaction_object is None:
                    print("Exception occurred during the conversion of ", end="")
                else:
                    print("Exception occurred during the sanitization of ", end="")

                print("'{}'. Detailed message:\n{}".format(reaction_string, exc_msg))

            return None

    @staticmethod
    def reaction_to_string(reaction_object: ChemicalReaction, str_format="smiles", canonical=True,
                           verbose=False) -> Union[str, None]:
        """ Description: Convert a ChemicalReaction object to a string representation.
                         In the case of reactions, it is usually better to rely on individual compound parsing. """

        try:
            if str_format == "smiles":
                return ReactionToSmiles(reaction_object, canonical=canonical)
            elif str_format == "smarts":
                return ReactionToSmarts(reaction_object)
            else:
                raise Exception("Supported string formats are 'smiles' and 'smarts'. Got: '{}'.".format(str_format))

        except Exception as exc_msg:
            if verbose:
                print("Exception occurred during the conversion of the Mol object. "
                      "Detailed message: {}".format(exc_msg))

            return None

    @staticmethod
    def string_to_canonical_string(reaction_string: str, str_format="smiles", verbose=True) -> Union[str, None]:
        """ Description: Canonicalize a chemical reaction string representation.
                         In the case of reactions, it is usually better to rely on individual compound parsing. """

        return ReactionConversionUtils.reaction_to_string(
            ReactionConversionUtils.string_to_reaction(reaction_string, str_format=str_format, verbose=verbose),
            str_format=str_format, canonical=True, verbose=verbose)

    @staticmethod
    def parse_roles_from_reaction_smiles(reaction_smiles: str, as_what="smiles") -> Tuple:
        """ Description: Convert a reaction SMILES string to lists of reactant, reagent and product information. """

        # Split the reaction SMILES string by the '>' symbol to obtain the reactants and products. In cases of extended
        # reaction SMILES strings, there can be additional characters on the product side separated by a whitespace. For
        # this reason, the product side string is always additionally split by a whitespace symbol and only the first
        # element is considered to ensure correct parsing for every reaction SMILES variation.
        reactants = reaction_smiles.split(">")[0].split(".")
        agents = reaction_smiles.split(">")[1].split(".")
        products = reaction_smiles.split(">")[2].split(" ")[0].split(".")

        # Return the original reaction role sub-strings including the reaction atom mappings.
        if as_what == "smiles":
            return [r_smiles for r_smiles in reactants if r_smiles != ""], \
                   [a_smiles for a_smiles in agents if a_smiles != ""], \
                   [p_smiles for p_smiles in products if p_smiles != ""]

        # Return the original reaction role sub-strings excluding the reaction atom mappings.
        elif as_what == "smiles_no_maps":
            return [re.sub(r":[-+]?[0-9]+", "", r_smiles) for r_smiles in reactants if r_smiles != ""], \
                   [re.sub(r":[-+]?[0-9]+", "", a_smiles) for a_smiles in agents if a_smiles != ""], \
                   [re.sub(r":[-+]?[0-9]+", "", p_smiles) for p_smiles in products if p_smiles != ""]

        # Return the lists of atom map numbers for each reaction role.
        elif as_what == "atom_maps":
            return [[int(r_atom_maps[1:]) for r_atom_maps in re.findall(r":[-+]?[0-9]+", r_smiles)]
                    for r_smiles in reactants if r_smiles != ""], \
                   [[int(a_atom_maps[1:]) for a_atom_maps in re.findall(r":[-+]?[0-9]+", a_smiles)]
                    for a_smiles in agents if a_smiles != ""], \
                   [[int(p_atom_maps[1:]) for p_atom_maps in re.findall(r":[-+]?[0-9]+", p_smiles)]
                    for p_smiles in products if products != ""]

        # Return the mol object or the canonical version of the SMILES string for each compound in the reaction roles.
        elif as_what in ["mol", "mol_no_maps", "canonical_smiles", "canonical_smiles_no_maps", "smarts",
                         "smarts_no_maps", "canonical_tuple", "canonical_tuple_no_maps"]:
            all_reaction_role_objects = []

            for reaction_role in [reactants, agents, products]:
                single_reaction_role_objects = []

                for rr_smiles in reaction_role:
                    if rr_smiles != "":
                        if as_what in ["mol", "canonical_smiles", "canonical_tuple"]:
                            mol_object = CompoundConversionUtils.string_to_mol(rr_smiles)
                        else:
                            mol_object = CompoundConversionUtils.string_to_mol(re.sub(r":[-+]?[0-9]+", "", rr_smiles))

                        if as_what in ["mol", "mol_no_maps"]:
                            single_reaction_role_objects.append(mol_object)
                            continue

                        if as_what in ["smarts", "smarts_no_maps"]:
                            single_reaction_role_objects.append(
                                CompoundConversionUtils.mol_to_string(mol_object, str_format="smarts"))
                            continue

                        canonical_smiles = CompoundConversionUtils.mol_to_string(mol_object, canonical=True)

                        if as_what in ["canonical_smiles", "canonical_smiles_no_maps"]:
                            single_reaction_role_objects.append(canonical_smiles)
                            continue

                        if as_what in ["canonical_tuple", "canonical_tuple_no_maps"]:
                            single_reaction_role_objects.append((canonical_smiles, mol_object))
                            continue

                all_reaction_role_objects.append(single_reaction_role_objects)

            return tuple(all_reaction_role_objects)

        # Raise exception for any other keyword.
        else:
            raise Exception("Unknown parsing type. Select one of the following: 'smiles', 'smiles_no_maps', "
                            "'atom_maps', 'mol', 'mol_no_maps', 'canonical_smiles', 'canonical_smiles_no_maps'.")


# noinspection PyArgumentList
class ReactionCoreUtils:
    """ Description: Group of methods for the handling of chemical reaction cores. """

    @staticmethod
    def __atom_in_core(mol_atom_ind: int, reaction_cores: Union[List, Tuple]) -> bool:
        """ Description: Check if a specific atom is in any of the lists of core atoms. """

        for reaction_core in reaction_cores:
            if mol_atom_ind in reaction_core:
                return True

        return False

    @staticmethod
    def __compound_is_mapped(compound: Union[str, Mol]) -> bool:
        """ Description: Check if a compound contains at least one mapped atom. """

        if isinstance(compound, str):
            return ":" in compound

        else:
            for atom in compound.GetAtoms():
                if atom.GetAtomMapNum() != 0:
                    return True

            return False

    @staticmethod
    def same_neighbourhood_size(compound_a: Mol, mol_atom_a: Union[Atom, int], compound_b: Mol,
                                mol_atom_b: Union[Atom, int]) -> bool:
        """ Description: Check whether the same atoms in two different molecules have the same neighbourhood size. """

        if isinstance(mol_atom_a, int):
            mol_atom_a = compound_a.GetAtomWithIdx(mol_atom_a)
        if isinstance(mol_atom_b, int):
            mol_atom_b = compound_b.GetAtomWithIdx(mol_atom_b)

        if len(mol_atom_a.GetNeighbors()) != len(mol_atom_b.GetNeighbors()):
            return False
        return True

    @staticmethod
    def same_neighbour_atoms(compound_a: Mol, mol_atom_a: Union[Atom, int], compound_b: Mol,
                             mol_atom_b: Union[Atom, int]) -> bool:
        """ Description: Check whether the same atoms in two different molecules have retained the same atoms and atom
                         attributes in their immediate neighbourhood according to reaction mapping numbers. """

        if isinstance(mol_atom_a, int):
            mol_atom_a = compound_a.GetAtomWithIdx(mol_atom_a)
        if isinstance(mol_atom_b, int):
            mol_atom_b = compound_b.GetAtomWithIdx(mol_atom_b)

        neighbourhood_a = [(mol_atom.GetAtomMapNum(), mol_atom.GetSymbol(), mol_atom.GetFormalCharge(),
                            mol_atom.GetNumRadicalElectrons(), mol_atom.GetTotalValence())
                           for mol_atom in mol_atom_a.GetNeighbors()]

        neighbourhood_b = [(mol_atom.GetAtomMapNum(), mol_atom.GetSymbol(), mol_atom.GetFormalCharge(),
                            mol_atom.GetNumRadicalElectrons(), mol_atom.GetTotalValence())
                           for mol_atom in mol_atom_b.GetNeighbors()]

        return sorted(neighbourhood_a) == sorted(neighbourhood_b)

    @staticmethod
    def same_neighbour_bonds(compound_a: Mol, mol_atom_a: Union[Atom, int], compound_b: Mol,
                             mol_atom_b: Union[Atom, int]) -> bool:
        """ Description: Check whether the same atoms in two different molecules have retained the same bonds and bond
                         attributes amongst each other in their immediate neighbourhood. """

        if isinstance(mol_atom_a, int):
            mol_atom_a_ind = mol_atom_a
            mol_atom_a = compound_a.GetAtomWithIdx(mol_atom_a)
        else:
            mol_atom_a_ind = mol_atom_a.GetIdx()

        if isinstance(mol_atom_b, int):
            mol_atom_b_ind = mol_atom_b
            mol_atom_b = compound_b.GetAtomWithIdx(mol_atom_b)
        else:
            mol_atom_b_ind = mol_atom_b.GetIdx()

        neighbourhood_1 = [(atom_ind.GetAtomMapNum(),
                            str(compound_a.GetBondBetweenAtoms(mol_atom_a_ind, atom_ind.GetIdx()).GetBondType()))
                           for atom_ind in mol_atom_a.GetNeighbors()]

        neighbourhood_2 = [(atom_ind.GetAtomMapNum(),
                            str(compound_b.GetBondBetweenAtoms(mol_atom_b_ind, atom_ind.GetIdx()).GetBondType()))
                           for atom_ind in mol_atom_b.GetNeighbors()]

        return sorted(neighbourhood_1) == sorted(neighbourhood_2)

    @staticmethod
    def get_reaction_core_atoms(reaction_smiles: str) -> Tuple[List, List]:
        """ Description: Get the indices of atoms that participate in the reaction for each molecule in the reaction.
                         If the molecule does not contain such atoms, return an empty list. This method is based on the
                         assumption that the mapping is correct and done in a 'complete' fashion. This means that only
                         atoms from the reactants that persists in the product are mapped. """

        reactants, _, products = ReactionConversionUtils.parse_roles_from_reaction_smiles(reaction_smiles,
                                                                                          as_what="mol")

        reactants_core_atoms = [set() for _ in range(len(reactants))]
        products_core_atoms = [set() for _ in range(len(products))]

        for p_ind, product in enumerate(products):
            # Only proceed to investigate products that are atom mapped.
            if ReactionCoreUtils.__compound_is_mapped(product):
                for r_ind, reactant in enumerate(reactants):
                    # Only proceed to investigate reactants that are atom mapped.
                    if ReactionCoreUtils.__compound_is_mapped(reactant):

                        for p_atom in product.GetAtoms():
                            # If there are atoms in the product that are not mapped, add them to the core.
                            if p_atom.GetAtomMapNum() <= 0:
                                products_core_atoms[p_ind].add(p_atom.GetIdx())
                                continue

                            for r_atom in reactant.GetAtoms():
                                # If there are atoms in the reactant that are not mapped, add them to the core.
                                if r_atom.GetAtomMapNum() <= 0:
                                    reactants_core_atoms[r_ind].add(r_atom.GetIdx())
                                    continue

                                # If there are atoms in the reactant and product that have the same atom map number,
                                # but different chemical surroundings, add them to the core.
                                if p_atom.GetAtomMapNum() == r_atom.GetAtomMapNum():
                                    if not ReactionCoreUtils.same_neighbourhood_size(product, p_atom.GetIdx(),
                                                                                     reactant, r_atom.GetIdx()) or \
                                       not ReactionCoreUtils.same_neighbour_atoms(product, p_atom.GetIdx(),
                                                                                  reactant, r_atom.GetIdx()) or \
                                       not ReactionCoreUtils.same_neighbour_bonds(product, p_atom.GetIdx(),
                                                                                  reactant, r_atom.GetIdx()):
                                        reactants_core_atoms[r_ind].add(r_atom.GetIdx())
                                        products_core_atoms[p_ind].add(p_atom.GetIdx())

        return reactants_core_atoms, products_core_atoms

    @staticmethod
    def get_reaction_non_core_atoms(reaction_smiles: str) -> Tuple[List, List]:
        """ Description: Get the atoms of the molecule which are not included in the specified reaction cores.
                         This method is just the inverse of the previous one. """

        reactants, _, products = ReactionConversionUtils.parse_roles_from_reaction_smiles(reaction_smiles,
                                                                                          as_what="mol")

        reactants_non_core_atoms = [set() for _ in range(len(reactants))]
        products_non_core_atoms = [set() for _ in range(len(products))]

        for p_ind, product in enumerate(products):
            for r_ind, reactant in enumerate(reactants):
                for p_atom in product.GetAtoms():

                    # If there are products that are not mapped, add all of their atoms to the non-core.
                    if not ReactionCoreUtils.__compound_is_mapped(product):
                        products_non_core_atoms[p_ind].add(p_atom.GetIdx())
                        continue

                    for r_atom in reactant.GetAtoms():

                        # If there are reactants that are not mapped, add all of their atoms to the non-core.
                        if not ReactionCoreUtils.__compound_is_mapped(reactant):
                            reactants_non_core_atoms[r_ind].add(r_atom.GetIdx())
                            continue

                        # If there are atoms in the reactant and product that have the same atom map number,
                        # and same chemical surroundings, add them to the core.
                        if p_atom.GetAtomMapNum() == r_atom.GetAtomMapNum():
                            if ReactionCoreUtils.same_neighbourhood_size(product, p_atom.GetIdx(),
                                                                         reactant, r_atom.GetIdx()) and \
                               ReactionCoreUtils.same_neighbour_atoms(product, p_atom.GetIdx(),
                                                                          reactant, r_atom.GetIdx()) and \
                               ReactionCoreUtils.same_neighbour_bonds(product, p_atom.GetIdx(),
                                                                          reactant, r_atom.GetIdx()):
                                reactants_non_core_atoms[r_ind].add(r_atom.GetIdx())
                                products_non_core_atoms[p_ind].add(p_atom.GetIdx())

        return reactants_non_core_atoms, products_non_core_atoms

    @staticmethod
    def get_inverse_atoms(reaction_smiles: str, marked_atoms: Tuple[List, List]) -> Tuple[List, List]:
        """ Description: Return the inverse from the marked atoms for each of the reaction roles. """

        reactants, _, products = ReactionConversionUtils.parse_roles_from_reaction_smiles(reaction_smiles,
                                                                                          as_what="mol_no_maps")
        reaction_roles = [reactants, products]
        reverse_cores = ([], [])

        for role_ind, reaction_role in enumerate(reaction_roles):
            for mol_ind, mol in enumerate(reaction_role):
                local_reverse = set()
                for atom in mol.GetAtoms():
                    if atom.GetIdx() not in marked_atoms[role_ind][mol_ind]:
                        local_reverse.add(atom.GetIdx())

                reverse_cores[role_ind].append(local_reverse)

        return reverse_cores

    @staticmethod
    def get_connected_core_indices_groups(reaction_smiles: str, reaction_cores: Tuple[List, List]):
        """ Description: Get the list of grouped reaction core indices. This grouping ads another layer in the standard
                         list of lists format. Ideally, the core indices for a single compound in the reaction should
                         all be connected, but sometimes this is not the case. This function can be called to check for
                         such multi-part cores, and to handle them appropriately. """

        reactants, _, products = ReactionConversionUtils.parse_roles_from_reaction_smiles(reaction_smiles,
                                                                                          as_what="mol")
        reaction_roles = [reactants, products]
        role_connections, connected_atoms, num_atoms = [[], []], [[], []], [[], []]

        # Step 1: Aggregate all of the atoms which are connected to each other.
        for rc_ind, reaction_core in enumerate(reaction_cores):
            for rr_ind, reaction_role in enumerate(reaction_core):
                atom_connections = []
                for ind_a, atom_a in enumerate(reaction_role):
                    for ind_b, atom_b in enumerate(reaction_role):
                        if ind_a != ind_b:
                            if reaction_roles[rc_ind][rr_ind].GetBondBetweenAtoms(atom_a, atom_b) is not None:
                                if [atom_a, atom_b] not in atom_connections and \
                                        [atom_b, atom_a] not in atom_connections:
                                    atom_connections.append([atom_a, atom_b])
                role_connections[rc_ind].append(atom_connections)

        # Step 2: Merge all of the individual connections which share the same indices.
        for rlc_ind, role_connection in enumerate(role_connections):
            [connected_atoms[rlc_ind].append(list(merge_common(rc))) for rc in role_connection]
            # [num_atoms[rlc_ind].append(len(ca)) for ca in connected_atoms[rlc_ind]]

        # for rc_ind, reaction_core in enumerate(reaction_cores):
        #    for rr_ind, reaction_role in enumerate(reaction_core):
        #        for atom in reaction_role:
        #            if not ReactionCoreUtils.__atom_in_core(atom, connected_atoms[rc_ind][rr_ind]):
        #                num_atoms[rc_ind][rr_ind] += 1

        final_connected_core_indices_groups = deepcopy(connected_atoms)

        # Step 3: Construct the final final connected core indices groups collection.
        for rc_ind, reaction_core in enumerate(reaction_cores):
            for rr_ind, reaction_role in enumerate(reaction_core):
                for atom in reaction_role:
                    if not ReactionCoreUtils.__atom_in_core(atom, connected_atoms[rc_ind][rr_ind]):
                        final_connected_core_indices_groups[rc_ind][rr_ind].append([atom])

        return final_connected_core_indices_groups


class ReactionAnalysisUtils:
    """ Description: Group of methods for the handling analysis of chemical reaction information. """

    @staticmethod
    def get_atoms_to_remove(editable_mol: RWMol):
        """ Description: Fetch all atoms that were marked for removal by the wildcard symbol '*'. """

        atoms_to_remove = []

        for atom in editable_mol.GetAtoms():
            if atom.GetSymbol() == "*":
                atoms_to_remove.append(atom.GetIdx())

        # Return the descending sorted list of atom indices to avoid errors during the removal of the atoms.
        return sorted(list(set(atoms_to_remove)), reverse=True)

    @staticmethod
    def get_bonds_to_remove(editable_mol: RWMol):
        """ Description: Fetch all bond atoms that were marked for removal by the wildcard symbol '*'. """

        bonds_to_remove = []

        for bond in editable_mol.GetBonds():
            if editable_mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol() == "*" and \
                    editable_mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol() == "*":
                bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

        # Return the descending sorted list of bond atom indices to avoid errors during the removal of the bond atoms.
        return sorted(list(set(bonds_to_remove)), key=lambda k: k[0], reverse=True)

    @staticmethod
    def remove_floating_atoms(editable_mol: RWMol):
        """ Description: Remove the wildcard atoms '*' that are disconnected from the rest of the compound atoms. """

        leftover_floating_atoms = sorted([atom.GetIdx() for atom in editable_mol.GetAtoms()
                                          if atom.GetSymbol() == "*" and atom.GetDegree() == 0], reverse=True)

        [editable_mol.RemoveAtom(rm_atom) for rm_atom in leftover_floating_atoms]

    @staticmethod
    def remove_marked_atoms(editable_mol: RWMol):
        """ Description: Remove all atoms that were marked for removal by the wildcard symbol '*'. """

        [editable_mol.RemoveAtom(rm_atom) for rm_atom in ReactionAnalysisUtils.get_atoms_to_remove(editable_mol)]

        # Sanitize the modified molecule.
        # SanitizeMol(editable_mol)

    @staticmethod
    def remove_marked_bonds(editable_mol: RWMol):
        """ Removes the all of the bonds that were marked for removal by the wildcard symbol '*'. """

        [editable_mol.RemoveBond(rm_bond[0], rm_bond[1])
         for rm_bond in ReactionAnalysisUtils.get_bonds_to_remove(editable_mol)]

        # Clean the editable mol from fully disconnected wildcard atoms '*'.
        ReactionAnalysisUtils.remove_floating_atoms(editable_mol)

        # Sanitize the modified molecule.
        # SanitizeMol(editable_mol)



    @staticmethod
    def extract_core_from_mol(compound: Union[str, Mol], reactive_atoms: List[int]):
        pass

    @staticmethod
    def extract_synthons_from_reactant(reactant_compound: Union[str, Mol], reactive_atoms: List[int]):
        pass

    @staticmethod
    def extract_synthons_from_product(product_compound: Union[str, Mol], reactive_atoms: List[int]):
        pass

    @staticmethod
    def generate_fragment_data(editable_mol: RWMol, reaction_side="product", basic_editable_mol=None):
        pass

    @staticmethod
    def extract_info_from_molecule(compound: Union[str, Mol], reactive_atoms: List[int], role="product"):
        pass

    @staticmethod
    def extract_info_from_reaction(reaction_smiles: str, reaction_cores=None):
        pass
