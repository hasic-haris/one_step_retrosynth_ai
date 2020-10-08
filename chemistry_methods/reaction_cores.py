"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  December 17th, 2019
Description: This file contains functions for finding and extracting reaction cores.
"""

from copy import deepcopy
from chemistry_methods.reactions import parse_reaction_roles
from chemistry_methods.molecules import molecule_is_mapped
from data_methods.helpers import merge_common


def same_neighbourhood_size(atom_index_1, molecule_1, atom_index_2, molecule_2):
    """ Checks whether the same atoms in two different molecules (e.g., reactant and product molecules) have the same
        neighbourhood size. """

    if len(molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors()) != \
            len(molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors()):
        return False
    return True


def same_neighbour_atoms(atom_index_1, molecule_1, atom_index_2, molecule_2):
    """ Checks whether the same atoms in two different molecules (e.g., reactant and product molecules) have retained
        the same types of chemical elements in their immediate neighbourhood according to reaction mapping numbers. """

    neighbourhood_1, neighbourhood_2 = [], []

    for i in molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors():
        neighbourhood_1.append((i.GetAtomMapNum(), i.GetSymbol(), i.GetFormalCharge(),
                                i.GetNumRadicalElectrons(), i.GetTotalValence()))
    for j in molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors():
        neighbourhood_2.append((j.GetAtomMapNum(), j.GetSymbol(), j.GetFormalCharge(),
                                j.GetNumRadicalElectrons(), j.GetTotalValence()))

    return sorted(neighbourhood_1) == sorted(neighbourhood_2)


def same_neighbour_bonds(atom_index_1, molecule_1, atom_index_2, molecule_2):
    """ Checks whether the same atoms in two different molecules (e.g., reactant and product molecules) have retained
        the same types of chemical bonds amongst each other in their immediate neighbourhood. """

    neighbourhood_1, neighbourhood_2 = [], []

    for i in molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors():
        neighbourhood_1.append((i.GetAtomMapNum(),
                                str(molecule_1.GetBondBetweenAtoms(atom_index_1, i.GetIdx()).GetBondType())))
    for j in molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors():
        neighbourhood_2.append((j.GetAtomMapNum(),
                                str(molecule_2.GetBondBetweenAtoms(atom_index_2, j.GetIdx()).GetBondType())))

    return sorted(neighbourhood_1) == sorted(neighbourhood_2)


def get_reaction_core_atoms(rsmiles):
    """ Returns the indices of atoms that participate in the reaction for each molecule in the reaction. If the molecule
        does not contain such atoms, return an empty list.
        NOTE: This method is based on the assumption that the reaction mapping is correct and done by matching the same
        atoms in the reactants and products. """

    reactants, _, products = parse_reaction_roles(rsmiles, as_what="mol")
    reactants_final = [set() for _ in range(len(reactants))]
    products_final = [set() for _ in range(len(products))]

    for p_ind, product in enumerate(products):
        for r_ind, reactant in enumerate(reactants):
            for p_atom in product.GetAtoms():
                if p_atom.GetAtomMapNum() <= 0:
                    products_final[p_ind].add(p_atom.GetIdx())
                    continue
                for r_atom in reactant.GetAtoms():
                    if molecule_is_mapped(reactant) and r_atom.GetAtomMapNum() <= 0:
                        reactants_final[r_ind].add(r_atom.GetIdx())
                        continue
                    if p_atom.GetAtomMapNum() == r_atom.GetAtomMapNum():
                        if not same_neighbourhood_size(p_atom.GetIdx(), product, r_atom.GetIdx(), reactant) or \
                                not same_neighbour_atoms(p_atom.GetIdx(), product, r_atom.GetIdx(), reactant) or \
                                not same_neighbour_bonds(p_atom.GetIdx(), product, r_atom.GetIdx(), reactant):
                            reactants_final[r_ind].add(r_atom.GetIdx())
                            products_final[p_ind].add(p_atom.GetIdx())

    return reactants_final, products_final


def atom_in_core(atom, separated_cores):
    """ Checks if an atom is in any of the lists of atoms. """

    for core in separated_cores:
        if atom in core:
            return True
    return False


def get_non_reaction_core_atoms(rsmiles, cores):
    """ Returns the atoms of the molecule which are not included in the specified reaction cores. """

    reactants, _, products = parse_reaction_roles(rsmiles, as_what="mol_no_maps")
    roles = [reactants, products]
    reverse_cores = ([], [])

    for role_ind, role in enumerate(roles):
        for mol_ind, mol in enumerate(role):
            local_reverse = set()
            for atom in mol.GetAtoms():
                if atom.GetIdx() not in cores[role_ind][mol_ind]:
                    local_reverse.add(atom.GetIdx())
            reverse_cores[role_ind].append(local_reverse)

    return reverse_cores


def get_separated_cores(rsmiles, cores):
    """ Returns the separated cores among the core atoms marked by the mapping. """

    reactants, _, products = parse_reaction_roles(rsmiles, as_what="mol")
    roles = [reactants, products]
    role_connections, connected_atoms, num_atoms = [[], []], [[], []], [[], []]

    for c_ind, core in enumerate(cores):
        for r_ind, role in enumerate(core):
            connections = []
            for ind1, atom1 in enumerate(role):
                for ind2, atom2 in enumerate(role):
                    if ind1 != ind2:
                        if roles[c_ind][r_ind].GetBondBetweenAtoms(atom1, atom2) is not None:
                            if [atom1, atom2] not in connections and [atom2, atom1] not in connections:
                                connections.append([atom1, atom2])
            role_connections[c_ind].append(connections)

    for r_ind, role in enumerate(role_connections):
        [connected_atoms[r_ind].append(list(merge_common(r))) for r in role]
        [num_atoms[r_ind].append(len(ca)) for ca in connected_atoms[r_ind]]

    for c_ind, core in enumerate(cores):
        for r_ind, role in enumerate(core):
            for atom in role:
                if not atom_in_core(atom, connected_atoms[c_ind][r_ind]):
                    num_atoms[c_ind][r_ind] += 1

    final_separated_cores = deepcopy(connected_atoms)

    for c_ind, core in enumerate(cores):
        for r_ind, role in enumerate(core):
            for atom in role:
                if not atom_in_core(atom, connected_atoms[c_ind][r_ind]):
                    final_separated_cores[c_ind][r_ind].append([atom])

    return final_separated_cores
