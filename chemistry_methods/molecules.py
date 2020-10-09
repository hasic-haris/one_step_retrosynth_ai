"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  December 18th, 2019
Description: This file contains necessary functions for the handling of molecules and their properties.
"""

import re
from rdkit.Chem import AllChem


def get_atom_environment(atom_indices, mol, degree=1):
    """ Returns the indices of all the n-degree neighbouring atoms for a single atom of the molecule. """

    # Check if the input molecule is given in SMILES or in the RDKit Mol format.
    if isinstance(mol, str):
        # Generate the RDKit Mol object from the input SMILES string.
        mol = AllChem.MolFromSmiles(mol)
        # Sanitize the molecule.
        AllChem.SanitizeMol(mol)

    # Input the known atoms in the final result and calculate a distance matrix for the molecule.
    neighbour_indices = [atom_ind for atom_ind in atom_indices]
    distance_matrix = AllChem.GetDistanceMatrix(mol)

    # Check the distances for all neighbours and add them if they are within the designated distance.
    for atom_ind in atom_indices:
        for ind, dist in enumerate(distance_matrix[atom_ind]):
            if dist <= degree:
                neighbour_indices.append(ind)

    # Return the set of the final neighbour indices list.
    return set(neighbour_indices)


def get_bond_environment(mol_bond, mol, degree=1):
    """ Returns the indices of all the n-degree neighbouring atoms for a single bond from the molecule. """

    # Check if the input molecule is given in SMILES or in the RDKit 'Mol' format.
    if isinstance(mol, str):
        # Generate the RDKit 'Mol' object from the input SMILES string.
        mol = AllChem.MolFromSmiles(mol)
        # Sanitize the molecule.
        AllChem.SanitizeMol(mol)

    # Return the set of the atom indices of both atoms of the bond.
    return set(get_atom_environment([mol_bond.GetBeginAtomIdx(), mol_bond.GetEndAtomIdx()], mol, degree))


def molecule_is_mapped(mol):
    """ Checks if a molecule created from a RDKit Mol object or a SMILES string contains at least one mapped atom."""

    # If it is a RDKit Mol object, check if any atom map number has a value other than zero.
    if not isinstance(mol, str):
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() != 0:
                return True
        return False

    # If it is a SMILES string, check if the string contains the symbol ":" used for mapping.
    else:
        return ":" in mol


def remove_mapping(mol):
    """ Removes any previous mapping from a molecule and returns a RDKit Mol object or a reaction SMILES string. """

    # If it is a RDKit Mol object, set all of the atom map number values to 0.
    if not isinstance(mol, str):
        mol = AllChem.MolToSmiles(mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return mol

    # If it is a SMILES string, delete every occurrence of the ':number' character combination used for mapping.
    else:
        return re.sub(r":[-+]?[0-9]+", "", mol)


def atoms_contain_complete_rings(mol, atom_indices):
    """ Checks if a set of marked atom indices contains complete aromatic rings. """

    all_rings_indices = mol.GetRingInfo().AtomRings()

    for ring_indices in all_rings_indices:
        if set(ring_indices).issubset(set(atom_indices)):
            return True

    return False


def count_atom_ring_memberships(mol, atom_ind):
    """ Counts the number of rings in which the atom with the index 'atom_ind' is a member of. """

    all_rings_indices = mol.GetRingInfo().AtomRings()
    member_of_rings = 0

    for ring_indices in all_rings_indices:
        if atom_ind in ring_indices:
            member_of_rings += 1

    return member_of_rings


def count_bond_ring_memberships(mol, begin_atom, end_atom):
    """ Counts the number of rings in which the bond between 'begin_atom' and 'end_atom' is a member of. """

    all_rings_indices = mol.GetRingInfo().AtomRings()
    member_of_rings = 0

    for ring_indices in all_rings_indices:
        if begin_atom in ring_indices and end_atom in ring_indices:
            member_of_rings += 1

    return member_of_rings


def fetch_rest_of_ring_atoms(mol, atom_indices):
    """ Fetches the rest of the aromatic ring atoms for all aromatic rings atoms contained in 'atom_indices'. """

    all_rings_indices = mol.GetRingInfo().AtomRings()
    new_atom_indices = [atom_ind for atom_ind in atom_indices]

    while True:
        # Detect whether some of the atom indices are members of a ring and add the rest of the ring.
        new_additions = []
        for atom_ind in new_atom_indices:
            for ring_ind in all_rings_indices:
                if atom_ind in ring_ind and not set(ring_ind).issubset(new_atom_indices):
                    new_additions.extend(ring_ind)

        # If there are no detected rings that are not already in the expanded core, break the loop.
        if len(new_additions) == 0:
            break
        else:
            new_atom_indices.extend(list(set(new_additions)))

    # Return the sorted list of atom indices.
    return sorted(list(set(new_atom_indices)), reverse=True)
