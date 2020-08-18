"""
Author:      Hasic Haris (Phd Student @ Ishida Lab, Department of Computer Science, Tokyo Institute of Technology)
Created on:  March 10th, 2020
Description: This file contains functions for handling RDKit editable 'RWMol' molecule objects.
"""
from rdkit.Chem import AllChem


# Done: 100%
def get_atoms_to_remove(editable_mol):
    """ Fetches all of the bonds that were marked for removal by the wildcard symbol '*'. """

    atoms_to_remove = []

    for atom in editable_mol.GetAtoms():
        if atom.GetSymbol() == "*":
            atoms_to_remove.append(atom.GetIdx())

    # Return the descending sorted list of atom indices to avoid errors during the removal of the atoms.
    return sorted(list(set(atoms_to_remove)), reverse=True)


# Done: 100%
def get_bonds_to_remove(editable_mol):
    """ Fetches all of the bonds that were marked for removal by the wildcard symbol '*'. """
    bonds_to_remove = []

    for bond in editable_mol.GetBonds():
        if editable_mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol() == "*" and \
                editable_mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol() == "*":
            bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    # Return the descending sorted list of bond atom indices to avoid errors during the removal of the bond atoms.
    return sorted(list(set(bonds_to_remove)), key=lambda k: k[0], reverse=True)


# Done: 100%
def remove_floating_atoms(editable_mol):
    """ Removes the leftover wildcard atoms '*' that are fully disconnected from the rest of the molecule. """

    leftover_floating_atoms = sorted([atom.GetIdx() for atom in editable_mol.GetAtoms()
                                      if atom.GetSymbol() == "*" and atom.GetDegree() == 0], reverse=True)

    [editable_mol.RemoveAtom(rm_atom) for rm_atom in leftover_floating_atoms]


# Done: 100%
def remove_marked_atoms(editable_mol):
    """ Removes the all of the bonds that were marked for removal by the wildcard symbol '*'. """

    [editable_mol.RemoveAtom(rm_atom) for rm_atom in get_atoms_to_remove(editable_mol)]
    # Sanitize the modified molecule.
    AllChem.SanitizeMol(editable_mol)


# Done: 100%
def remove_marked_bonds(editable_mol):
    """ Removes the all of the bonds that were marked for removal by the wildcard symbol '*'. """

    [editable_mol.RemoveBond(rm_bond[0], rm_bond[1]) for rm_bond in get_bonds_to_remove(editable_mol)]
    # Clean the editable mol from fully disconnected wildcard atoms '*'.
    remove_floating_atoms(editable_mol)
    # Sanitize the modified molecule.
    AllChem.SanitizeMol(editable_mol)
