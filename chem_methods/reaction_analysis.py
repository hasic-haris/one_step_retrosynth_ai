"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  March 10th, 2020
Description: This file contains necessary functions for the analysis of chemical reactions and extraction of reactive
             and non-reactive substructures from reactant and product molecules.
"""

from copy import deepcopy

from rdkit.Chem import AllChem

from chem_methods.reactions import parse_reaction_roles
from chem_methods.reaction_cores import get_reaction_core_atoms
from chem_methods.molecules import fetch_rest_of_ring_atoms, count_atom_ring_memberships, atoms_contain_complete_rings
from chem_methods.editable_molecules import remove_marked_bonds


def extract_core_from_mol(mol, reactive_atoms):
    """ Marks and removes non-core atoms from the reactant molecules and returns the reactive part of the reactants or
        the isolated reaction core for the products. """

    # Create an editable RDKit RWMol object and create a backup copy of it for later use.
    editable_mol = AllChem.RWMol(mol)
    default_editable_mol = deepcopy(editable_mol)

    # If the indices are not correct or an empty array, return the starting molecule.
    if len(reactive_atoms) == len(mol.GetAtoms()) or len(reactive_atoms) == 0:
        return editable_mol, default_editable_mol

    # Create the list of atoms that are not part of the core and sort them in DESC order to avoid removal conflicts.
    nr_atoms = sorted([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in reactive_atoms], reverse=True)

    # First, try to remove molecule atoms that are marked as non-reactive.
    try:
        for rm_atom in nr_atoms:
            editable_mol.ReplaceAtom(rm_atom, AllChem.Atom("*"))

        # Save the state with the wildcard atoms for later use in fragmentation of reactants.
        basic_editable_mol = deepcopy(editable_mol)
        # Remove all bonds that were replaced with the wildcard atom '*'.
        remove_marked_bonds(editable_mol)

    # If this fails, usually due to sanitization errors, remove molecule atoms according to the expanded core.Some cores
    # are complete aromatic rings or fused rings which cannot be broken and converted to a RDKit Mol object. Expand the
    # core indices to include all rings or fused rings, and generate a list of atoms that are not part of this core.
    except:
        # Generate the expanded core.
        expanded_core = fetch_rest_of_ring_atoms(mol, reactive_atoms)
        expanded_synthon_atoms = sorted([atom.GetIdx() for atom in mol.GetAtoms()
                                         if atom.GetIdx() not in expanded_core], reverse=True)

        # Create a new copy of the molecule because the previous one may have been modified.
        editable_mol = deepcopy(default_editable_mol)

        # To prevent later sanitization errors due to incorrect valences, skip any isolated atoms connected to the core.
        skip_atoms = []
        for bond in editable_mol.GetBonds():
            if bond.GetBeginAtomIdx() in expanded_core and bond.GetEndAtomIdx() not in expanded_core and \
                    editable_mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetDegree() == 1:
                skip_atoms.append(bond.GetEndAtomIdx())
            if bond.GetEndAtomIdx() in expanded_core and bond.GetBeginAtomIdx() not in expanded_core and \
                    editable_mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetDegree() == 1:
                skip_atoms.append(bond.GetBeginAtomIdx())

        # Replace all of the atoms with the wildcard atom symbol '*'.
        for atom in set(expanded_synthon_atoms).difference(set(skip_atoms)):
            editable_mol.ReplaceAtom(atom, AllChem.Atom("*"))

        # Save the state with the wildcard atoms for later use in fragmentation of reactants.
        basic_editable_mol = deepcopy(editable_mol)
        # Remove all bonds that were replaced with the wildcard atom '*'.
        remove_marked_bonds(editable_mol)

    # Return the editable RDKit RWMol object after the changes have been made.
    return editable_mol, basic_editable_mol


def extract_synthons_from_reactant(reactant_mol, reactive_atoms):
    """ Marks and removes marked reactive atoms from the reactant molecules and returns only the non-reactive part. """

    # Create an editable RDKit RWMol object and create a backup copy of it for later use.
    editable_mol = AllChem.RWMol(reactant_mol)
    default_editable_mol = deepcopy(editable_mol)

    # If the indices are not correct or an empty array, return the starting molecule.
    if len(reactive_atoms) == len(reactant_mol.GetAtoms()) or len(reactive_atoms) == 0:
        return editable_mol, default_editable_mol

    # First, try to just remove all of the core atoms without any further modifications.
    try:
        for rm_atom in reactive_atoms:
            editable_mol.ReplaceAtom(rm_atom, AllChem.Atom("*"))

        # Save the state with the wildcard atoms for later use in fragmentation of reactants.
        basic_editable_mol = deepcopy(editable_mol)
        # Remove all bonds that were replaced with the wildcard atom '*'.
        remove_marked_bonds(editable_mol)

    # If that fails, it's most likely due to the incorrect decomposition of aromatic rings.
    except:
        # Create a new copy of the molecule because the previous one may have been modified.
        editable_mol = deepcopy(default_editable_mol)

        # Try removing all atoms in the present aromatic ring, except atoms shared between fused rings.
        try:
            for bond in editable_mol.GetBonds():
                if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() in reactive_atoms:
                    if count_atom_ring_memberships(editable_mol, bond.GetBeginAtomIdx()) < 2:
                        editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                    if count_atom_ring_memberships(editable_mol, bond.GetEndAtomIdx()) < 2:
                        editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))

            # Save the state with the wildcard atoms for later use in fragmentation of reactants.
            basic_editable_mol = deepcopy(editable_mol)
            # Remove all bonds that were replaced with the wildcard atom '*'.
            remove_marked_bonds(editable_mol)

        # If this also fails, only remove non-aromatic bond atoms attached to the ring if there are any.
        except:
            # Create a new copy of the molecule because the previous one may have been modified.
            editable_mol = deepcopy(default_editable_mol)

            for bond in editable_mol.GetBonds():
                if str(bond.GetBondType()) != "AROMATIC":
                    if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() not in reactive_atoms:
                        editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                        editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))
                    elif bond.GetEndAtomIdx() in reactive_atoms and bond.GetBeginAtomIdx() not in reactive_atoms:
                        editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                        editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))

            # Save the state with the wildcard atoms for later use in fragmentation of reactants.
            basic_editable_mol = deepcopy(editable_mol)
            # Remove all bonds that were replaced with the wildcard atom '*'.
            remove_marked_bonds(editable_mol)

    # Return the editable RDKit RWMol object after the changes have been made.
    return editable_mol, basic_editable_mol


def extract_synthons_from_product(product_mol, reactive_atoms):
    """  Marks and removes the reactive atoms from the product molecules and returns only synthon templates. """

    # Create an editable RDKit RWMol object and create a backup copy of it for later use.
    editable_mol = AllChem.RWMol(product_mol)
    default_editable_mol = deepcopy(editable_mol)

    # If the indices are not correct or an empty array, return the starting molecule.
    if len(reactive_atoms) == len(product_mol.GetAtoms()) or len(reactive_atoms) == 0:
        return editable_mol

    # First check if all of the core bonds are aromatic.
    if all([str(bond.GetBondType()) == "AROMATIC" for bond in editable_mol.GetBonds()
            if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() in reactive_atoms]):
        # ------------------------------------------------
        # Reactive atoms CONTAIN ONLY FULL AROMATIC RINGS.
        # ------------------------------------------------
        if atoms_contain_complete_rings(editable_mol, reactive_atoms):
            # Try removing all aromatic atoms in the core ring, except atoms shared between fused rings.
            try:
                for bond in editable_mol.GetBonds():
                    if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() in reactive_atoms:
                        if count_atom_ring_memberships(editable_mol, bond.GetBeginAtomIdx()) < 2:
                            editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                        if count_atom_ring_memberships(editable_mol, bond.GetEndAtomIdx()) < 2:
                            editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))

                # Remove all bonds that were replaced with the wildcard atom '*'.
                remove_marked_bonds(editable_mol)

            # If this fails, it's mostly due to kekulization issues. Remove all non-aromatic atoms attached to the ring.
            except:
                try:
                    # Create a new copy of the molecule because the previous one may have been modified.
                    editable_mol = deepcopy(default_editable_mol)

                    for bond in editable_mol.GetBonds():
                        if str(bond.GetBondType()) != "AROMATIC":
                            if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() not in reactive_atoms:
                                editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                                editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))
                            elif bond.GetEndAtomIdx() in reactive_atoms and \
                                    bond.GetBeginAtomIdx() not in reactive_atoms:
                                editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                                editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))

                    # Remove all bonds that were replaced with the wildcard atom '*'.
                    remove_marked_bonds(editable_mol)

                # If that also fails, completely remove all aromatic atoms from the ring.
                except:
                    # Create a new copy of the molecule because the previous one may have been modified.
                    editable_mol = deepcopy(default_editable_mol)

                    for bond in editable_mol.GetBonds():
                        if str(bond.GetBondType()) == "AROMATIC":
                            if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() in reactive_atoms:
                                editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                                editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))

                    # Remove all bonds that were replaced with the wildcard atom '*'.
                    remove_marked_bonds(editable_mol)

        # ------------------------------------------------------------------
        # Reactive atoms CONTAIN ALL AROMATIC BONDS, BUT NOT COMPLETE RINGS.
        # ------------------------------------------------------------------
        else:
            # Try removing all of the atoms normally.
            try:
                for rm_atom in reactive_atoms:
                    editable_mol.ReplaceAtom(rm_atom, AllChem.Atom("*"))

                # Remove all bonds that were replaced with the wildcard atom '*'.
                remove_marked_bonds(editable_mol)

            # If this fails, mark and remove only the nearest non-aromatic bond connected to the aromatic part.
            except:
                # Create a new copy of the molecule because the previous one may have been modified.
                editable_mol = deepcopy(default_editable_mol)

                for bond in editable_mol.GetBonds():
                    if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() in reactive_atoms:
                        if str(bond.GetBondType()) != "AROMATIC":
                            editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                            editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))

                # Remove all bonds that were replaced with the wildcard atom '*'.
                remove_marked_bonds(editable_mol)

    # -----------------------------------------
    # Reactive atoms CONTAIN NO AROMATIC BONDS.
    # -----------------------------------------
    elif not any([str(bond.GetBondType()) == "AROMATIC" for bond in editable_mol.GetBonds()
                  if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() in reactive_atoms]):
        # Try removing all of the atoms normally. This should work for all cases.
        for rm_atom in reactive_atoms:
            editable_mol.ReplaceAtom(rm_atom, AllChem.Atom("*"))

        # Remove all bonds that were replaced with the wildcard atom '*'.
        remove_marked_bonds(editable_mol)

    # -------------------------------------------------------------
    # Reactive atoms CONTAIN COMPLETE RINGS AND NON-AROMATIC BONDS.
    # -------------------------------------------------------------
    else:
        if atoms_contain_complete_rings(editable_mol, reactive_atoms):
            # Try removing all aromatic atoms in the core ring, except atoms shared between fused rings.
            try:
                for bond in editable_mol.GetBonds():
                    if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() in reactive_atoms:
                        if count_atom_ring_memberships(editable_mol, bond.GetBeginAtomIdx()) < 2:
                            editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                        if count_atom_ring_memberships(editable_mol, bond.GetEndAtomIdx()) < 2:
                            editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))

                # Remove all bonds that were replaced with the wildcard atom '*'.
                remove_marked_bonds(editable_mol)

            # If this fails, it's mostly due to kekulization issues. Remove all non-aromatic atoms attached to the ring.
            except:
                # Create a new copy of the molecule because the previous one may have been modified.
                editable_mol = deepcopy(default_editable_mol)

                for bond in editable_mol.GetBonds():
                    if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() in reactive_atoms:
                        if str(bond.GetBondType()) != "AROMATIC":
                            editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                            editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))

                # Sanitize the molecule at this point because some of these entries require it here.
                AllChem.SanitizeMol(editable_mol)
                # Remove all bonds that were replaced with the wildcard atom '*'.
                remove_marked_bonds(editable_mol)

        # ------------------------------------------------------------------------------
        # Reactive atoms CONTAIN AROMATIC AND NON-AROMATIC BONDS, BUT NO COMPLETE RINGS.
        # ------------------------------------------------------------------------------
        else:
            try:
                # Try removing all of the atoms normally. This should work for all cases.
                for rm_atom in reactive_atoms:
                    editable_mol.ReplaceAtom(rm_atom, AllChem.Atom("*"))

                # Remove all bonds that were replaced with the wildcard atom '*'.
                remove_marked_bonds(editable_mol)

            # If this fails, remove only the non-aromatic bonds.
            except:
                # Create a new copy of the molecule because the previous one may have been modified.
                editable_mol = deepcopy(default_editable_mol)

                for bond in editable_mol.GetBonds():
                    if bond.GetBeginAtomIdx() in reactive_atoms and bond.GetEndAtomIdx() in reactive_atoms:
                        if str(bond.GetBondType()) != "AROMATIC":
                            editable_mol.ReplaceAtom(bond.GetBeginAtomIdx(), AllChem.Atom("*"))
                            editable_mol.ReplaceAtom(bond.GetEndAtomIdx(), AllChem.Atom("*"))

                # Remove all bonds that were replaced with the wildcard atom '*'.
                remove_marked_bonds(editable_mol)

    # Return the editable RDKit RWMol object after the changes have been made.
    return editable_mol


def generate_fragment_data(editable_mol, reaction_side="product", basic_editable_mol=None):
    """ Generates and returns various formats of the fragmented molecules. """

    # Generate a copy of the molecules to work with and sanitize them.
    focus_mol = deepcopy(editable_mol)
    AllChem.SanitizeMol(focus_mol)
    smiles = AllChem.MolToSmiles(focus_mol)
    mol_from_smiles = AllChem.MolFromSmiles(smiles)
    AllChem.SanitizeMol(mol_from_smiles)

    if reaction_side == "reactant":
        # If the editable reactant molecule is split into multiple parts, fall back to the default molecule because for
        # the reactant molecules, just the reactive part needs to be marked.
        if "." in smiles:
            focus_mol = deepcopy(basic_editable_mol)
            AllChem.SanitizeMol(focus_mol)
            smiles = AllChem.MolToSmiles(focus_mol)
            mol_from_smiles = AllChem.MolFromSmiles(smiles)
            AllChem.SanitizeMol(mol_from_smiles)

        smarts = AllChem.MolToSmarts(mol_from_smiles)

        try:
            mol_from_smarts = AllChem.MolFromSmarts(smiles)
            AllChem.SanitizeMol(mol_from_smarts)
        except:
            mol_from_smarts = deepcopy(mol_from_smiles)

        # Return the generated data in various formats.
        return smiles, smarts, mol_from_smiles, mol_from_smarts

    elif reaction_side == "product":
        all_frag_smiles, all_frag_smarts, all_frag_smiles_mols, all_frag_smarts_mols = [], [], [], []

        for frag_smi in smiles.split("."):
            focus_mol = AllChem.MolFromSmiles(frag_smi)
            AllChem.SanitizeMol(focus_mol)

            all_frag_smiles_mols.append(focus_mol)
            all_frag_smiles.append(AllChem.MolToSmiles(focus_mol))
            all_frag_smarts.append(AllChem.MolToSmarts(focus_mol))

            try:
                mol_from_smarts = AllChem.MolFromSmarts(frag_smi)
                AllChem.SanitizeMol(mol_from_smarts)
                all_frag_smarts_mols.append(mol_from_smarts)
            except:
                mol_from_smarts = deepcopy(focus_mol)
                all_frag_smarts_mols.append(mol_from_smarts)
                continue

        # Sort the fragments based on the number of atoms in the molecule.
        all_frag_smiles_mols, all_frag_smiles, all_frag_smarts, all_frag_smarts_mols = \
            zip(*sorted(zip(all_frag_smiles_mols, all_frag_smiles, all_frag_smarts, all_frag_smarts_mols),
                        key=lambda k: len(k[0].GetAtoms()), reverse=True))

        # Return the generated data in various formats.
        return list(all_frag_smiles), list(all_frag_smarts), list(all_frag_smiles_mols), list(all_frag_smarts_mols)
    else:
        raise Exception("The only acceptable keywords are 'reactant' and 'product'.")


def extract_info_from_molecule(mol, reactive_atoms, role="product"):
    """ Extract the reactive and non-reactive parts of the reactant and product molecules from the molecule. """

    # Check if the input molecule is given in SMILES or in the RDKit 'Mol' format.
    if isinstance(mol, str):
        # Generate the RDKit 'Mol' object from the input SMILES string.
        mol = AllChem.MolFromSmiles(mol)
        # Sanitize the molecule.
        AllChem.SanitizeMol(mol)

    reactive_atoms = sorted(reactive_atoms, reverse=True)

    if role == "reactant":
        rw_mol, basic_rw_mol = extract_core_from_mol(mol, reactive_atoms)
        reactive_part = generate_fragment_data(rw_mol, reaction_side="reactant", basic_editable_mol=basic_rw_mol)

        rw_mol, basic_rw_mol = extract_synthons_from_reactant(mol, reactive_atoms)
        non_reactive_part = generate_fragment_data(rw_mol, reaction_side="reactant", basic_editable_mol=basic_rw_mol)

        return reactive_part, non_reactive_part
    else:
        rw_mol, _ = extract_core_from_mol(mol, reactive_atoms)
        reactive_part = generate_fragment_data(rw_mol)

        rw_mol = extract_synthons_from_product(mol, reactive_atoms)
        non_reactive_part = generate_fragment_data(rw_mol)

        return reactive_part, non_reactive_part


def extract_info_from_reaction(reaction_smiles, reaction_cores=None):
    """ Extract the reactive and non-reactive parts of the reactant and product molecules from the reaction. """

    reactant_fragments, product_fragments = [], []

    # Extract the reactants and products as RDKit Mol objects and find the reaction cores if none are specified.
    reactants, _, products = parse_reaction_roles(reaction_smiles, as_what="mol_no_maps")

    if reaction_cores is None:
        reaction_cores = get_reaction_core_atoms(reaction_smiles)

    # Extraction of information from the reactant molecules.
    for r_ind, reactant in enumerate(reactants):
        # Sanitize the focus molecule.
        AllChem.SanitizeMol(reactant)
        # Sort the core atom indices in descending order to avoid removal conflicts.
        reactive_atoms = sorted(reaction_cores[0][r_ind], reverse=True)

        # Mark and remove all of the atoms which are not in the reaction core.
        rw_mol, basic_rw_mol = extract_core_from_mol(reactant, reactive_atoms)

        # Clean and convert the extracted core candidates to different data formats.
        reactive_part = generate_fragment_data(rw_mol, reaction_side="reactant", basic_editable_mol=basic_rw_mol)

        # Mark and remove all of the atoms from the reaction core.
        rw_mol, basic_rw_mol = extract_synthons_from_reactant(reactant, reactive_atoms)

        # Clean and convert the extracted core candidates to different data formats.
        non_reactive_part = generate_fragment_data(rw_mol, reaction_side="reactant", basic_editable_mol=basic_rw_mol)

        reactant_fragments.append((reactive_part, non_reactive_part))

    # Extraction of information from the product molecules.
    for p_ind, product in enumerate(products):
        # Sanitize the focus molecule.
        AllChem.SanitizeMol(product)
        # Sort the core atom indices in DESC order to avoid removal conflicts.
        reactive_atoms = sorted(reaction_cores[1][p_ind], reverse=True)

        # Mark and remove all of the atoms which are not in the reaction core.
        rw_mol, _ = extract_core_from_mol(product, reactive_atoms)

        # Clean and convert the extracted core candidates to different data formats.
        reactive_part = generate_fragment_data(rw_mol)

        # Mark and remove all of the atoms from the reaction core.
        rw_mol = extract_synthons_from_product(product, reactive_atoms)

        # Clean and convert the extracted synthon candidates to different data formats.
        non_reactive_part = generate_fragment_data(rw_mol)

        product_fragments.append((reactive_part, non_reactive_part))

    # Return all of the generated data for a single chemical reaction.
    return reactant_fragments, product_fragments
