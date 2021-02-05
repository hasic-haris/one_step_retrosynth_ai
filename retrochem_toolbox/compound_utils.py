"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  December 18th, 2019
Edited on:   January 31st, 2021
"""

from typing import List, Set, Tuple, Union

from rdkit.Chem.AllChem import Atom, Bond, Mol
from rdkit.Chem.AllChem import MolFromSmiles, MolFromSmarts, MolToSmiles, MolToSmarts, SanitizeMol, GetDistanceMatrix
from rdkit.Chem.SaltRemover import SaltRemover


class CompoundConversionUtils:
    """ Description: Group of methods for the handling of chemical compound representation conversions. """

    @staticmethod
    def string_to_mol(compound_string: str, str_format="smiles", verbose=True) -> Union[Mol, None]:
        """ Description: Convert a chemical compound string representation to a Mol object. """

        mol_object = None

        try:
            if str_format == "smiles":
                mol_object = MolFromSmiles(compound_string)
            elif str_format == "smarts":
                mol_object = MolFromSmarts(compound_string)
            else:
                raise Exception("Supported string formats are 'smiles' and 'smarts'. Got: '{}'.".format(str_format))

            SanitizeMol(mol_object)

            return mol_object

        except Exception as exc_msg:
            if verbose:
                if mol_object is None:
                    print("Exception occurred during the conversion of ", end="")
                else:
                    print("Exception occurred during the sanitization of ", end="")

                print("'{}'. Detailed message:\n{}".format(compound_string, exc_msg))

            return None

    @staticmethod
    def mol_to_string(mol_object: Mol, str_format="smiles", canonical=True, verbose=False) -> Union[str, None]:
        """ Description: Convert a chemical compound Mol object to a string representation. """

        try:
            if str_format == "smiles":
                return MolToSmiles(mol_object, canonical=canonical)
            elif str_format == "smarts":
                return MolToSmarts(mol_object)
            else:
                raise Exception("Supported string formats are 'smiles' and 'smarts'. Got: '{}'.".format(str_format))

        except Exception as exc_msg:
            if verbose:
                print("Exception occurred during the conversion of the Mol object. "
                      "Detailed message: {}".format(exc_msg))

            return None

    @staticmethod
    def string_to_canonical_string(compound_string: str, str_format="smiles", verbose=True) -> Union[str, None]:
        """ Description: Canonicalize a chemical compound string representation. """

        return CompoundConversionUtils.mol_to_string(
            CompoundConversionUtils.string_to_mol(compound_string, str_format=str_format, verbose=verbose),
            str_format=str_format, canonical=True, verbose=verbose)


# noinspection PyArgumentList
class CompoundStructureUtils:
    """ Description: Group of methods for the handling of chemical compound structures. """

    @staticmethod
    def get_atom_environment(compound: Union[str, Mol],
                             mol_atoms: Union[List[Atom], Tuple[Atom], List[int], Set[int], Tuple[int]],
                             n_degree=1) -> Set:
        """ Description: Get the indices of all 'n_degree' neighbouring atoms for a collection of Mol atoms or Mol atom
                         indices. The 'compound' parameter can be either a SMILES string or a Mol object. """

        if isinstance(compound, str):
            compound = CompoundConversionUtils.string_to_mol(compound)

        mol_atom_indices = []

        for mol_atom in mol_atoms:
            if isinstance(mol_atom, Atom):
                mol_atom_indices.append(mol_atom.GetIdx())
            else:
                mol_atom_indices.append(mol_atom)

        # Input the known atoms in the final result and calculate a distance matrix for the molecule.
        neighbour_indices = [atom_ind for atom_ind in mol_atom_indices]
        distance_matrix = GetDistanceMatrix(compound)

        # Check the distances for all neighbours and add them if they are within the designated distance.
        for atom_ind in mol_atom_indices:
            for ind, dist in enumerate(distance_matrix[atom_ind]):
                if dist <= n_degree:
                    neighbour_indices.append(ind)

        return set(neighbour_indices)

    @staticmethod
    def get_bond_environment(compound: Union[str, Mol],
                             mol_bonds: Union[List[Bond], Tuple[Bond], List[int], Set[int], Tuple[int]],
                             n_degree=1) -> Set:
        """ Description: Get the indices of all 'n_degree' neighbouring atoms for a collection of Mol bonds or Mol bond
                         indices. The 'compound' parameter can be either a SMILES string or a Mol object. """

        if isinstance(compound, str):
            compound = CompoundConversionUtils.string_to_mol(compound)

        all_mol_bonds_atom_tuples = []

        for mol_bond in mol_bonds:
            if isinstance(mol_bond, Bond):
                all_mol_bonds_atom_tuples.append((mol_bond.GetBeginAtomIdx(), mol_bond.GetEndAtomIdx()))
            else:
                all_mol_bonds_atom_tuples.append((compound.GetBonds()[mol_bond].GetBeginAtomIdx(),
                                                  compound.GetBonds()[mol_bond].GetEndAtomIdx()))

        all_atom_indices = set([atom_ind for single_bond_atom_tuple in all_mol_bonds_atom_tuples
                                for atom_ind in single_bond_atom_tuple])

        return set(CompoundStructureUtils.get_atom_environment(compound, all_atom_indices, n_degree))

    @staticmethod
    def atom_indices_cover_complete_rings(mol_object: Mol, atom_indices: Union[Set, List, Tuple]) -> bool:
        """ Description: Check if a set of atom indices covers complete aromatic rings. """

        all_rings_indices = mol_object.GetRingInfo().AtomRings()

        for ring_indices in all_rings_indices:
            if set(ring_indices).issubset(set(atom_indices)):
                return True

        return False

    @staticmethod
    def count_atom_index_ring_memberships(mol_object: Mol, atom_index: int) -> int:
        """ Description: Count the number of rings in which the atom with the index 'atom_index' is a member of. """

        all_rings_indices = mol_object.GetRingInfo().AtomRings()

        return len([1 for ring_indices in all_rings_indices if atom_index in ring_indices])

    @staticmethod
    def count_mol_bond_ring_memberships(mol_object: Mol, mol_bond: Union[Bond, int]) -> int:
        """ Description: Count the number of rings in which the bond 'mol_bond' is a member of.
                         The 'mol_bond' parameter can be either a Mol bond or a Mol bond index."""

        all_rings_indices = mol_object.GetRingInfo().AtomRings()

        if isinstance(mol_bond, Bond):
            begin_atom_ind = mol_bond.GetBeginAtomIdx()
            end_atom_ind = mol_bond.GetEndAtomIdx()
        else:
            begin_atom_ind = mol_object.GetBonds()[mol_bond].GetBeginAtomIdx()
            end_atom_ind = mol_object.GetBonds()[mol_bond].GetEndAtomIdx()

        return len([1 for ring_indices in all_rings_indices
                    if begin_atom_ind in ring_indices and end_atom_ind in ring_indices])

    @staticmethod
    def get_rest_of_ring_atoms(mol_object: Mol, atom_indices: Union[Set, List, Tuple]) -> List:
        """ Description: Gets the rest of ring atoms for all aromatic rings atoms covered by 'atom_indices'. """

        all_rings_indices = mol_object.GetRingInfo().AtomRings()
        new_atom_indices = [atom_ind for atom_ind in atom_indices]

        while True:
            # Detect whether some of the atom indices are members of a ring, and add the rest of this ring.
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

        return sorted(list(set(new_atom_indices)))

    @staticmethod
    def remove_salts_from_compound(compound: Union[str, Mol], salts_definition_file_path=None,
                                   verbose=False) -> Union[str, None]:
        """ Description: Remove specified salts from a chemical compound using the RDKit salt stripper. """

        try:
            if isinstance(compound, str):
                compound = CompoundConversionUtils.string_to_mol(compound)

            salt_remover = SaltRemover(defnFilename=salts_definition_file_path)

            return salt_remover.StripMol(compound)

        except Exception as exc_msg:
            if verbose:
                print("Exception occurred during the stripping of the salts from the specified compound. "
                      "Detailed message:\n{}".format(exc_msg))

            return None
