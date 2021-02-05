"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  November 9th, 2019
Edited on:   January 31st, 2021
"""

import io

from cairosvg import svg2png
from PIL import Image
from typing import Dict, List, Tuple, Union

from rdkit.Chem import AllChem, Draw

from .compound_utils import CompoundConversionUtils
from .descriptor_utils import MolecularFingerprintsUtils
from .reaction_utils import ReactionConversionUtils


class VisualizationUtils:
    """ Description: Group of methods for the handling of the visualization of chemical compounds and reactions. """

    @staticmethod
    def __assign_colors_to_indices(indices_subsets: Union[List, Tuple]) -> Tuple[List, Dict]:
        """ Description: Assign different colors to different subsets of indices. """

        # If there are no highlighted elements, return no colors.
        if indices_subsets is None:
            return [], {}

        # Define the colors that will be used for highlighting different groups of elements.
        color_codes = {1: (0.9, 0.4, 0.4), 2: (0.1, 0.9, 0.4), 3: (0.1, 0.4, 0.9), 4: (0.9, 1, 0.4), 5: (0.9, 0.4, 0.9)}
        colors, unified_indices = {}, []

        # Add colors to different subsets.
        color_key = 1
        for subset in indices_subsets:
            for s in subset:
                unified_indices.append(s)

                if color_key in color_codes.keys():
                    colors.update({s: color_codes[color_key]})
                else:
                    colors.update({s: color_codes[color_key - 1]})

            color_key = color_key + 1

        return unified_indices, colors

    @staticmethod
    def draw_molecule(compound: Union[str, AllChem.Mol], image_size_x=300, image_size_y=200, highlight_atoms=None,
                      highlight_bonds=None) -> Image:
        """ Description: Draw a single chemical compound with or without highlighted individual atoms/bonds. """

        if isinstance(compound, str):
            compound = CompoundConversionUtils.string_to_mol(compound)

            if compound is None:
                raise Exception("The given input is not a valid chemical compound.")

        # Assign different colors to the atoms/bonds that need to be highlighted.
        highlight_atoms, highlight_atom_colors = VisualizationUtils.__assign_colors_to_indices(highlight_atoms)
        highlight_bonds, highlight_bond_colors = VisualizationUtils.__assign_colors_to_indices(highlight_bonds)

        # Check whether the molecule has correct property notations.
        try:
            AllChem.Compute2DCoords(compound)
            compound.GetAtomWithIdx(0).GetExplicitValence()
        except RuntimeError:
            compound.UpdatePropertyCache(False)
        try:
            compound = Draw.rdMolDraw2D.PrepareMolForDrawing(compound, kekulize=True)
        except ValueError:
            compound = Draw.rdMolDraw2D.PrepareMolForDrawing(compound, kekulize=False)

        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(image_size_x, image_size_y)

        # Draw molecules according to the specified highlighted elements.
        if highlight_atoms is not None and highlight_bonds is None:
            drawer.DrawMolecule(compound, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors)

        elif highlight_atoms is None and highlight_bonds is not None:
            drawer.DrawMolecule(compound, highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)

        else:
            drawer.DrawMolecule(compound, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors,
                                highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)

        # noinspection PyArgumentList
        drawer.FinishDrawing()

        # noinspection PyArgumentList
        svg = drawer.GetDrawingText().replace("svg:", "")

        # Convert the .svg string to .png and then to an Image object.
        return Image.open(io.BytesIO(svg2png(svg)))

    @staticmethod
    def draw_reaction(reaction: Union[str, AllChem.ChemicalReaction], show_reagents=True, reaction_cores=None,
                      image_size_x=300, image_size_y=200) -> Image:
        """ Description: Draw a chemical reaction with or without highlighted reaction cores. """

        if isinstance(reaction, AllChem.ChemicalReaction):
            reaction = ReactionConversionUtils.reaction_to_string(reaction)

            if reaction is None:
                raise Exception("The given input is not a valid chemical reaction.")

        reactants, reagents, products = ReactionConversionUtils.parse_roles_from_reaction_smiles(reaction,
                                                                                                 as_what="mol")

        if reaction_cores is None:
            reaction_cores = [[], []]

        mol_images = []

        for r_ind, reactant in enumerate(reactants):
            if len(reaction_cores[0]) > 0:
                mol_images.append(VisualizationUtils.draw_molecule(reactant, image_size_x, image_size_y,
                                                                   highlight_atoms=[reaction_cores[0][r_ind]]))
            else:
                mol_images.append(VisualizationUtils.draw_molecule(reactant, image_size_x, image_size_y))

            if r_ind == len(reactants) - 1:
                mol_images.append(Image.open("assets/arrow.png"))
            else:
                mol_images.append(Image.open("assets/plus.png"))

        # Only if it is specified, draw all reagent molecules in similar fashion as the reactants.
        if len(reagents) > 0 and show_reagents:
            for rg_ind, reagent in enumerate(reagents):
                mol_images.append(VisualizationUtils.draw_molecule(reagent, image_size_x, image_size_y))

                if rg_ind == len(reagents) - 1:
                    mol_images.append(Image.open("assets/arrow.png"))
                else:
                    mol_images.append(Image.open("assets/plus.png"))

        for p_ind, product in enumerate(products):
            if len(reaction_cores[1]) > 0:
                mol_images.append(VisualizationUtils.draw_molecule(product, image_size_x, image_size_y,
                                                                   highlight_atoms=[reaction_cores[1][p_ind]]))
            else:
                mol_images.append(VisualizationUtils.draw_molecule(product, image_size_x, image_size_y,
                                                                   highlight_atoms=[]))

            if p_ind != len(products) - 1:
                mol_images.append(Image.open("assets/plus.png"))

        # Adjust the widths and the heights of the final images.
        widths, heights = zip(*(i.size for i in mol_images))
        total_width = sum(widths)
        max_height = max(heights)
        new_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))

        # Calculate the height and width offsets for the asset images and append everything into a single image.
        x_offset, y_offset = 0, 0

        for ind, image in enumerate(mol_images):
            if ind % 2 != 0:
                y_offset = round(image_size_y / 2 - image.size[1] / 2)
            else:
                y_offset = 0

            new_image.paste(image, (x_offset, y_offset))
            x_offset += image.size[0]

        return new_image

    @staticmethod
    def draw_fingerprint_substructures(compound: Union[str, AllChem.Mol], radius: int, bits: int, from_atoms=None,
                                       image_size_x=300, image_size_y=200) -> Image:
        """ Description: Draw a fingerprint substructures of a chemical compound. """

        if isinstance(compound, str):
            compound = CompoundConversionUtils.string_to_mol(compound)

            if compound is None:
                raise Exception("The given input is not a valid chemical compound.")

        ecfp, ecfp_bit_info = MolecularFingerprintsUtils.construct_ecfp(compound, radius=radius, bits=bits,
                                                                        from_atoms=from_atoms, return_bit_info=True)
        on_bits = [(compound, on_bit, ecfp_bit_info) for on_bit in ecfp.GetOnBits()]

        # Create the drawer object only for active bits.
        drawer = Draw.DrawMorganBits(on_bits, molsPerRow=3, subImgSize=(image_size_x, image_size_y),
                                     legends=[str(on_bit) for on_bit in ecfp.GetOnBits()])
        svg = drawer.replace("svg:", "")

        return Image.open(io.BytesIO(svg2png(svg)))
