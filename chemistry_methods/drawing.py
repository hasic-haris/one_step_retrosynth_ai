"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  November 9th, 2019
Explanation: This file contains necessary functions that help with the visualization of molecules and reactions.
"""

from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from PIL import Image
from cairosvg import svg2png
import io

from chemistry_methods.reactions import parse_reaction_roles


def assign_colors_to_indices(indices_subsets):
    """ Assigns different colors to different subsets of indices. """

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

    # Return the generated colors.
    return unified_indices, colors


def draw_molecule(mol, im_size_x=300, im_size_y=200, highlight_atoms=None, highlight_bonds=None):
    """ Draws the molecule with or without highlighted individual atoms/bonds and return the image object. """

    # Check if the input molecule is given in SMILES or in the RDKit 'Mol' format.
    if isinstance(mol, str):
        try:
            # Generate the RDKit 'Mol' object from the input SMILES string.
            mol = AllChem.MolFromSmiles(mol)
            # Sanitize the molecule.
            AllChem.SanitizeMol(mol)
        except:
            print("Unable to sanitize generated molecule. Check the validity of the input SMILES string.")

    # Assign different colors to the atoms/bonds that need to be highlighted.
    highlight_atoms, highlight_atom_colors = assign_colors_to_indices(highlight_atoms)
    highlight_bonds, highlight_bond_colors = assign_colors_to_indices(highlight_bonds)

    # Calculate the 2D coordinates for the molecule.
    AllChem.Compute2DCoords(mol)

    # Check whether the molecule has correct property notations.
    try:
        mol.GetAtomWithIdx(0).GetExplicitValence()
    except RuntimeError:
        mol.UpdatePropertyCache(False)
    try:
        new_mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True)
    except ValueError:
        new_mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)

    # Create the drawer object, draw the molecule and return the final Image object.
    drawer = rdMolDraw2D.MolDraw2DSVG(im_size_x, im_size_y)

    # Draw molecules according to the specified highlighted elements.
    if highlight_atoms is not None and highlight_bonds is None:
        drawer.DrawMolecule(new_mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors)
    elif highlight_atoms is None and highlight_bonds is not None:
        drawer.DrawMolecule(new_mol, highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)
    else:
        drawer.DrawMolecule(new_mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors,
                            highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)

    # Finish drawing and edit the .svg string.
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")

    # Convert the .svg string to .png and then to an Image object.
    return Image.open(io.BytesIO(svg2png(svg)))


def draw_reaction(rxn, show_reagents=True, reaction_cores=None, im_size_x=300, im_size_y=200):
    """ Draws the chemical reaction with or without highlighted reaction cores and reactive parts. """

    # Parse the roles from the input object.
    if reaction_cores is None:
        reaction_cores = [[], []]
    if isinstance(rxn, str):
        reactants, reagents, products = parse_reaction_roles(rxn, as_what="mol")
    else:
        reactants = rxn.GetReactants()
        products = rxn.GetProducts()
        reagents = []

    mol_images = []

    # Draw images of the reactant molecules and append '+' symbol image after each one, except the last one which needs
    # to be followed by the '->' symbol.
    for r_ind, reactant in enumerate(reactants):
        if len(reaction_cores[0]) > 0:
            mol_images.append(draw_molecule(reactant, im_size_x, im_size_y, highlight_atoms=[reaction_cores[0][r_ind]]))
        else:
            mol_images.append(draw_molecule(reactant, im_size_x, im_size_y))

        if r_ind == len(reactants) - 1:
            mol_images.append(Image.open("assets/arrow.png"))
        else:
            mol_images.append(Image.open("assets/plus.png"))

    # If specified, draw all agent molecules in similar fashion as the reactants.
    if len(reagents) > 0 and show_reagents:
        for rg_ind, reagent in enumerate(reagents):
            mol_images.append(draw_molecule(reagent, im_size_x, im_size_y))
            if rg_ind == len(reagents) - 1:
                mol_images.append(Image.open("assets/arrow.png"))
            else:
                mol_images.append(Image.open("assets/plus.png"))

    # Draw all product molecules.
    for p_ind, product in enumerate(products):
        if len(reaction_cores[1]) > 0:
            mol_images.append(draw_molecule(product, im_size_x, im_size_y, highlight_atoms=[reaction_cores[1][p_ind]]))
        else:
            mol_images.append(draw_molecule(product, im_size_x, im_size_y, highlight_atoms=[]))
        if p_ind != len(products) - 1:
            mol_images.append(Image.open("assets/plus.png"))

    # Adjust the widths and the heights of the images and generate the final images.
    widths, heights = zip(*(i.size for i in mol_images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    # Calculate the height and width offsets for the smaller '+' and '->' images and append everything into a single
    # image representing the reaction.
    x_offset, y_offset = 0, 0
    for ind, im in enumerate(mol_images):
        if ind % 2 != 0:
            y_offset = round(im_size_y / 2 - im.size[1] / 2)
        else:
            y_offset = 0

        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]

    # Return the newly created image.
    return new_im


def draw_fingerprint_substructures(mol, radius, from_atoms=None, im_size_x=250, im_size_y=250):
    """ Draws the fingerprint substructures of a molecule for a specified radius. """

    # Check if the input molecule is given in SMILES or in the RDKit Mol format.
    if isinstance(mol, str):
        try:
            # Generate the RDKit Mol object from the input SMILES string.
            mol = AllChem.MolFromSmiles(mol)
            # Sanitize the molecule.
            AllChem.SanitizeMol(mol)
        except:
            print("Unable to sanitize generated molecule. Check the validity of the input SMILES string.")

    # Generate the full-length fingerprint of the molecule and collect the info about the active bits.
    bit_info = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, fromAtoms=from_atoms, bitInfo=bit_info)
    on_bits = [(mol, x, bit_info) for x in fp.GetOnBits()]

    # Create the drawer object only for active bits.
    drawer = Draw.DrawMorganBits(on_bits, molsPerRow=3, subImgSize=(im_size_x, im_size_y),
                                 legends=[str(x) for x in fp.GetOnBits()])
    # Modify the .svg string.
    svg = drawer.replace("svg:", "")

    # Return the final Image object.
    return Image.open(io.BytesIO(svg2png(svg)))
