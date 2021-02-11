from indigo import Indigo


def indigo_atom_map_reaction(rxn_smiles: str, timeout_period: int, existing_mapping="discard", verbose=False):

    try:
        # Instantiate the Indigo class object and set the timeout period.
        indigo_mapper = Indigo()
        indigo_mapper.setOption("aam-timeout", timeout_period)

        # Return the atom mapping of the reaction SMILES string.
        rxn = indigo_mapper.loadReaction(rxn_smiles)
        rxn.automap(existing_mapping)

        return rxn.smiles()

    # If an exception occurs for any reason, print the message if indicated, and return None.
    except Exception as exception:
        if verbose:
            print(
                "Exception occured during atom mapping of the reaction SMILES. Detailed message: {}".format(exception))

        return None
