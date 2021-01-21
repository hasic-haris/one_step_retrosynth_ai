from typing import Union, Tuple, List
import re

from rdkit.Chem import AllChem

from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from rdchiral.main import rdchiralRun, rdchiralRunText
from rdchiral.template_extractor import extract_from_reaction


class PyReactor:

    @staticmethod
    def forward_apply_rdkit(reactants: Union[Tuple[AllChem.Mol], List[AllChem.Mol]], reaction_template: str,
                            return_type="mol") -> List[tuple]:
        """ Description: Apply the specified RDKit chemical reaction on a list of reactants. """

        # Create and Sanitize the RDKit ChemicalReaction object.
        reaction_rdk = AllChem.ReactionFromSmarts(reaction_template)
        AllChem.SanitizeRxn(reaction_rdk)

        # Create the RDKit Mol objects for each of the reactants.
        reactants_rdk = [reactant for reactant in reactants]

        # Remove any potential duplicate suggestions.
        products_suggestions = PyReactor.__remove_duplicate_reactants(reaction_rdk.RunReactants(reactants_rdk))

        if return_type == "str":
            return products_suggestions
        else:
            # Convert the SMILES strings into RDKit Mol objects and check their chemical validity.
            return PyReactor.__check_suggested_mol_validity(products_suggestions)

    @staticmethod
    def forward_apply_rdchiral(reactants: Union[Tuple[AllChem.Mol], List[AllChem.Mol]], reaction_template: str,
                               return_type="mol", processing_mode="rdc_full") -> List[tuple]:
        """ Description: Apply the specified RDChiral chemical reaction on a list of reactants.
            Dev Note: It seems as if RDChiral cannot run multiple reactants. This needs to be checked.
        """

        reactants_smiles = [AllChem.MolToSmiles(reactant, canonical=True) for reactant in reactants]

        if processing_mode == "rdc_partial":
            # Run the textual RDChiral suggestion generator.
            products_suggestions = rdchiralRunText(reaction_template, ".".join([rs for rs in reactants_smiles]))
        else:
            # Pre-initialize the reaction rule and the product molecule using RDChiral.
            reaction_template_rdc = rdchiralReaction(reaction_template)
            reactants_rdc = rdchiralReactants(".".join([rs for rs in reactants_smiles]))

            # Run the RDChiral suggestion generator.
            products_suggestions = rdchiralRun(reaction_template_rdc, reactants_rdc)

        # Remove any potential duplicate suggestions.
        return products_suggestions

        # products_suggestions = list(set([tuple(ps.split(".")) for ps in products_suggestions]))

        if return_type == "str":
            return products_suggestions
        else:
            # Convert the SMILES strings into RDKit Mol objects and check their chemical validity.
            return PyReactor.__check_suggested_mol_validity(products_suggestions)

    @staticmethod
    def reverse_apply_rdkit(products: Union[Tuple[AllChem.Mol], List[AllChem.Mol]], reaction_template: str,
                            return_type="mol") -> List[tuple]:
        """ Description: Apply the RDKit chemical reaction backwards on a specific product. """

        # Convert the product string to a RDKit Mol object.
        products_rdk = [CxnUtils.string_to_molecule(product) for product in products]

        # Split the reaction SMARTS string into reaction reactants and products substrings.
        reactants_side, _, products_side = reaction_template.split(">")

        # Generate a reverse reaction SMARTS string.
        reverse_rxn_template = ">>".join([products_side, reactants_side])

        # Create and Sanitize the RDKit ChemicalReaction object.
        reaction_rdk = AllChem.ReactionFromSmarts(reverse_rxn_template)
        AllChem.SanitizeRxn(reaction_rdk)

        # Remove any potential duplicate suggestions.
        reactants_suggestions = PyReactor.__remove_duplicate_reactants(reaction_rdk.RunReactants(products_rdk))

        if return_type == "str":
            return reactants_suggestions
        else:
            # Convert the SMILES strings into RDKit Mol objects and check their chemical validity.
            return PyReactor.__check_suggested_mol_validity(reactants_suggestions)

    @staticmethod
    def reverse_apply_rdchiral(products: Union[Tuple[AllChem.Mol], List[AllChem.Mol]], reaction_template: str,
                               return_type="mol", processing_mode="rdc_full") -> List[tuple]:
        """ Description: Apply the RDKit chemical reaction backwards on a specific product using RDChiral. """

        # Split the reaction SMARTS string into reaction reactants and products substrings.
        reactants_side, _, products_side = reaction_template.split(">")

        if processing_mode == "rdc_partial":
            # Generate a reverse reaction SMARTS string.
            reverse_rxn_template = ">>".join([products_side, reactants_side])

            # Run the textual RDChiral suggestion generator.
            reactants_suggestions = rdchiralRunText(reverse_rxn_template, ".".join(products))

        else:
            # Extract the reverse reaction template using RDChiral.
            reverse_rxn_template_rdc = \
            extract_from_reaction({"reactants": reactants_side, "products": products_side, "_id": "0"})[
                "reaction_smarts"]

            # Pre-initialize the reaction rule and the product molecules using RDChiral.
            reverse_rxn_template_rdc = rdchiralReaction(reverse_rxn_template_rdc)
            product_rdc = rdchiralReactants(".".join(products))

            # Run the RDChiral suggestion generator.
            reactants_suggestions = rdchiralRun(reverse_rxn_template_rdc, product_rdc)

        # Remove any potential duplicate suggestions.
        reactants_suggestions = list(set([tuple(rs.split(".")) for rs in reactants_suggestions]))

        if return_type == "str":
            return reactants_suggestions
        else:
            # Convert the SMILES strings into RDKit Mol objects and check their chemical validity.
            return PyReactor.__check_suggested_mol_validity(reactants_suggestions)

    @staticmethod
    def __check_suggested_mol_validity(reactant_suggestions: List[Tuple[str]]):
        """ Description: Conver the reactant combination SMILES strings into RDKit Mol objects and check their chemical validity."""

        unique_compound_combinations_mol = []

        for reactant_tuple in reactant_suggestions:
            try:
                reactant_mols = [CxnUtils.string_to_molecule(reactant_smiles) for reactant_smiles in reactant_tuple]
                unique_compound_combinations_mol.append(tuple(reactant_mols))
            except Exception:
                continue

        return unique_compound_combinations_mol

    @staticmethod
    def __remove_duplicate_reactants(reactant_suggestions: List[Tuple[AllChem.Mol]]):
        """ Description: Remove the duplicate reactant combinations from a given list."""

        unique_compound_combinations = set()

        for reactant_tuple in reactant_suggestions:
            unique_compound_combinations.add(tuple([AllChem.MolToSmiles(reactant_smiles, canonical=True)
                                                    for reactant_smiles in reactant_tuple]))

        return list(unique_compound_combinations)


class CxnUtils:

    def __init__(self):
        self.rollout_list = []

    @staticmethod
    def react_product_to_reactants(str_pro: str, str_rxn_rule: str, return_type="str", processing_mode="rdk") -> List[
        List[str]]:
        """
        Java Prototype:  public static List<List<String>> reactProductToReactants(String strPro, String strRxnRule) throws IOException
        Description:  Applies the given reaction rule string to a specific product molecule string.
        Dev Note:  There are currently three versions of this approach available, testing will show which one works the best.
        """

        try:
            if processing_mode == "rdk":
                # Processing Mode 1: Use the newly-constructed class based on RDKit to get the reverse suggestions.
                reactant_suggestions = PyReactor.reverse_apply_rdkit(str_pro, str_rxn_rule, return_type=return_type)
            else:
                # Processing Mode 2: Use RDChiral functionalities to get the reverse suggestions.
                reactant_suggestions = PyReactor.reverse_apply_rdchiral(str_pro, str_rxn_rule, return_type=return_type,
                                                                        processing_mode=processing_mode)

        except Exception as ex:
            raise Exception(
                "The reverse application of the reaction template was unsuccessful. Detailed exception message:\n{}".format(
                    ex))

        return reactant_suggestions

    @staticmethod
    def __react_product_to_reactants(product: AllChem.Mol, reaction_rule: AllChem.ChemicalReaction,
                                     processing_mode="rdc_full") -> List[List[AllChem.Mol]]:
        """
        Java Prototype:  static List<Molecule> reactProductToReactants(Molecule product, RxnMolecule reactionRule) throws ReactionException
        Description:  Applies the given reaction rule to a specific product molecule.
        Dev Note:  There are currently three versions of this approach available, testing will show which one works the best.
                   This particular function is not that usefull except for handling input formats which can be combined together.
        """

        try:
            product = AllChem.MolToSmiles(product, canonical=True)
            reaction_rule = AllChem.ReactionToSmarts(reaction_rule)
            reactant_suggestions = react_product_to_reactants(product, reaction_rule, return_type="mol",
                                                              processing_mode=processing_mode)

            return reactant_suggestions

        except Exception as ex:
            raise Exception(
                "The reverse application of the reaction template was unsuccessful. Detailed exception message:\n{}".format(
                    ex))

    @staticmethod
    def string_to_molecule(str_mol: str, input_format="smiles"):
        """
        Java Prototype:  public static Molecule stringToMolecule(String strMol) throws MolFormatException;
        Description:  Converts a string molecule representation to a RDKit Mol object.
        Dev Note:  Original ChemAxon function supports the following string formats:
                   supported_molecule_formats: ["cml", "mdl", "rgf", "sdf", "rxn", "rdf", "csmol", "csrgf", "cssdf",
                                                "csrdf", "cdxml", "smiles", "smarts", "cxsmiles", "cxsmarts",
                                                "cxsmilesag", "iupac", "inchi", "inchikey", "name", "seq", "smol",
                                                "mol2", "pdb", "fasta", 'helm', "xyz", "cube", "gio"]
        """

        # Try converting the input string to the RDKit Mol object depending on the specified format.
        mol = None
        try:
            if input_format == "smiles":
                mol_object = AllChem.MolFromSmiles(str_mol)
            elif input_format == "smarts":
                mol_object = AllChem.MolFromSmarts(str_mol)
            else:
                raise Exception("Choose one of the currently supported formats: 'smiles', or 'smarts'.")

            # Try sanitizing the generated object.
            AllChem.SanitizeMol(mol_object)

            return mol_object

        # If an exception occurs for any reason, display the detailed message.
        except Exception as ex:
            if mol_object is None:
                print("Exception occured during the conversion process for the molecule ", end="")
            else:
                print("Exception occured during sanitization of the molecule ", end="")
                mol_object = None

            print("'{}'. Detailed exception message:\n{}".format(str_mol, ex))

    @staticmethod
    def is_valid_valence(mol: AllChem.Mol):
        """
        Java Prototype:  public static Boolean isValidValence(Molecule mol);
        Description:  Checks if the given RDKit Mol object has correct atom valence values.
        """

        # Try sanitizing the given RDKit Mol object. The sanitization operation includes:
        # 1. Kekulization, 2. Setting Valences, 3. Setting Aromaticity, 4. Setting Conjugation and Hybridization
        try:
            AllChem.SanitizeMol(mol)
            return True

        except Exception as ex:
            print(
                "Exception occured during the sanitization of the molecule. Detailed exception message:\n{}".format(ex))
            return False

    @staticmethod
    def __is_valid_reaction_result(product: AllChem.Mol, reactants: List[AllChem.Mol], rxn_template: str,
                                   processing_mode="rdk", return_type="str"):
        """
        Java Prototype:  static Boolean isValidReactionResult(Molecule product, Molecule[] reactants, String rxnTemplate)
        Description:  Applies the template in the forward direction on the given reactants and compares them to a given product to check the validity of the suggestions.
        """

        # Convert the product molecule into canonical SMILES.
        try:
            unique_product = AllChem.MolToSmiles(product, canonical=True)
        except Exception:
            print(str(ex))
            return False

        try:
            if processing_mode == "rdk":
                # Processing Mode 1: Use the newly-constructed class based on RDKit to get the forward suggestions.
                products_suggestions = PyReactor.forward_apply_rdkit(reactants, rxn_template, return_type="str")
            else:
                # Processing Mode 2: Use RDChiral functionalities to get the forward suggestions.
                products_suggestions = PyReactor.forward_apply_rdchiral(reactants, rxn_template, return_type="str",
                                                                        processing_mode=processing_mode)

            # If the unique product is found in any of the generated suggestions, then return True.
            for ps in products_suggestions:
                if unique_product in ps:
                    return True

        except Exception as ex:
            print(str(ex))
            return False

        # Otherwise, return False.
        return False

    @staticmethod
    def __get_reaction_list(reaction_rule_list_path: str, sep="\n") -> List[AllChem.ChemicalReaction]:
        """
        Java Prototype:  static List<RxnMolecule> getReactionList(String reactionRuleListPath) throws IOException;
        Description:  Reads a file containing the list of SMARTS reaction rules.
        Dev Note:  This needs to be re-evaluated since it is very limited in terms of flexibility right now.
        """

        # Try reading the specified file and convert the contents into RDKit ChemicalReaction objects.
        try:
            reaction_rules = open(reaction_rule_list_path, "r").read().split(sep)
            reaction_rules = [AllChem.ReactionFromSmarts(reaction_rule) for reaction_rule in reaction_rules]

            return reaction_rules

        except Exception as ex:
            print("Exception occured during the reading of the file. Detailed exception message:\n{}".format(ex))

    @staticmethod
    def __get_mol_list(mols: List[str], input_format="smiles") -> List[AllChem.Mol]:
        """
        Java Prototype:  static List<Molecule> getMolList(List<String> mols);
        Description:  Converts a list of molecules in string notation into a list of RDKit Mol objects.
        Dev Note:  This method seems a bit impractical since it breaks the whole conversion process if
                   a single molecule conversion fails. Worth taking a look later for improvement.
        """

        output_mols = []

        for mol in mols:
            try:
                output_mols.append(string_to_molecule(mol, input_format=input_format))

            except Exception(ex):
                print("Exception occured during the conversion. Detailed exception message:\n{}".format(ex))

    @staticmethod
    def is_terminal(self, str_mol_list: List[str]):
        """
        Java Prototype:  public static Boolean isTerminal(ArrayList<String> strMolList) throws ReactionException;
        Description:  Checks if a list of reactants is found in the list of starting materials.
        """

        # Iterate through the list of molecules and reactions.
        for mol in CxnUtils.__get_mol_list(str_mol_list):
            for reaction in self.rollout_list:
                try:
                    # If the application of the template fails or returns an empty list, return False.
                    if len(react_product_to_reactants(mol, reaction, processing_mode="rdkit")) == 0:
                        return False
                except Exception:
                    return False

        # If everything went according to plan, return True.
        return True

    def main(self):
        """
        Java Prototype:  public static void main(String[] args);
        Description: Unecessary in this context, but used for quick testing.
        """

        # Quick rundown testing of the main functions.
        normal_molecule_ex = "c1ccc(cc1)"
        failed_kekulization_ex = "c1ccc(cc1)-c1nnc(n1)-c1ccccc1"
        valid_valence_ex = "c2ncc3n2-c2ccc(nc2)-s-3C"
        invalid_valence_ex = "c2ncc3n2-c2ccc(nc2)-o(C)-s-3C"

        # Testing 'string_to_molecule'.
        print("\n\n1. Regular molecule '{}':".format(normal_molecule_ex))
        print("Result: Success ({})".format(CxnUtils.string_to_molecule(normal_molecule_ex)))

        print("\n2. Molecule with failed kekulization '{}':".format(failed_kekulization_ex))

        try:
            print("Result: Success ({})".format(CxnUtils.string_to_molecule(failed_kekulization_ex)))
        except:
            print("Result: Failure (Nothing Returned)")

        # Testing 'is_valid_valence'.
        print("\n3. Molecule with valid valence '{}':".format(valid_valence_ex))
        print("Status: {}".format(CxnUtils.is_valid_valence(AllChem.MolFromSmiles(valid_valence_ex))))

        print("\n4. Molecule with invalid valence '{}':".format(invalid_valence_ex))
        print("Status: {}".format(CxnUtils.is_valid_valence(AllChem.MolFromSmiles(invalid_valence_ex))))

        # Testing 'react_product_to_reactants'.
        print("\n5. Testing the reverse application of templates.")
        example_rxn_rules = ["[C:1][OH:2]>>[C:1][O:2][C]", "[C:1](=[O:3])[OH:2]>>[C:1](=[O:3])[O:2]CC",
                             "[C:1](=[O:3])[OH:2]>>[C:1](=[O:3])[O:2]CC", "[C:1](=[O:2])O.[N:3]>>[C:1](=[O:2])[N:3]"]

        example_reactants = [["CC(=O)OCCCO"], ["OC(=O)CCCCCC"], ["OC(=O)CCCC[C@H](Cl)C"], ["C(=O)O", "CNC"]]
        example_reactants_mol = [[AllChem.MolFromSmiles("CC(=O)OCCCO")], [AllChem.MolFromSmiles("OC(=O)CCCCCC")],
                                 [AllChem.MolFromSmiles("OC(=O)CCCC[C@H](Cl)C")],
                                 [AllChem.MolFromSmiles("C(=O)O"), AllChem.MolFromSmiles("CNC")]]

        example_products = [["COCCCOC(C)=O"], ["CCCCCCC(=O)OCC"], ["CCOC(=O)CCCC[C@@H](C)Cl"], ["CN(C)C=O"]]
        example_products_mol = [[AllChem.MolFromSmiles("COCCCOC(C)=O")], [AllChem.MolFromSmiles("CCCCCCC(=O)OCC")],
                                [AllChem.MolFromSmiles("CCOC(=O)CCCC[C@@H](C)Cl")], [AllChem.MolFromSmiles("CN(C)C=O")]]

        for example_ind in range(len(example_rxn_rules)):
            print("\nCurrently testing example {}:".format(example_ind + 1))
            print("----------------------------")

            rdc_partial = CxnUtils.react_product_to_reactants(example_products[example_ind],
                                                              example_rxn_rules[example_ind],
                                                              processing_mode="rdc_partial")
            rdc_full = CxnUtils.react_product_to_reactants(example_products[example_ind],
                                                           example_rxn_rules[example_ind], processing_mode="rdc_full")
            rdk = CxnUtils.react_product_to_reactants(example_products[example_ind], example_rxn_rules[example_ind],
                                                      processing_mode="rdk")

            results_dict = {"RDChiral_Partial": rdc_partial, "RDChiral_Full": rdc_full, "RDKit": rdk}

            print("Real: {}\n".format([AllChem.MolToSmiles(AllChem.MolFromSmiles(r), canonical=True) for r in
                                       example_reactants[example_ind]]))

            for results_key in results_dict.keys():
                if len(results_dict[results_key]) > 0:
                    print("Suggested by using {}:".format(results_key))

                    for rt_ind, reactant_tuple in enumerate(results_dict[results_key]):
                        print("#{}: {}".format(rt_ind + 1, list(reactant_tuple)))

                else:
                    print("Suggested by using {}:\nNone".format(results_key))

                print("")

        # Testing 'is_valid_reaction_result'.
        print("\n6. Testing the checking of the applied templates.")
        for example_ind in range(len(example_rxn_rules)):
            print("\nCurrently testing example {}:".format(example_ind + 1))
            print("----------------------------")

            print("RDChiral_Partial:")
            print(CxnUtils.__is_valid_reaction_result(example_products_mol[example_ind][0],
                                                      example_reactants_mol[example_ind],
                                                      example_rxn_rules[example_ind], processing_mode="rdc_partial"))
            print("\nRDChiral_Full:")
            print(CxnUtils.__is_valid_reaction_result(example_products_mol[example_ind][0],
                                                      example_reactants_mol[example_ind],
                                                      example_rxn_rules[example_ind], processing_mode="rdc_full"))
            print("\nRDKit:")
            print(CxnUtils.__is_valid_reaction_result(example_products_mol[example_ind][0],
                                                      example_reactants_mol[example_ind],
                                                      example_rxn_rules[example_ind], processing_mode="rdk"))

        print("")


cxn_utils = CxnUtils()
cxn_utils.main()
