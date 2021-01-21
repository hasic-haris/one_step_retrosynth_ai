import multiprocessing as mp
import pandas as pd

from typing import List
from rdkit.Chem import AllChem
from general_utils import ConversionUtils, StructureUtils
from config.config import FullConfig

from tqdm import tqdm


class DatasetPreProcessing:
    """
    Description:
        Python implementation of the dataset pre-processing functions applied in the KNIME worklfow sent by RIKEN/Kyoto University.
    Configuration:
        Configuration parameters are defined and clarified in the 'config.json' and 'config.py' files.
    """

    use_multiprocessing: bool
    unwanted_elements: List[str]

    def __init__(self, config_params):
        self.use_multiprocessing = config_params.use_multiprocessing
        self.unwanted_elements = [ConversionUtils.smarts_to_mol(uwe_smarts) for uwe_smarts in
                                  open(config_params.unwanted_elements, "r").read().split("\n")]
        self.unwanted_elements = [uw_element for uw_element in self.unwanted_elements if uw_element is not None]

    def pre_process_entry(self, smiles: str) -> str:
        """
        Description:
            Pre-process a single molecule SMILES string by converting it to the canonical SMILES represantation,
            removing the salts, checking for unwanted elements, and normalizing the molecular structure using RDKit.
            Returns None if any of the mentioned processing stages fail.
        Input:
            smiles (str): A SMILES string of the molecule entry which is being processed.
        Output:
            (str): A Canonical SMILES string of the processed molecule entry.
        """

        # Convert the SMILES string into a Canonical SMILES representation.
        canonical_smiles = ConversionUtils.smiles_to_canonical_smiles(smiles)

        # Remove any salts that are present in the SMILES string.
        if "." in canonical_smiles:
            canonical_smiles = StructureUtils.remove_salts(canonical_smiles, apply_ad_hoc_stripper=True)

        # Check if the SMILES string of the compound contains any of the unwanted elements like
        # inappropriate substructures and non-druglike elements.
        if canonical_smiles is None or any(ConversionUtils.smiles_to_mol(smiles).HasSubstructMatch(uw_element)
                                           for uw_element in self.unwanted_elements):
            return None
        else:
            normalized_structure = StructureUtils.normalize_structure(canonical_smiles)

            # Check the consistency between the Canonical SMILES and normalized structure SMILES.
            if canonical_smiles != normalized_structure:
                return None
            else:
                return normalized_structure

    def pre_process_dataset(self, compound_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Description:
            Pre-processing of the raw input data.
        Requirements:
            The dataset needs to be a Pandas dataframe and needs to contain the following columns for this method to function correctly: id, smiles.
        """

        # Remove duplicates rows according to the 'id' values and rows that are missing the 'smiles' value.
        compound_dataset = compound_dataset.drop_duplicates(subset=["id"])
        compound_dataset = compound_dataset.dropna(subset=["smiles"])
        compound_dataset = compound_dataset[compound_dataset["smiles"] != ""]

        # Generate a new 'canonical_smiles' column, and remove duplicates and rows with missing values.
        if self.use_multiprocessing:
            process_pool = mp.Pool(mp.cpu_count())
            compound_dataset["canonical_smiles"] = tqdm(
                process_pool.imap(self.pre_process_entry, compound_dataset["smiles"].values),
                total=len(compound_dataset["smiles"].values), ascii=True, desc="Processing entries")
            process_pool.close()
            process_pool.join()
        else:
            compound_dataset["canonical_smiles"] = [self.pre_process_entry(smiles) for smiles in
                                                    tqdm(compound_dataset["smiles"].values,
                                                         ascii=True, desc="Processing entries")]

        compound_dataset = compound_dataset.dropna(subset=["canonical_smiles"])
        compound_dataset = compound_dataset.drop_duplicates(subset=["canonical_smiles"])

        # Return the processed dataset sorted by the 'id' column.
        return compound_dataset.sort_values(by=["id"])

    def example_usage(self):
        # Reading example datasets which were provided by RIKEN/Kyoto University for this task.
        input_dataset_path = "/nasa/datasets/riken_retrosynthesis/knime_input/Chemble_A2AR_human_ki.tsv"
        test_dataset_path = "/nasa/datasets/riken_retrosynthesis/knime_input/A2AR_Mol_NoSalt_Norm_CanSmiles.xlsx"

        # Since datasets can have different formats and columns, the processing of raw sources is not included in the code.
        # If you want to run this code on any input dataset, you need to do the following:

        # 1. Read the raw dataset into Pandas.
        input_dataset = pd.read_csv(input_dataset_path, sep="\t")[
            ["Molecule ChEMBL ID", "Molecule Name", "Compound Key",
             "Molecule Max Phase", "Molecular Weight", "#RO5 Violations",
             "AlogP", "Smiles"]]

        # 2. Make sure that columns called 'id' and 'smiles' exist. The other columns don't matter for this processing.
        input_dataset.columns = ["id", "name", "compound_key", "max_phase", "molecular_weight", "ro5_violations",
                                 "a_logp", "smiles"]
        before_pre_processing = input_dataset.shape

        # 3. Run the pre-processing code.
        input_dataset = self.pre_process_dataset(input_dataset)

        print("\nInput dataset shape before pre-processing: {}".format(before_pre_processing))
        print("Input dataset shape after pre-processing: {}".format(input_dataset.shape))

        # This is everything. The rest of this code is just the analysis with the original KNIME workflow results.
        test_dataset = pd.read_excel(test_dataset_path)
        print("Test dataset shape: {}".format(test_dataset.shape))

        combined_dataset = pd.merge(input_dataset, test_dataset, left_on="id", right_on="Molecule ChEMBL ID",
                                    how="left").drop("Molecule ChEMBL ID", axis=1)

        diff_ctr, diff_ctr_nan = 0, 0
        for _, row in combined_dataset.iterrows():
            if row["canonical_smiles"] != row["Mol_NoSalt_Norm_CanonicalSMILES"]:
                diff_ctr += 1
                # print(row["smiles"] + "," + row["canonical_smiles"] + "," + str(row["Mol_NoSalt_Norm_CanonicalSMILES"]))
                if type(row["Mol_NoSalt_Norm_CanonicalSMILES"]) == str:
                    diff_ctr_nan += 1

        print("\nProcessed Dataset vs. Test Dataset:")
        print("-----------------------------------")
        print("1. Identical Result: {} ({}%)".format(len(combined_dataset.index) - diff_ctr,
                                                     round(((len(combined_dataset.index) - diff_ctr) * 100) / (
                                                         len(input_dataset.index)), 2)))
        print("2. Different Result: {} ({}%)".format(diff_ctr, round((diff_ctr * 100) / (len(input_dataset.index)), 2)))
        print("   2.a. Test NaN Values: {} ({}%)".format(diff_ctr - diff_ctr_nan,
                                                         round(((diff_ctr - diff_ctr_nan) * 100) / diff_ctr, 2)))
        print("   2.b. Test SMILES Values: {} ({}%)\n".format(diff_ctr_nan, round((diff_ctr_nan * 100) / diff_ctr, 2)))

    def example_usage_zinc(self):
        # Reading example datasets which were provided by RIKEN/Kyoto University for this task.
        input_dataset_path = "/home/haris/projects/sa_score_analysis/dataset_pickles/zinc15_dataset.csv"
        test_dataset_path = ""

        # 1. Read the raw dataset into Pandas.
        input_dataset = pd.read_csv(input_dataset_path)

        # 2. Make sure that columns called 'id' and 'smiles' exist. The other columns don't matter for this processing.
        input_dataset.columns = ["id", "smiles", "sa_score"]
        before_pre_processing = input_dataset.shape

        # 3. Run the pre-processing code.
        input_dataset = self.pre_process_dataset(input_dataset)

        print("\nInput dataset shape before pre-processing: {}".format(before_pre_processing))
        print("Input dataset shape after pre-processing: {}".format(input_dataset.shape))

        input_dataset.to_pickle("/home/haris/projects/sa_score_analysis/dataset_pickles/zinc15_test.pkl")

        # 2. Make sure that columns called 'id' and 'smiles' exist. The other columns don't matter for this processing.
        # input_dataset.columns = ["id", "name", "compound_key", "max_phase", "molecular_weight", "ro5_violations", "a_logp", "smiles"]
        # before_pre_processing = input_dataset.shape


full_config = FullConfig.load()
knime_utils = DatasetPreProcessing(full_config.dataset_pre_processing_config)
knime_utils.example_usage_zinc()
