import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from typing import List

from rdkit.Chem import AllChem
from general_utils import ConversionUtils, StructureUtils, ReactionMapper
from utils_config import FullConfig
from itertools import repeat
from rdchiral.template_extractor import extract_from_reaction
import signal
from typing import List, Dict
from multiprocessing.dummy import Pool as ThreadPool
from ctypes import c_char_p
from functools import partial
from collections import Counter
import datetime, time


class TemplateUtils:
    """ Description:   Python implementation of the functions for the automatic extraction of reaction rule templates.
        Configuration: Configuration parameters are defined and clarified in the utils_config files.
        Author:        Haris Hasic (Elix Inc.) on 30th November, 2020. """

    def __init__(self, config_params):
        """" Description: Constructor to initialize the configuration parameters. """

        # General settings.
        self.use_multiprocessing = config_params.use_multiprocessing
        self.output_folder_path = config_params.output_folder_path

        # Reaction SMILES cleaning settings.
        self.salts_file_path = config_params.salts_file_path

        # Epam Indigo reaction SMILES mapping API settings.
        self.atom_mapping_timeout = config_params.atom_mapping_timeout
        self.handle_existing_mapping = config_params.handle_existing_mapping

        # Reaction template extraction and filtering settings.
        self.extract_template_timeout = int(config_params.extract_template_timeout / 1000)
        self.top_n_templates = config_params.top_n_templates

        # Additional settings.
        pd.options.mode.chained_assignment = None

    def clean_entry(self, rxn_smiles: str) -> str:
        """ Description: Clean a reaction SMILES entry using the approach described in the journal article. """

        # Extract reaction roles from the reaction SMILES string.
        reactant_smiles, _, product_smiles = ConversionUtils.rxn_smiles_to_rxn_roles(rxn_smiles)

        # Concatenate the individual reactant and product SMILES for salt removal. The reagents are ignored.
        reactant_side = ".".join(reactant_smiles)
        product_side = ".".join(product_smiles)

        # Remove the user-defined salts.
        reactant_side = StructureUtils.remove_salts(reactant_side, salt_list_file_path=self.salts_file_path)
        product_side = StructureUtils.remove_salts(product_side, salt_list_file_path=self.salts_file_path)

        # Filter reactions that do not have any reactants or products.
        if reactant_side is None or reactant_side == "" or product_side is None or product_side == "":
            return None
        else:
            return ">>".join([reactant_side, product_side])

    def atom_map_entry_wrapper(self, rxn_smiles: str) -> str:
        """ Description: Wrapper function for the atom mappping a reaction SMILES entry. """

        return ReactionMapper.atom_map_reaction(rxn_smiles,
                                                timeout_period=self.atom_mapping_timeout,
                                                existing_mapping=self.handle_existing_mapping)

    def extract_template_from_entry_mp_wrapper(self, rxn_smiles: str) -> str:
        """ Description: Wrapper function for the template extraction from a reaction SMILES entry. """

        return ReactionMapper.extract_reaction_template(rxn_smiles)

    def lol(self, rxn_smiles: str, ret_value) -> str:
        """ Description: TBD. """

        ret_value.value = ReactionMapper.extract_reaction_template(rxn_smiles)

    def extract_template_from_entry_wrapper(self, rxn_smiles: str) -> str:
        """ Description: TBD. """

        manager = mp.Manager()
        ret_val = manager.Value(c_char_p, "")

        process = mp.Process(target=self.lol, args=(rxn_smiles, ret_val,))

        process.start()
        process.join(timeout=self.extract_template_timeout)

        if process.is_alive():
            process.terminate()
            return None

        return ret_val.value

    def collect_result(self, func_result):
        """ Description: Helper function to fetch the result of the asynchronously called function. """

        return func_result

    def abortable_worker(self, func, *args, **kwargs):
        """ Description: Helper function to implement a timeout for the Pool object. """

        try:
            thread_pool_process = ThreadPool(1)
            func_result = thread_pool_process.apply_async(func, args=args)

            return func_result.get(self.extract_template_timeout)

        # If a timeout happens, print the error message and return None.
        except mp.TimeoutError as ex:
            print("Timeout error happened. Detailed message: {}".format(ex))
            return None

    def clean_reaction_smiles(self, input_dataset: pd.DataFrame, input_column: str, output_column: str,
                              save_result_to_file=False) -> pd.DataFrame:
        """ Description: Clean the reaction SMILES entries from a specified dataset. """

        # Remember the starting length of the dataset.
        entries_before_cleaning = len(input_dataset.index)

        # Step 1: Remove duplicate reaction SMILES strings.
        input_dataset = input_dataset.drop_duplicates(subset=[input_column])

        # Step 2: Clean all of the reaction SMILES strings.
        if self.use_multiprocessing:
            process_pool = mp.Pool(mp.cpu_count() - 10)
            input_dataset[output_column] = tqdm(process_pool.imap(self.clean_entry, input_dataset[input_column].values),
                                                total=len(input_dataset[input_column].values), ascii=True,
                                                desc="Cleaning reaction SMILES")
            process_pool.close()
            process_pool.join()
        else:
            input_dataset[output_column] = [self.clean_entry(rxn_smiles) for rxn_smiles in
                                            tqdm(input_dataset[input_column].values,
                                                 ascii=True, desc="Cleaning reaction SMILES")]

        # Step 3: Drop all of the reactions with no reactant or product side and duplicate reaction SMILES strings.
        input_dataset = input_dataset.dropna(subset=[output_column])
        input_dataset = input_dataset.drop_duplicates(subset=[output_column])

        # Print a short summary of the cleaning process.
        print("Dataset entries before cleaning: {}".format(entries_before_cleaning))
        print("Dataset entries after cleaning: {}".format(len(input_dataset.index)))

        # If indicated, save the dataset containing the cleaned reaction SMILES strings.
        if save_result_to_file:
            input_dataset.to_csv(self.output_folder_path +
                                 datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S_") +
                                 "input_dataset_clean.csv")

        # Finally, return the modified dataset.
        return input_dataset

    def atom_map_reaction_smiles(self, input_dataset: pd.DataFrame, input_column: str, output_column: str,
                                 save_result_to_file=False) -> pd.DataFrame:
        """ Description: Atom map the reaction SMILES entries from a specified dataset using the Epam Indigo API. """

        # Perform the atom mapping on the reaction SMILES.
        if self.use_multiprocessing:
            process_pool = mp.Pool(mp.cpu_count() - 10)
            input_dataset[output_column] = tqdm(
                process_pool.imap(self.atom_map_entry_wrapper, input_dataset[input_column].values),
                total=len(input_dataset[input_column].values), ascii=True, desc="Mapping reaction SMILES")
            process_pool.close()
            process_pool.join()
        else:
            input_dataset[output_column] = [self.atom_map_entry_wrapper(rxn_smiles) for rxn_smiles in
                                            tqdm(input_dataset[input_column].values,
                                                 ascii=True, desc="Mapping reaction SMILES")]

        # If indicated, save the dataset containing the atom mapped reaction SMILES strings.
        if save_result_to_file:
            input_dataset.to_csv(self.output_folder_path +
                                 datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S_") +
                                 "input_dataset_mapped.csv")

        # Print a short summary of the atom mapping process.
        print("Successfully mapped {}/{} reaction SMILES using the Epam Indigo API.".format(
            len(input_dataset.dropna(subset=[output_column]).index),
            len(input_dataset.index)))

        # Finally, return the modified dataset.
        return input_dataset

    def extract_reaction_templates(self, input_dataset: pd.DataFrame, input_column: str, output_column: str,
                                   save_result_to_file=False) -> pd.DataFrame:
        """ Description: Extract reaction templates from the reaction SMILES entries from a specified dataset using the RDChiral library. """

        reaction_templates_async, reaction_templates = [], []

        # Extract reaction templatets from the specified reaction SMILES.
        if self.use_multiprocessing:
            pool = mp.Pool(mp.cpu_count(), maxtasksperchild=1)

            for rxn_smiles in input_dataset[input_column].values:
                abortable_func = partial(self.abortable_worker, self.extract_template_from_entry_mp_wrapper)
                reaction_templates_async.append(
                    pool.apply_async(abortable_func, args=(rxn_smiles,), callback=self.collect_result).get())

            pool.close()
            pool.join()

            # for reaction_template in reaction_templates_async:
            #    try:
            #        reaction_templates.append(reaction_template.get())
            #    except:
            #        reaction_templates.append(None)
            #        continue

            input_dataset[output_column] = reaction_templates_async

        else:
            input_dataset[output_column] = [self.extract_template_from_entry_wrapper(rxn_smiles) for rxn_smiles in
                                            tqdm(input_dataset[input_column].values, ascii=True,
                                                 desc="Extracting reaction templates")]

        # If indicated, save the dataset containing the extracted reaction templates.
        if save_result_to_file:
            input_dataset.to_csv(self.output_folder_path +
                                 datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S_") +
                                 "input_dataset_templates.csv")

        # Print a short summary of the atom mapping process.
        print("Successfully extracted {}/{} reaction templates using the RDChiral library.".format(
            len(input_dataset.dropna(subset=[output_column]).index),
            len(input_dataset.index)))

        # Finally, return the modified dataset.
        return input_dataset

    def filter_reaction_templates(self, all_reaction_templates: List[str], top_n: int, save_result_to_file=False) -> \
    List[str]:
        """ Description: Extract the most frequent reaction templates from a list of all reaction templates. """

        # Count the occurences of all reaction templates.
        template_counter = Counter(
            [reaction_template for reaction_template in all_reaction_templates if reaction_template is not None])

        # Consider only the top_n most frequent reaction templates.
        top_n_templates = [template_item[0] for template_item in
                           sorted(template_counter.items(), key=lambda item: item[1], reverse=True)
                           if template_item[1] >= top_n]

        # If indicated, save the dataset containing the extracted reaction templates.
        if save_result_to_file:
            pd.DataFrame(top_n_templates, index=None, columns=None).to_csv(self.output_folder_path +
                                                                           "reaction_templates_template_uspto_filtered_elix.sma",
                                                                           header=None, index=None)

        return top_n_templates

    def compare_generated_templates():
        pass

    def example_usage(self):
        """ Description: Example on how to use the developed class methods. """

        # Read the raw USPTO dataset.
        # input_dataset = pd.read_csv("/nasa/datasets/retro_transfer/uspto_raw/1976_Sep2016_USPTOgrants_smiles.rsmi", sep="\t",
        #                            dtype={"ReactionSmiles": str, "PatentNumber": str, "ParagraphNum": str, "Year": int,
        #                                   "TextMinedYield": str, "CalculatedYield": str}) #[0:10000]

        # Step 1: Clean the reaction SMILES entries.
        # input_dataset = self.clean_reaction_smiles(input_dataset=input_dataset, input_column="ReactionSmiles", output_column="clean_rxn_smiles", save_result_to_file=True)
        # input_dataset = pd.read_csv("/nasa/datasets/retro_transfer/uspto_raw/")
        # print(input_dataset.head(5))

        # time.sleep(10)

        # Step 2: Re-map the reaction SMILES entries with the Epam Indigo Mapping API.
        # input_dataset = self.atom_map_reaction_smiles(input_dataset=input_dataset, input_column="clean_rxn_smiles", output_column="mapped_rxn_smiles", save_result_to_file=True)
        # input_dataset = pd.read_csv("/nasa/datasets/retro_transfer/uspto_raw/")
        # print(input_dataset.head(5))

        # time.sleep(10)

        input_dataset = pd.read_csv(
            "/nasa/datasets/retro_transfer/uspto_raw/20201223012109_input_dataset_templates.csv",
            dtype={"ReactionSmiles": str, "PatentNumber": str, "ParagraphNum": str, "Year": int,
                   "TextMinedYield": str, "CalculatedYield": str, "clean_rxn_smiles": str, "mapped_rxn_smiles": str,
                   "rxn_template": str})

        # Step 3: Extract the reaction template from the mapped reaction SMILES entries with the RDChiral library.
        # input_dataset = self.extract_reaction_templates(input_dataset=input_dataset, input_column="mapped_rxn_smiles", output_column="rxn_template", save_result_to_file=False)
        # input_dataset = pd.read_csv("/nasa/datasets/retro_transfer/uspto_raw/")
        print(len(input_dataset.index))
        print(len(input_dataset.dropna(subset=["rxn_template"]).index))
        print(input_dataset.head(5))

        # Step 4: Filter only the Top-N most frequent reaction templates.
        top_n_templates = self.filter_reaction_templates(all_reaction_templates=input_dataset["rxn_template"].values,
                                                         top_n=self.top_n_templates, save_result_to_file=False)

        top_n_templates = [temp.split(">>")[1] + ">>" + temp.split(">>")[0] for temp in top_n_templates[1:]]

        pd.DataFrame(top_n_templates, index=None, columns=None).to_csv(
            self.output_folder_path + "reaction_templates_template_uspto_filtered_elix_50x.sma",
            index=None)

        print(len(top_n_templates))

        """
        top_n_templates = pd.read_csv("/nasa/datasets/retro_transfer/uspto_raw/20201223012109_input_dataset_templates.csv")

        template_counter = Counter([reaction_template for reaction_template in input_dataset["rxn_template"].values if reaction_template is not None])

        # Consider only the top_n most frequent reaction templates.
        top_n_templates = [template_item[0] for template_item in sorted(template_counter.items(), key=lambda item: item[1], reverse=True)
                           if template_item[1] >= 30]

        applicable_ctr = [template_item[1] for template_item in sorted(template_counter.items(), key=lambda item: item[1], reverse=True)
                          if template_item[1] >= 30]

        print(len(top_n_templates))
        print(sum(applicable_ctr))

        old_templates = pd.read_csv("/home/haris/projects/real_retro_riken/data/reaction_template_uspto_filtered.sma", header=None)
        new_templates = pd.read_csv(self.output_folder_path + "reaction_templates_template_uspto_filtered_elix.sma", header=None)

        print(len(old_templates.index))
        print(len(new_templates.index))



        for old_template in old_templates.values:
            for new_template in new_templates.values:
                try:
                    old_r, _, old_p = ConversionUtils.rxn_smiles_to_rxn_roles(old_template[0])
                    old_rxn = ".".join(sorted(old_r, key=len, reverse=True)) + ">>" + ".".join(sorted(old_p, key=len, reverse=True)) 
                    old_rxn = AllChem.ReactionFromSmarts(old_template[0])
                    AllChem.SanitizeRxn(old_rxn)

                    new_r, _, new_p = ConversionUtils.rxn_smiles_to_rxn_roles(new_template[0])
                    new_rxn = ".".join(sorted(new_p, key=len, reverse=True)) + ">>" + ".".join(sorted(new_r, key=len, reverse=True))
                    new_rxn = AllChem.ReactionFromSmarts(new_rxn)
                    AllChem.SanitizeRxn(new_rxn)

                    #if len(AllChem.ReactionToSmiles(old_rxn, canonical=True)) == len(AllChem.ReactionToSmiles(new_rxn, canonical=True)):
                    if "N" in AllChem.ReactionToSmiles(new_rxn, canonical=True) and "S" in AllChem.ReactionToSmiles(new_rxn, canonical=True):
                        print(AllChem.ReactionToSmiles(old_rxn, canonical=True))
                        print(AllChem.ReactionToSmiles(new_rxn, canonical=True))
                        print()
                except Exception as ex:
                    print(ex)
                    continue
            break
        """


full_config = FullConfig.load()
template_utils = TemplateUtils(full_config.template_utils_config)
template_utils.example_usage()
