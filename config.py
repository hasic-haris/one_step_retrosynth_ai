import json
from argparse import ArgumentParser
from typing import NamedTuple, Optional


class DataConfig(NamedTuple):
    """ Description: Class containing all of the necessary configuration parameters for data pre-processing. """

    # The path to the input dataset file which contains the chemical reactions.
    input_dataset_file_path: str
    # The extension of the input dataset file.
    input_dataset_file_extension: str
    # The separator of the input dataset file.
    input_dataset_file_separator: str
    # The name of the column containing the reaction SMILES strings in the input dataset file.
    reaction_smiles_column: str
    # The name of the column containing the reaction class label in the input dataset file.
    reaction_class_column: str
    # The path where all pre-processed dataset files should be saved.
    output_folder_path: str

    # The flag whether to use multi-processing or not.
    use_multiprocessing: bool



    # Number of folds for the n-fold cross-validation procedure.
    num_folds: int
    negative_samples: str
    # The percentage of
    validation_split: dict
    # Random seed for reproducibility purpouses.
    random_seed: int
    # The final dataset.
    final_classes: list


class DescriptorConfig(NamedTuple):

    similarity_search: dict
    model_training: list


class ModelConfig(NamedTuple):

    logs_folder: str
    fixed_model: int
    random_seed: float
    use_oversampling: int
    learning_rate: float
    max_epochs: int
    batch_size: int
    early_stopping: int

    input_layer: dict
    output_layer: dict
    hidden_layers: list


class EvaluationConfig(NamedTuple):

    best_fold: int
    best_input_config: dict
    final_evaluation_dataset: str


class FullConfig(NamedTuple):

    data_config: DataConfig
    descriptor_config: DescriptorConfig
    model_config: ModelConfig
    evaluation_config: EvaluationConfig

    @classmethod
    def load(cls, file_path: Optional[str] = None) -> "FullConfig":
        if file_path is None:
            parser = ArgumentParser()
            parser.add_argument("config", type=str)
            args = parser.parse_args()
            file_path = args.config

        with open(file_path) as read_handle:
            settings = json.load(read_handle)

            if "data_config" not in settings or "descriptor_config" not in settings or "model_config" not in settings \
                    or "evaluation_config" not in settings:
                raise ValueError("Mandatory setting groups are missing from the configuration file.")

            return cls(data_config=DataConfig(**settings["data_config"]),
                       descriptor_config=DescriptorConfig(**settings["descriptor_config"]),
                       model_config=ModelConfig(**settings["model_config"]),
                       evaluation_config=EvaluationConfig(**settings["evaluation_config"]))
