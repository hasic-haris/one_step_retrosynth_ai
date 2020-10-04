import json
from argparse import ArgumentParser
from typing import NamedTuple, Optional


class DatasetsConfig(NamedTuple):

    # Dataset paths information.
    raw_dataset: str
    output_folder: str

    # The n-fold cross-validation parameters.
    num_folds: int
    validation_split: dict
    random_seed: int

    # The final dataset.
    final_classes: list


class DescriptorConfig(NamedTuple):

    similarity_search: dict
    model_training: list


class ModelConfig(NamedTuple):

    fixed_model: int


class FullConfig(NamedTuple):

    dataset_config: DatasetsConfig
    descriptor_config: DescriptorConfig
    model_config: ModelConfig

    @classmethod
    def load(cls, file_path: Optional[str] = None) -> "FullConfig":
        if file_path is None:
            parser = ArgumentParser()
            parser.add_argument("config", type=str)
            args = parser.parse_args()
            file_path = args.config

        with open(file_path) as read_handle:
            settings = json.load(read_handle)

            if "dataset_config" not in settings or "descriptor_config" not in settings or \
                    "model_config" not in settings:
                raise ValueError("Mandatory setting groups are missing from the configuration file.")

            return cls(dataset_config=DatasetsConfig(**settings["dataset_config"]),
                       descriptor_config=DescriptorConfig(**settings["descriptor_config"]),
                       model_config=ModelConfig(**settings["model_config"]))