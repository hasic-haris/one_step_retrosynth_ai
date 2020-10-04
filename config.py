import json
from argparse import ArgumentParser
from typing import NamedTuple, Optional


class DatasetsConfig(NamedTuple):

    # Dataset paths information.
    raw_dataset_path: str
    output_folder: str

    # The n-fold cross-validation parameters.
    num_folds: int
    validation_split: dict
    random_seed: int

    # The final dataset.
    final_classes: list


class FingerprintConfig(NamedTuple):

    similarity_search: dict
    model_training: list


class FullConfig(NamedTuple):
    datasets: DatasetsConfig
    fp_config: FingerprintConfig

    @classmethod
    def load(cls, file_path: Optional[str] = None) -> "FullConfig":
        if file_path is None:
            parser = ArgumentParser()
            parser.add_argument("config", type=str)
            args = parser.parse_args()
            file_path = args.config

        with open(file_path) as read_handle:
            settings = json.load(read_handle)

            if "datasets" not in settings or "fp_config" not in settings:
                raise ValueError("Mandatory setting groups are missing from the configuration file.")

            return cls(datasets=DatasetsConfig(**settings["datasets"]),
                       fp_config=FingerprintConfig(**settings["fp_config"]))
