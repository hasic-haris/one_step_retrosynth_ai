"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  October 4th, 2020
Description: This script encapsulates all of the configuration parameters as a Python Class.
"""

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

    dataset_config: DatasetsConfig
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

            if "dataset_config" not in settings or "descriptor_config" not in settings or \
                    "model_config" not in settings:
                raise ValueError("Mandatory setting groups are missing from the configuration file.")

            return cls(dataset_config=DatasetsConfig(**settings["dataset_config"]),
                       descriptor_config=DescriptorConfig(**settings["descriptor_config"]),
                       model_config=ModelConfig(**settings["model_config"]),
                       evaluation_config=EvaluationConfig(**settings["evaluation_config"]))
