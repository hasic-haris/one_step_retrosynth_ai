"""
Author:      Hasic Haris (Phd Student @ Ishida Lab, Department of Computer Science, Tokyo Institute of Technology)
Created on:  February 24th, 2020
Description: This file contains general TensorFlow 1.12 functions used in other files for the constructing the models.
"""

import tensorflow as tf
import pandas as pd
import numpy as np

from neural_networks.tf_model_construction import train_model, test_model
from data_methods.data_handling import read_datasets_from_fold
from data_methods.dataset_construction import generate_fps_from_reaction_products
from data_methods.data_handling import encode_one_hot


# Done: 100%
def generate_model_hp_config(input_fp_data, oversample, fixed_model_ind=0, classification_type="multi-class", **kwargs):
    """ Generates a fixed configuration dictionary for a neural network model. """

    # Pre-define the fixed architecture configurations.
    fixed_model = [[1, ["fcl"], [1024], [tf.nn.relu], [tf.initializers.glorot_normal()], [0.33]],

                   [2, ["fcl", "fcl"], [1024, 1024], [tf.nn.relu, tf.nn.relu],
                    [tf.initializers.glorot_normal(), tf.initializers.glorot_normal()], [0.33, 0.33]],

                   [3, ["fcl", "fcl", "fcl"], [1024, 1024, 1024], [tf.nn.relu, tf.nn.relu, tf.nn.relu],
                    [tf.initializers.glorot_normal(), tf.initializers.glorot_normal(), tf.initializers.glorot_normal()],
                    [0.2, 0.2, 0.2]],

                   [2, ["fcl", "hl"], [1024, 1024], [tf.nn.elu, tf.nn.relu],
                    [tf.initializers.glorot_normal(), tf.initializers.glorot_normal()], [0.2, 0.2]],

                   [4, ["fcl", "hl", "hl", "hl"], [1024, 1024, 1024, 1024],
                    [tf.nn.elu, tf.nn.relu, tf.nn.relu, tf.nn.relu],
                    [tf.initializers.glorot_normal(), tf.initializers.glorot_normal(), tf.initializers.glorot_normal(),
                     tf.initializers.glorot_normal()], [0.2, 0.2, 0.2, 0.2]],

                   [6, ["fcl", "hl", "hl", "hl", "hl", "hl"], [1024, 1024, 1024, 1024, 1024, 1024],
                    [tf.nn.elu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu],
                    [tf.initializers.glorot_normal(), tf.initializers.glorot_normal(), tf.initializers.glorot_normal(),
                     tf.initializers.glorot_normal(), tf.initializers.glorot_normal(), tf.initializers.glorot_normal()],
                    [0.2, 0.2, 0.2, 0.2]]]

    # Define the default model hyper-parameter configuration.
    model_hp_config = {
        # Input data specifications.
        "data_path": kwargs["output_folder"],                   # Input data folder path.
        "reaction_classes": kwargs["reaction_classes"],         # Reaction classes.
        "fold_ind": 1,                                          # Index of the current data fold.
        "fp_specs": input_fp_data,                              # Which fingerprint dataset to use.
        "oversample": oversample,                               # Usage of dataset oversampling.

        # General network specification.
        "classification_type": "multi-class",                   # Type of classification problem.
        "random_seed": 101,                                     # Random seed used for reproducibility purposes.
        "learning_rate": 0.0005,                                # ADAM optimizer learning rate.
        "max_num_epochs": 200,                                  # Maximum number of epochs.
        "batch_size": 128,                                      # Batch size.
        "early_stopping_interval": 10,                          # Number of epochs for early stopping detection.

        # Input layer specifications.
        "input_size": 1024,                                     # Input layer size.
        "input_activation": tf.nn.relu,                         # Input layer activation function.
        "input_init": tf.initializers.glorot_normal(),          # Input layer weight initialization function.
        "input_dropout": 0.0,                                   # Input layer dropout value.

        # Hidden layers specifications.
        "num_hidden_layers": fixed_model[fixed_model_ind][0],   # Total number of layers in the network.
        "hidden_types": fixed_model[fixed_model_ind][1],        # Individual hidden layer sizes.
        "hidden_sizes": fixed_model[fixed_model_ind][2],        # Individual hidden layer sizes.
        "hidden_activations": fixed_model[fixed_model_ind][3],  # Hidden layer activation function.
        "hidden_inits": fixed_model[fixed_model_ind][4],        # Hidden layer weight initialization function.
        "hidden_dropouts": fixed_model[fixed_model_ind][5],     # Hidden layer dropout value.

        # Output layer specifications.
        "output_size": 11,                                      # Output layer size.
        "output_activation": tf.nn.softmax,                     # Output layer activation function.
        "output_init": tf.initializers.glorot_normal(),         # Output layer weight initialization function.

        # Path to a folder to save the TensorFlow summaries of the trained network.
        "main_log_folder": "neural_networks/constructed_models/multiclass_classification/"}

    # If it is binary classification, make appropriate changes to the constructed hyper-parameter configuration.
    if classification_type == "binary":
        model_hp_config["classification_type"] = "binary"
        model_hp_config["output_size"] = 1
        model_hp_config["output_activation"] = tf.nn.sigmoid
        model_hp_config["model_log_folder"] = "neural_networks/constructed_models/binary_classification/"

    # Return the constructed hyper-parameter configuration.
    return model_hp_config


# Done: 0%
def hyperparameter_tuning():

    # Define the variable hyper-parameters that will be tested in the grid search.
    fixed_model_ind = 0
    input_fp_specs = ["hsfp_2_2_1024"]
    fold_num = 1
    batch_sizes = [16, 32, 64, 128, 256]
    learning_rates = [0.0005, 0.001, 0.00146]
    dropout_values = [[0.2], [0.33], [0.5]]

    return None


# Done: 0.5%
def train_and_evaluate_model(**kwargs):
    """ Construct, train and evaluate a binary or multi-class model through according to given specifications. """

    # Generate the model hyper-parameter configuration.
    model_hp_config = generate_model_hp_config(input_fp_data=kwargs["best_fp_config"]["folder_name"], oversample=True,
                                               **kwargs)

    avg_vl, avg_va, avg_vu, avg_vm, avg_tl, avg_ta, avg_tu, avg_tm = [], [], [], [], [], [], [], []

    # Iterate through dataset folds.
    for fold_ind in range(kwargs["num_folds"]):
        print("Training and testing models for Fold {}...".format(fold_ind+1))
        model_hp_config["fold_ind"] = fold_ind+1

        # Train the specified model on the current fold.
        _, _, val_loss, val_acc, val_auc, val_map = train_model(model_hp_config,
                                                                log_folder=kwargs["best_fp_config"]["folder_name"])
        avg_vl.append(val_loss)
        avg_va.append(val_acc)
        avg_vu.append(val_auc)
        avg_vm.append(val_map)

        # Test the specified model on the current fold.
        _, _, test_loss, test_acc, test_auc, test_map = test_model(model_hp_config,
                                                                   log_folder=kwargs["best_fp_config"]["folder_name"])
        avg_tl.append(test_loss)
        avg_ta.append(test_acc)
        avg_tu.append(test_auc["micro"])
        avg_tm.append(test_map["micro"])

        print("Fold {} Summary: ".format(fold_ind+1))
        print("-----------------------------")
        print("Validation loss: {:1.3f}".format(val_loss))
        print("Validation accuracy: {:2.2f}%".format(val_acc * 100))
        print("Validation AUC: {:2.2f}%".format(val_auc * 100))
        print("Validation mAP: {:2.2f}%".format(val_map * 100))
        print("-----------------------------")
        print("Test loss: {:1.3f}".format(test_loss))
        print("Test accuracy: {:2.2f}%".format(test_acc * 100))
        print("Test AUC: {:2.2f}%".format(test_auc["micro"] * 100))
        print("Test mAP: {:2.2f}%".format(test_map["micro"] * 100))
        print("-----------------------------")

    print("Overall Summary:")
    print("-----------------------------")
    print("Average Validation loss: {:1.3f}".format(np.mean(avg_vl)))
    print("Average Validation accuracy: {:2.2f}%".format(np.mean(avg_va) * 100))
    print("Average Validation AUC: {:2.2f}%".format(np.mean(avg_vu) * 100))
    print("Average Validation mAP: {:2.2f}%".format(np.mean(avg_vm) * 100))
    print("-----------------------------")
    print("Average Test loss: {:1.3f}".format(np.mean(avg_tl)))
    print("Average Test accuracy: {:2.2f}%".format(np.mean(avg_ta) * 100))
    print("Average Test AUC: {:2.2f}%".format(np.mean(avg_tu) * 100))
    print("Average Test mAP: {:2.2f}%".format(np.mean(avg_tm) * 100))
    print("-----------------------------")




from rdkit.Chem import AllChem
from chem_methods.fingerprints import construct_hsfp

# Done: 0.5%
def qualitative_model_assessment(**kwargs):

    # Load the test datasets.
    # x_dc = pd.read_pickle(kwargs["output_folder"] + "pipeline_dataset/x_dc.pkl").values
    # y_dc = pd.read_pickle(kwargs["output_folder"] + "pipeline_dataset/y_dc.pkl").values
    # info_dc = pd.read_pickle(kwargs["output_folder"] + "pipeline_dataset/info_dc.pkl").values
    # x_ndc = pd.read_pickle(kwargs["output_folder"] + "pipeline_dataset/x_ndc.pkl").values
    # y_ndc = pd.read_pickle(kwargs["output_folder"] + "pipeline_dataset/y_ndc.pkl").values
    # info_ndc = pd.read_pickle(kwargs["output_folder"] + "pipeline_dataset/info_ndc.pkl").values

    # print(len(x_dc))
    # print(len(y_dc))
    # print(len(info_dc))
    # print(len(x_ndc))
    # print(len(y_ndc))
    # print(len(info_ndc))

    novel_mol_examples = [AllChem.MolFromSmiles("O=C1NC(=O)/C(=C\c2cnn3c(NC4CC4)cc(Cl)nc23)S1"),  # Random Test Mol 1
                          AllChem.MolFromSmiles("O=C1NC(=O)/C(=C\c2cnn3c(NC4CC4)cc(Cl)nc23)S1"),  # Random Test Mol 2
                          AllChem.MolFromSmiles("ON(O)C1=C2CN(C3CCC(=O)NC3=O)C(=O)C2=CC=C1"),  # Lenalidomide
                          AllChem.MolFromSmiles("OCC1=C(O)C=CC(=C1)[C@H](O)CNCCCCCCOCCCCC1=CC=CC=C1"),  # Salmeterol
                          AllChem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(O)=O"),  # Aspirin
                          AllChem.MolFromSmiles("CN(C)C(=O)CC1=C(N=C2C=CC(C)=CN12)C1=CC=C(C)C=C1")]  # Zolpidem


    mol = novel_mol_examples[4]
    AllChem.SanitizeMol(mol)

    fps, labs = [], []

    for bond in mol.GetBonds():
        fps.append(construct_hsfp(mol, 2, 1024, from_atoms=[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()], nghb_size=2))
        labs.append(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    fps = np.array(fps)
    labs = np.array(labs)

    # Iterate through the test dataset and predict each substructure.
    model_hp_config = generate_model_hp_config(input_fp_data="hsfp_2_2_1024", oversample=True, fixed_model_ind=0, **kwargs)
    correct_label, predicted_label, _, _, _, _ = test_model(model_hp_config, log_folder="hsfp_2_2_1024", x_test=fps, y_test=labs)

    for rind, r in enumerate(np.round(predicted_label, 2)):
        #if np.argmax(r) > 0:
        print("{} - {}".format(rind, np.round(r, 2)))

    return None


