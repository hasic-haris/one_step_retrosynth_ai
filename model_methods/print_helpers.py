"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 28th, 2020
Description: This file contains necessary functions for monitoring the model training procedure.
"""

import numpy as np


def print_epoch_summary(curr_epoch, max_epoch, duration, train_loss, train_acc, val_loss, val_acc, val_roc_auc, val_map,
                        early_stopping_ctr):
    """ Prints the summary of the current training epoch. """

    print("\nEpoch: {}/{}".format(curr_epoch + 1, max_epoch))
    print("-------------")
    print("Epoch Time: {:3.2f}s".format(duration),
          "  |  Training Loss: {:1.3f}".format(train_loss),
          "  |  Training Accuracy: {:2.2f}%".format(train_acc * 100),
          "  |  Validation Loss: {:2.3f}".format(val_loss),
          "  |  Validation Accuracy: {:2.2f}%".format(val_acc * 100),
          "  |  Validation ROC-AUC: {:2.2f}%".format(val_roc_auc * 100),
          "  |  Validation mAP: {:2.2f}%".format(val_map * 100),
          "  |  Early Stopping Counter: {:2d}\n".format(early_stopping_ctr))


def print_early_stopping_info(current_epc, early_stop):
    """ Prints the activation of the early stopping mechanism. """

    print("\nEarly stopping activated in epoch {:2d}. Checkpoint saved.".format(current_epc + 1),
          "No improvement in validation accuracy in the last {} training steps.".format(early_stop),
          "Reverting back to epoch {:2d}.\n".format(current_epc + 1 - early_stop))


def print_training_summary(train_time, best_epc, train_loss, train_acc, val_loss, val_acc, val_roc_auc, val_map):
    """ Prints the model performance summary of the training process. """

    print("\nTraining Time: {:2.2f}s".format(train_time),
          "  |  Best Epoch: {:3d}".format(best_epc),
          "  |  Training Loss: {:1.3f}".format(train_loss),
          "  |  Training Accuracy: {:2.2f}%".format(train_acc * 100),
          "  |  Best Validation Loss: {:1.3f}".format(val_loss),
          "  |  Best Validation Accuracy: {:2.2f}%".format(val_acc * 100),
          "  |  Best Validation ROC-AUC: {:2.2f}%".format(val_roc_auc * 100),
          "  |  Best Validation mAP: {:2.2f}%\n".format(val_map * 100))


def print_test_summary(test_time, test_loss, test_acc, test_roc_auc, test_map):
    """ Prints the model performance summary on the test set. """

    print("\nTesting Time: {:2.2f}s".format(test_time),
          "  |  Test Loss: {:1.3f}".format(test_loss),
          "  |  Test Accuracy: {:2.2f}%".format(test_acc * 100),
          "  |  Test ROC-AUC: {:2.2f}%".format(test_roc_auc * 100),
          "  |  Test mAP: {:2.2f}%\n".format(test_map * 100))


def print_cross_validation_summary(cv_perf):
    """ Prints the model performance across all trained folds. """

    for desc_config in cv_perf.keys():
        print("\nModel trained on the '{}' input configuration.".format(desc_config))
        print("------------------------------------------------")
        print("Best Performing Fold: {:2d}".format(np.argmin([v[0] for v in cv_perf[desc_config].values()]) + 1),
              "  |  Avg. Test Loss: {:1.3f}".format(np.mean([v[0] for v in cv_perf[desc_config].values()])),
              "  |  Avg. Test Accuracy: {:2.2f}%".format(np.mean([v[1] * 100 for v in cv_perf[desc_config].values()])),
              "  |  Avg. Test ROC-AUC: {:2.2f}%".format(np.mean([v[2] * 100 for v in cv_perf[desc_config].values()])),
              "  |  Avg. Test mAP: {:2.2f}%\n".format(np.mean([v[3] * 100 for v in cv_perf[desc_config].values()])))
