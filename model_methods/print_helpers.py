"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 28th, 2020
Description: This file contains necessary functions for monitoring the model training procedure.
"""


def print_epoch_summary(curr_epoch, max_epochs, elapsed_time, train_loss, train_acc, val_loss, val_acc, val_map,
                        early_stop_ctr):
    """ Prints the summary of the current training epoch. """

    print("Epoch: {}/{}".format(curr_epoch + 1, max_epochs))
    print("-------------")
    print("Epoch Time: {:3.2f}s".format(elapsed_time),
          "  |  Training Loss: {:1.3f}".format(train_loss),
          "  |  Training Accuracy: {:2.2f}%".format(train_acc * 100),
          "  |  Validation Loss: {:2.3f}".format(val_loss),
          "  |  Validation Accuracy: {:2.2f}%".format(val_acc * 100),
          "  |  Validation mAP: {:2.2f}%".format(val_map * 100),
          "  |  Early Stopping Counter: {:2d}\n".format(early_stop_ctr))


def print_early_stopping_info(current_epoch, early_stopping_interval):
    """ Prints the activation of the early stopping mechanism. """

    print("Early stopping activated in epoch {:2d}. Checkpoint saved.".format(current_epoch),
          "No improvement in validation accuracy in the last {} steps.".format(early_stopping_interval),
          "Reverting back to epoch {:2d}.\n".format(current_epoch - early_stopping_interval))


def print_training_summary(train_time, best_epoch, train_loss, train_acc, val_loss, val_acc, val_roc_auc, val_map):
    """ Prints the model performance summary of the training process. """

    print("Training Time: {:2.2f}s".format(train_time),
          "  |  Best Epoch: {:3d}".format(best_epoch),
          "  |  Training Loss: {:1.3f}".format(train_loss),
          "  |  Training Accuracy: {:2.2f}%".format(train_acc * 100),
          "  |  Best Validation Loss: {:1.3f}".format(val_loss),
          "  |  Best Validation Accuracy: {:2.2f}%".format(val_acc * 100),
          "  |  Best Validation ROC-AUC: {:2.2f}%".format(val_roc_auc * 100),
          "  |  Best Validation mAP: {:2.2f}%\n".format(val_map * 100))


def print_test_summary(test_time, test_loss, test_acc, test_roc_auc, test_map):
    """ Prints the model performance summary on the test set. """

    print("Testing Time: {:2.2f}s".format(test_time),
          "  |  Test Loss: {:1.3f}".format(test_loss),
          "  |  Test Accuracy: {:2.2f}%".format(test_acc * 100),
          "  |  Test ROC-AUC: {:2.2f}%".format(test_roc_auc * 100),
          "  |  Test mAP: {:2.2f}%\n".format(test_map * 100))
