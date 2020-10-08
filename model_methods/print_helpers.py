"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 28th, 2020
Description: This file contains necessary functions for monitoring the model training procedure.
"""


def print_epoch_summary(current_epoch, max_epochs, elapsed_time, train_loss, val_loss, val_acc, val_map, val_loss_min,
                        val_acc_max, val_map_max, steps_since_last_improvement):
    """ Prints the summary of the current training epoch. """

    print("Epoch: {}/{}".format(current_epoch + 1, max_epochs))
    print("-------------")
    print("|  Epoch Time: {:3.2f}s".format(elapsed_time),
          "  |  Training Loss: {:1.3f}".format(train_loss),
          "  |  Validation Loss: {:2.3f}".format(val_loss),
          "  |  Validation Accuracy: {:2.2f}%".format(val_acc * 100),
          "  |  Validation mAP: {:2.2f}%".format(val_map * 100),
          "  |  Minimum Loss: {:1.3f}".format(val_loss_min),
          "  |  Maximum Accuracy: {:2.2f}%".format(val_acc_max * 100),
          "  |  Maximum mAP: {:2.2f}%".format(val_map_max * 100),
          "  |  Early Stopping Counter: {:2d}  |\n".format(steps_since_last_improvement))


def print_early_stopping_info(current_epoch, early_stopping_interval):
    """ Prints the activation of the early stopping mechanism. """

    print("Early stopping activated in epoch {:2d}. Checkpoint saved.".format(current_epoch),
          "No improvement in validation accuracy in the last {} steps.".format(early_stopping_interval),
          "Reverting back to epoch {:2d}.\n".format(current_epoch - early_stopping_interval))


def print_training_summary(best_epoch, best_epoch_loss, val_loss_min, val_acc_max, val_auc_max, val_map_max):
    """ Prints the model performance summary of the training process. """

    print("Best epoch: " + str(best_epoch))
    print("Best epoch loss: " + str(best_epoch_loss))
    print("Validation loss minimum: " + str(val_loss_min))
    print("Validation accuracy maximum: " + str(val_acc_max))
    print("Validation AUC maximum: " + str(val_auc_max))
    print("Validation mAP maximum: " + str(val_map_max))


def print_test_summary(elapsed_time, test_loss, test_acc, test_roc_auc, test_map):
    """ Prints the model performance summary on the test set. """

    print("------------------------------------------------------------------------------------------------------------"
          "--------------")
    print("|  Testing Time: {:2.2f}s".format(elapsed_time),
          "  |  Test Loss: {:1.3f}".format(test_loss),
          "  |  Test Accuracy: {:2.2f}%".format(test_acc * 100),
          "  |  Test ROC-AUC: {:2.2f}%".format(test_roc_auc * 100),
          "  |  Test mAP: {:2.2f}%  |".format(test_map * 100))
    print("------------------------------------------------------------------------------------------------------------"
          "--------------")
