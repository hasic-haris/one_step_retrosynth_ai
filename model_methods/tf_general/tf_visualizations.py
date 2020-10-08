"""
Author:      Haris Hasic, Phd Student @ Ishida Laboratory, Department of Computer Science, Tokyo Institute of Technology
Created on:  February 28th, 2020
Description: This file contains the TensorFlow 1.12 neural network visualization functions for the TensorBoard tool.
"""

import re
import itertools
import tfplot

import numpy as np
import matplotlib.pyplot as plt

from textwrap import wrap
from itertools import cycle

from model_methods.scores import generate_confusion_matrices
from model_methods.scores import calculate_roc_values, calculate_prc_values, average_roc_prc_values


def plot_confusion_matrix(correct_out, predicted_out, class_labels, out_type="single", normalize=False,
                          mode="numerical", return_type="tf_summary", tensor_name="confusion_matrix/img"):
    """ Plots numerical or percentage versions of confusion matrices. """

    # Create a confusion matrix based on the predicted and correct labels.
    conf_mat = generate_confusion_matrices(correct_out, predicted_out, class_labels, out_type, normalize)

    # Set the precision of the numbers displayed in the matrix figure.
    np.set_printoptions(precision=2)

    # Generate the figure.
    figure = plt.Figure(figsize=(4, 4), dpi=320, facecolor="w", edgecolor="k")
    axis = figure.add_subplot(1, 1, 1)
    axis.imshow(conf_mat, cmap="Oranges")

    # Generate the display names for each of the class labels.
    class_label_names = [re.sub(r"([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))", r"\1 ", str(label)) for label in class_labels]
    class_label_names = ['\n'.join(wrap(label_tag, 40)) for label_tag in class_label_names]
    tick_marks = np.arange(len(class_label_names))

    # Configure the axis options.
    axis.set_xlabel("Predicted Label", fontsize=7)
    axis.set_xticks(tick_marks)
    axis.set_xticklabels(class_label_names, fontsize=4, rotation=-90,  ha="center")
    axis.xaxis.set_label_position("bottom")
    axis.xaxis.tick_bottom()
    axis.set_ylabel("Correct Label", fontsize=7)
    axis.set_yticks(tick_marks)
    axis.set_yticklabels(class_label_names, fontsize=4, va="center")
    axis.yaxis.set_label_position("left")
    axis.yaxis.tick_left()

    # Configure the text of the axis
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        if mode == "percentage":
            axis.text(j, i, "{}%".format(np.round(conf_mat[i, j]/conf_mat.sum(axis=0)[j]*100, 1)) if conf_mat[i, j] != 0
                      else '.', horizontalalignment="center", fontsize=6, verticalalignment="center", color="black")
        else:
            axis.text(j, i, format(conf_mat[i, j], 'd') if conf_mat[i, j] != 0 else '.', horizontalalignment="center",
                      fontsize=6, verticalalignment="center", color="black")

    # Set the layout of the figure.
    figure.set_tight_layout("tight")

    # Finally, return the generated matrix in the tf.summary format or as a figure.
    if return_type == "tf_summary":
        return tfplot.figure.to_summary(figure, tag=tensor_name), conf_mat
    else:
        return figure, conf_mat


def plot_roc_curve(correct_out, predicted_out, class_labels, show_values="all", out_type="single",
                   return_type="tf_summary", tensor_name="auc/image"):
    """ Plots the ROC curve. """

    # Generate the ROC curve and calculate individual and averaged AUC values for each class.
    fp_rate, tp_rate, auc_val = calculate_roc_values(correct_out, predicted_out, class_labels) \
        if out_type == "single" else average_roc_prc_values(correct_out, predicted_out, class_labels, curve_type="roc")

    # Generate the figure.
    figure = plt.Figure(figsize=(10, 10))
    axis = figure.add_subplot(1, 1, 1)

    # If specified, plot the AUC values for each individual class.
    if show_values == "all" or show_values == "class":
        colors = cycle(["gray", "firebrick", "sandybrown", "gold", "chartreuse", "darkgreen", "mediumspringgreen",
                        "darkcyan", "deepskyblue", "darkorchid", "palevioletred"])
        for i, color in zip(class_labels, colors):
            axis.plot(fp_rate["class_{}".format(i)], tp_rate["class_{}".format(i)], color=color, linewidth=2,
                      label="Class {0} (ROC-AUC = {1:.2f})".format(i, auc_val["class_{}".format(i)]))

    # If specified, plot the average AUC values of the classes.
    elif show_values == "all" or show_values == "avg":
        axis.plot(fp_rate["micro"], tp_rate["micro"], color="deeppink", linestyle=":", linewidth=2,
                  label="Micro-average (ROC-AUC = {0:.2f})".format(auc_val["micro"]))
        axis.plot(fp_rate["macro"], tp_rate["macro"], color="navy", linestyle=":", linewidth=2,
                  label="Macro-average (ROC-AUC = {0:.2f})".format(auc_val["macro"]))

    # Configure the general axis options.
    axis.set_xlim([-0.05, 1.05])
    axis.set_ylim([-0.05, 1.05])

    # Configure the axis options for the plotting of ROC curves.
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=2)
    figure.legend(loc="lower right", bbox_to_anchor=(0.48, 0.06, 0.5, 0.5), prop={"size": 13})
    axis.set_xlabel("False Positive Rate (1 - Specificity)", fontdict={"size": 13})
    axis.set_ylabel("True Positive Rate (Sensitivity)", fontdict={"size": 13})

    # Set the layout of the figure.
    figure.set_tight_layout("tight")

    # Finally, return the generated curve values in the tf.summary format or as a figure.
    if return_type == "tf_summary":
        return tfplot.figure.to_summary(figure, tag=tensor_name), auc_val
    else:
        return figure, auc_val


def plot_prc_curve(correct_out, predicted_out, class_labels, show_values="all", out_type="single",
                   return_type="tf_summary", tensor_name="auc/image"):
    """ Plots the PRC curve. """

    # Generate the ROC curve and calculate individual and averaged AUC values for each class.
    precision_val, recall_val, avg_precision, _ = calculate_prc_values(correct_out, predicted_out, class_labels) \
        if out_type == "single" else average_roc_prc_values(correct_out, predicted_out, class_labels, curve_type="prc")

    # Generate the figure.
    figure = plt.Figure(figsize=(10, 10))
    axis = figure.add_subplot(1, 1, 1)

    # Generate and plot the Iso-F1 lines.
    f_scores = np.linspace(0.2, 0.8, num=4)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        axis.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        axis.annotate("F1 = {0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    # If specified, plot the AUC values for each individual class.
    if show_values == "all" or show_values == "class":
        colors = cycle(["gray", "firebrick", "sandybrown", "gold", "chartreuse", "darkgreen", "mediumspringgreen",
                        "darkcyan", "deepskyblue", "darkorchid", "palevioletred"])
        for i, color in zip(class_labels, colors):
            axis.plot(precision_val["class_{}".format(i)], recall_val["class_{}".format(i)], color=color, linewidth=2,
                      label="Class {0} (PRC Avg. Precision = {1:.2f})".format(i, avg_precision["class_{}".format(i)]))
    # If specified, plot the average AUC values of the classes.
    elif show_values == "all" or show_values == "avg":
        axis.plot(precision_val["micro"], recall_val["micro"], color="deeppink", linestyle=":", linewidth=2,
                  label="Micro-average (PRC Avg. Precision = {0:.2f})".format(avg_precision["micro"]))
        axis.plot(precision_val["macro"], recall_val["macro"], color="navy", linestyle=":", linewidth=2,
                  label="Macro-average (PRC Avg. Precision = {0:.2f})".format(avg_precision["macro"]))

    # Configure the general axis options.
    axis.set_xlim([-0.05, 1.05])
    axis.set_ylim([-0.05, 1.05])

    # Configure the axis options for the plotting of PR curves.
    axis.plot([0, 1], [0.1, 0.1], color="black", linestyle="--", linewidth=2)
    figure.legend(loc="lower center", bbox_to_anchor=(0.27, 0.06, 0.5, 0.5), prop={"size": 13})
    axis.set_xlabel("Recall", fontdict={"size": 13})
    axis.set_ylabel("Precision", fontdict={"size": 13})

    # Set the layout of the figure.
    figure.set_tight_layout("tight")

    # Finally, return the generated curve values in the tf.summary format or as a figure.
    if return_type == "tf_summary":
        return tfplot.figure.to_summary(figure, tag=tensor_name), avg_precision
    else:
        return figure, avg_precision
