"""
Author:      Hasic Haris (Phd Student @ Ishida Lab, Department of Computer Science, Tokyo Institute of Technology)
Created on:  February 28th, 2020
Description: This file contains TensorFlow 1.12 neural network precision, recall and F1 score calculation functions.
"""
import numpy as np

from sklearn import metrics

from data_methods.data_handling import monotonic


# Done: 100%
def generate_confusion_matrices(correct_out, predicted_out, class_labels, out_type="single", normalize=False):
    """ Calculate confusion matrices for one or multiple prediction attempts. """

    # Create a confusion matrix based on the predicted and correct labels for a single or multiple predictions.
    if out_type == "single":
        confusion_matrix = metrics.confusion_matrix(correct_out, predicted_out, labels=class_labels)
    # In the case of multiple predicted outputs, generate and use the average value of all matrices.
    else:
        confusion_matrix = np.mean([metrics.confusion_matrix(correct_out, p_out, labels=class_labels)
                                    for p_out in predicted_out], axis=0)

    # Normalize the generated matrix, if it is specified.
    if normalize:
        confusion_matrix = confusion_matrix.astype("float")*10 / confusion_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix = np.nan_to_num(confusion_matrix, copy=True)
        confusion_matrix = confusion_matrix.astype("int")

    # Return the generated confusion matrix.
    return confusion_matrix


# Done: 100%
def calculate_roc_values(correct_out, predicted_out, class_labels):
    """ Calculate the Receiver Operation Characteristic (ROC) curve and area under that curve. """

    # Create dictionaries for each value to store the class and averaged values.
    false_positive_rate, true_positive_rate, curve_auc_value = dict(), dict(), dict()

    # Calculate the ROC and PRC curve area for each individual class.
    for i in class_labels:
        # Calculate the false positive rates, true positive rates and the area under the ROC curve.
        false_positive_rate["class_{}".format(i)], true_positive_rate["class_{}".format(i)], _ = \
            metrics.roc_curve(correct_out[:, i], predicted_out[:, i])
        curve_auc_value["class_{}".format(i)] = metrics.auc(false_positive_rate["class_{}".format(i)],
                                                            true_positive_rate["class_{}".format(i)])

    # Calculate the micro-average ROC curve area.
    false_positive_rate["micro"], true_positive_rate["micro"], _ = metrics.roc_curve(correct_out.ravel(),
                                                                                     predicted_out.ravel())
    curve_auc_value["micro"] = metrics.auc(false_positive_rate["micro"], true_positive_rate["micro"])

    # Aggregate all false positive rates.
    all_false_positives = np.unique(np.concatenate([false_positive_rate["class_{}".format(i)] for i in class_labels]))

    # Interpolate all ROC and PRC curves at these points.
    mean_true_positives = np.zeros_like(all_false_positives)
    for i in class_labels:
        mean_true_positives += np.interp(all_false_positives, false_positive_rate["class_{}".format(i)],
                                         true_positive_rate["class_{}".format(i)])

    # Average this result and calculate the macro-average ROC and PRC curve area.
    mean_true_positives /= len(class_labels)
    false_positive_rate["macro"] = all_false_positives
    true_positive_rate["macro"] = mean_true_positives
    curve_auc_value["macro"] = metrics.auc(false_positive_rate["macro"], true_positive_rate["macro"])

    # Finally return the calculated values.
    return false_positive_rate, true_positive_rate, curve_auc_value


# Done: 100%
def calculate_prc_values(correct_out, predicted_out, class_labels):
    """ Calculate the Precision-Recall (PR) curve and if possible, area under that curve. """

    # Create dictionaries for each value to store the class and averaged values.
    precision_value, recall_value, curve_auc_value, avg_precision, f1_score = dict(), dict(), dict(), dict(), dict()

    # Calculate the ROC and PRC curve area for each individual class.
    for i in class_labels:
        # Calculate the false positive rates, true positive rates and the area under the PRC curve.
        precision_value["class_{}".format(i)], recall_value["class_{}".format(i)], _ = \
            metrics.precision_recall_curve(correct_out[:, i], predicted_out[:, i])

        avg_precision["class_{}".format(i)] = metrics.average_precision_score(correct_out[:, i], predicted_out[:, i])

    # Calculate the micro-average PR curve area.
    precision_value["micro"], recall_value["micro"], _ = \
        metrics.precision_recall_curve(correct_out.ravel(), predicted_out.ravel())
    avg_precision["micro"] = metrics.average_precision_score(correct_out, predicted_out, average="micro")

    # Aggregate all false positive rates.
    all_precision_values = np.unique(np.concatenate([precision_value["class_{}".format(i)] for i in class_labels]))

    # Interpolate all PR curves at these points.
    mean_recall_values = np.zeros_like(all_precision_values)
    for i in class_labels:
        mean_recall_values += np.interp(all_precision_values, precision_value["class_{}".format(i)],
                                        recall_value["class_{}".format(i)])

    # Average this result and calculate the macro-average PR curve area.
    mean_recall_values /= len(class_labels)
    precision_value["macro"] = all_precision_values
    recall_value["macro"] = mean_recall_values
    avg_precision["macro"] = metrics.average_precision_score(correct_out, predicted_out, average="macro")

    # Calculate the micro, macro and weighted average F1-Score for the Precision-Recall curve.
    correct_out = np.argmax(correct_out, axis=1)
    predicted_out = np.argmax(predicted_out, axis=1)
    f1_score["f1_micro"] = metrics.f1_score(correct_out, predicted_out, average="micro")
    f1_score["f1_macro"] = metrics.f1_score(correct_out, predicted_out, average="macro")
    f1_score["f1_weighted"] = metrics.f1_score(correct_out, predicted_out, average="weighted")

    # Finally return the calculated values.
    return precision_value, recall_value, avg_precision, f1_score


# Done: 100%
def average_roc_prc_values(correct_out, multiple_predicted_outs, class_labels, curve_type="roc"):
    """ Calculate average ROC or PRC curves and area values. """

    # Calculate all of the values for each individual prediction.
    avg_val_1, avg_val_2, avg_val_3, final_val_1, final_val_2, final_val_3 = [], [], [], {}, {}, {}
    dict_labels = ["class_{}".format(i) for i in class_labels]
    dict_labels.extend(["micro", "macro"])

    for predicted_out in multiple_predicted_outs:
        local_val_1, local_val_2, local_val_3 = calculate_roc_values(correct_out, predicted_out, class_labels) \
            if curve_type == "roc" else calculate_prc_values(correct_out, predicted_out, class_labels)

        avg_val_1.append(local_val_1)
        avg_val_2.append(local_val_2)
        avg_val_3.append(local_val_3)

    # Update the average value dictionaries.
    for label in dict_labels:
        final_val_1[label] = np.mean([av_1[label] for av_1 in avg_val_1], axis=0)
        final_val_2[label] = np.mean([av_2[label] for av_2 in avg_val_2], axis=0)
        final_val_3[label] = np.mean([av_3[label] for av_3 in avg_val_3], axis=0)

    # Return the average of the averaged values.
    return final_val_1, final_val_2, final_val_3


# Done: 100%
def generate_prf1_scores(correct_out, predicted_out):
    """ Calculate the Precision, Recall and F1 Scores (overall and per class) for a single prediction. """

    # Convert the predictions into integer values.
    correct_out = np.argmax(correct_out, axis=1)
    predicted_out = np.argmax(predicted_out, axis=1)

    # Calculate the overall "micro", "macro" and "weighted"/"f1" scores for the received predictions without focusing
    # on particular threshold values.
    overall_scores = np.round(np.array([
        metrics.precision_score(correct_out, predicted_out, average="micro"),
        metrics.recall_score(correct_out, predicted_out, average="micro"),
        metrics.f1_score(correct_out, predicted_out, average="micro"),
        metrics.precision_score(correct_out, predicted_out, average="macro"),
        metrics.recall_score(correct_out, predicted_out, average="macro"),
        metrics.f1_score(correct_out, predicted_out, average="macro"),
        metrics.precision_score(correct_out, predicted_out, average="weighted"),
        metrics.recall_score(correct_out, predicted_out, average="weighted"),
        metrics.f1_score(correct_out, predicted_out, average="weighted")
    ]), 3)

    # Calculate those same scores for each individual class.
    cls_precision, cls_recall, cls_f1, cls_count = metrics.precision_recall_fscore_support(correct_out, predicted_out)
    class_scores = np.stack((cls_count, cls_precision, cls_recall, cls_f1), axis=1)

    # Finally, return the calculated scores.
    return overall_scores, class_scores


# Done: 100%
def generate_avg_prf1_scores(correct_out, multiple_predicted_outs):
    """ Calculate average Precision, Recall and F1 Scores (overall and per class) for multiple prediction outputs. """

    # Calculate and collect the scores for each individual prediction.
    avg_overall_score, avg_class_scores = [], []

    for predicted_out in multiple_predicted_outs:
        local_os, local_cs = generate_prf1_scores(correct_out, predicted_out)
        avg_overall_score.append(local_os)
        avg_class_scores.append(local_cs)

    # Return the average value of all predictions.
    return np.mean(avg_overall_score, axis=0), np.mean(avg_overall_score, axis=0)
