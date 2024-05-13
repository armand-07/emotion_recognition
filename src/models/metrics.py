import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torcheval.metrics
from torcheval.metrics.functional import multiclass_f1_score
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score, classification_report, f1_score

import seaborn as sns
import matplotlib.pyplot as plt
import wandb

from src import NUMBER_OF_EMOT, AFFECTNET_CAT_EMOT
import src.visualization.visualize as vis



def calculate_tpr_fpr(binary_targets, binary_pred):
    '''
    Computes the metrics: True Positive Rate (tpr) and the True Negative Rate (fpr) based on the given binary targets and predictions.
    
    Args:
        binary_targets: The list with the real binary classes
        binary_pred: The list with the predicted binary classes
        
    Returns:
        tpr: True Positive Rate based on binary_targets and binary_pred
        fpr: False Positive Rate based on binary_targets and binary_pred
    '''
    
    # Compute the confusion matrix and get TN;FP;FN;TP
    conf_matrix = confusion_matrix(binary_targets, binary_pred)
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TP = conf_matrix[1, 1]
    
    # Compute the tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity or true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity or false positive rate
    return tpr, fpr



def get_coordinates_roc(binary_targets: list, binary_prob :list) -> Tuple [list, list]:
    ''' Computes the ROC Curve coordinates using 20 points between [0,1] to make faster the computation.
    Params:
        binary_targets (list): The list with the real binary classes.
        binary_prob: The list with the probabilities of a class.
    Returns (list):
        tpr_list (list): The list of TPRs for the thresholds.
        fpr_list (list): The list of FPRs for the thresholds.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for threshold in np.linspace(0, 1, 20): # Compute 20 points for the ROC Curve
        binary_pred = binary_prob >= threshold # Predicts the class for all based on the threshold
        tpr, fpr = calculate_tpr_fpr(binary_targets, binary_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list



def plot_roc_curve(tpr: list, fpr: list, scatter:bool = True, ax:plt.axes = None) -> None:
    """Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    Params:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
        ax: The axis where the plot will be drawn (default = None).
    Returns:
        None
    """
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")



def compute_ROC_AUC_OVR(all_targets:torch.Tensor, all_preds_dist:torch.Tensor, labels:list
                        ) -> Tuple[plt.Figure, dict, float]:
    """Compute the ROC AUC score for the model using the One-Versus-Rest strategy and 
    then returns the resulting plot and area under the curve for each class. Only 20 
    points are used to make the computation faster.
    Params:
        - all_targets(torch.Tensor): The tensor with all the targets.
        - all_preds_dist(torch.Tensor): The tensor with all the predictions in distribution form.
        - labels(list): The list with the labels of the classes.
    Returns:
        - fig(plt.Figure): The figure with the plots of the probability distribution and the ROC Curves.
        - roc_auc_per_label(dict): The dictionary with the ROC AUC score for each class.
        - roc_auc_ovr(float): The ROC AUC score with the One-Versus-Rest strategy.
    """
    fig = plt.figure(figsize = (24, 8))
    bins = [i/20 for i in range(20)] + [1]
    roc_auc_per_label = {}
    # Set the model to evaluation
    for i in range(len(labels)):
        # Precompute the binary targets and the probability of the class
        label = labels[i]
        df = pd.DataFrame()
        df['binary_targets'] = torch.eq(all_targets, i).numpy()
        df['prob'] = all_preds_dist[:, i].numpy() # Gets the probability of the class, the rest is 1-prob
        df = df.reset_index(drop = True)
        
        # Plot the probability distribution using OvR strategy
        ax = plt.subplot(2, len(labels), i+1)
        sns.histplot(x = "prob", data = df, hue = 'binary_targets', color = 'b', ax = ax, bins = bins)
        ax.set_title(label)
        ax.legend([f"Class: {label}", "Rest of classes"])
        ax.set_xlim([0, 1])  # Set x-axis limits to [0, 1]
        ax.set_xlabel(f"P(x = {label})")
        
        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, len(labels), len(labels)+i+1)
        tpr, fpr = get_coordinates_roc(df['binary_targets'], df['prob'])
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
        ax_bottom.set_title("ROC Curve OvR")
        
        # Calculates the ROC AUC OvR
        roc_auc_per_label[label] = roc_auc_score(df['binary_targets'], df['prob'])
        plt.tight_layout()
    #ROC AUC OvR as the mean of all of them with no weights for the classes keeping in mind the imbalance of the dataset
    roc_auc_ovr = roc_auc_score(all_targets, all_preds_dist, multi_class = 'ovr')
    return fig, roc_auc_per_label, roc_auc_ovr



def compute_multiclass_precision_recall(all_targets:torch.Tensor, all_preds_labels:torch.Tensor) -> plt.Figure:
    """Compute the precision and recall for the model per class and then returns the resulting plot.
    Params:
        - all_targets (torch.Tensor): The tensor with all the targets.
        - all_preds_labels (torch.Tensor): The tensor with all the predictions in label form.
    Returns:
        - fig (plt.Figure): The figure with the plots of the precision and recall for each class.
    """
    report = classification_report(all_targets, all_preds_labels, output_dict=True, target_names=AFFECTNET_CAT_EMOT)

    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]

    x = np.arange(len(classes))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width/2, precision, width, label='Precision', color='sandybrown')
    ax.bar(x + width/2, recall, width, label='Recall', color='cornflowerblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_ylim([0.0, 1.0])
    ax.grid(axis = 'y', linestyle = '--', linewidth = 0.5, color = 'black')
    ax.set_title('Precision and Recall by emotion class')
    ax.set_xlabel('Emotion classes')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    fig.tight_layout()
    return fig



def compute_multiclass_f1_score(all_targets:torch.Tensor, all_preds_labels:torch.Tensor) -> plt.Figure:
    """Compute the F1 score for each class in a multiclass classification problem. 
    Params:
        - all_targets (torch.Tensor): The tensor with all the targets.
        - all_preds_labels (torch.Tensor): The tensor with all the predictions in label form.
    Returns:
        - fig (plt.Figure): The figure with the plot of the F1 score for each class.
    """
    f1_scores = f1_score(all_targets, all_preds_labels, average=None)
    x = np.arange(len(AFFECTNET_CAT_EMOT))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x, f1_scores, width, color='mediumseagreen')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0.0, 1.0])
    ax.grid(axis = 'y', linestyle = '--', linewidth = 0.5, color = 'black')
    ax.set_title('F1 Score by class')
    ax.set_xticks(x)
    ax.set_xticklabels(AFFECTNET_CAT_EMOT)

    fig.tight_layout()
    return fig



def save_val_wandb_metrics(acc1:torcheval.metrics, acc2:torcheval.metrics, val_loader:DataLoader, batch_size:int, all_targets:torch.Tensor, 
                           all_preds_distrib :torch.Tensor, all_preds_labels: torch.Tensor, global_epoch_loss:float, epoch:int, 
                           run:wandb.run, extra_metrics:bool) -> dict:
    """Save the validation metrics in Weights and Biases for the model.
    Params:
        - acc1 (torch.metrics): The accuracy metric for the model.
        - acc2 (torch.metrics): The top-2 accuracy metric for the model.
        - val_loader (DataLoader): The validation data loader.
        - batch_size (int): The batch size used for the validation.
        - all_targets (torch.Tensor): The tensor with all the targets.
        - all_preds_distrib (torch.Tensor): The tensor with all the predictions in distribution form.
        - all_preds_labels (torch.Tensor): The tensor with all the predictions in label form.
        - global_epoch_loss (float): The global loss for the epoch.
        - epoch (int): The epoch number.
        - run (wandb.run): The Weights and Biases run object.
        - extra_metrics (bool): When True, the precision and recall for each class will be computed and logged.
    Returns:
        - metrics (dict): The dictionary with the metrics to be saved locally when saving the model.
    """
    # Compute the metrics
    acc1 = acc1.compute().item()
    acc2 = acc2.compute().item()
    global_epoch_loss = global_epoch_loss /(len(val_loader) * batch_size) # Mean loss
    f1_score = multiclass_f1_score(input=all_preds_labels, target=all_targets, num_classes=NUMBER_OF_EMOT, average = 'micro').item() # F1-Score
    cohen_kappa = cohen_kappa_score(all_targets, all_preds_labels) # Cohen Kappa coefficient

    # ROC AUC score with OvR strategy
    chart_ROC_AUC, roc_auc_per_label, roc_auc_ovr = compute_ROC_AUC_OVR(all_targets, all_preds_distrib, AFFECTNET_CAT_EMOT)

    # Log the confusion matrix
    conf_matrix = confusion_matrix(all_targets.numpy(), all_preds_labels.numpy(), normalize = 'true')
    chart_conf_matrix = vis.create_conf_matrix(conf_matrix)

    if extra_metrics:
        # Compute the precision and recall for the model per each class
        chart_precision_recall = compute_multiclass_precision_recall(all_targets, all_preds_labels)
        chart_f1_score = compute_multiclass_f1_score(all_targets, all_preds_labels)
        # Log the metrics
        run.log({"Val accuracy per epoch": acc1,
                "Val top-2 accuracy per epoch": acc2,
                "Val mean loss per epoch": global_epoch_loss,
                "Val F1-Score per epoch": f1_score,
                "Val Cohen Kappa coefficient per epoch": cohen_kappa,
                "Plot ROC AUC score with OvR strategy":  wandb.Image(chart_ROC_AUC),
                "Area Under the (ROC AUC) Curve per label": roc_auc_per_label,
                "Area Under the (ROC AUC) Curve OvR": roc_auc_ovr,
                "Plot Precision and Recall by class": wandb.Image(chart_precision_recall),
                "Plot F1-Score by class": wandb.Image(chart_f1_score),
                "Confusion Matrix": chart_conf_matrix
                }, step=epoch+1, commit=True)
        metrics = {
            "Accuracy": acc1,
            "Top-2 Accuracy": acc2,
            "Global Val Mean Loss": global_epoch_loss,
            "F1-Score": f1_score,
            "Cohen Kappa coefficient": cohen_kappa,
            "Area Under the (ROC AUC) Curve OvR": roc_auc_ovr}
    else:
        # Log the metrics
        run.log({"Val accuracy per epoch": acc1,
                "Val top-2 accuracy per epoch": acc2,
                "Val mean loss per epoch": global_epoch_loss,
                "Val F1-Score per epoch": f1_score,
                "Val Cohen Kappa coefficient per epoch": cohen_kappa,
                "Plot ROC AUC score with OvR strategy":  wandb.Image(chart_ROC_AUC),
                "Area Under the (ROC AUC) Curve per label": roc_auc_per_label,
                "Area Under the (ROC AUC) Curve OvR": roc_auc_ovr,
                "Confusion Matrix": chart_conf_matrix
                }, step=epoch+1, commit=True)

        metrics ={
            "Accuracy": acc1,
            "Top-2 Accuracy": acc2,
            "Global Val Mean Loss": global_epoch_loss,
            "F1-Score": f1_score,
            "Cohen Kappa coefficient": cohen_kappa,
            "Area Under the (ROC AUC) Curve OvR": roc_auc_ovr}
        
    return metrics



def save_val_wandb_metrics_dist(acc1:torcheval.metrics, acc2:torcheval.metrics, val_loader:DataLoader, batch_size:int, 
                                all_targets:torch.Tensor, all_preds_distrib: torch.Tensor, all_preds_labels: torch.Tensor,
                                global_epoch_loss: float, global_cosine_sim:torch.Tensor, epoch:int,
                                run: wandb.run, extra_metrics:bool) -> dict:
    """Save the validation metrics in Weights and Biases for the model when distillilation is applied.
    Params:
        - acc1 (torch.metrics): The accuracy metric for the model.
        - acc2 (torch.metrics): The top-2 accuracy metric for the model.
        - val_loader (DataLoader): The validation data loader.
        - batch_size (int): The batch size used for the validation.
        - all_targets (torch.Tensor): The tensor with all the targets.
        - all_preds_distrib (torch.Tensor): The tensor with all the predictions in distribution form.
        - all_preds_labels (torch.Tensor): The tensor with all the predictions in label form.
        - global_epoch_loss (float): The global loss for the epoch.
        - global_cosine_sim (torch.Tensor): The global cosine similarity for the epoch between the output embedding before the head of the class and distillation token.
        - epoch (int): The epoch number.
        - run (wandb.run): The Weights and Biases run object.
        - extra_metrics (bool): When True, the precision and recall for each class will be computed and logged.
    Returns:
        - metrics (dict): The dictionary with the metrics to be saved locally when saving the model.
    """
    # Compute the metrics
    acc1 = acc1.compute().item()
    acc2 = acc2.compute().item()
    global_epoch_loss = global_epoch_loss /(len(val_loader) * batch_size) # Mean loss
    f1_score = multiclass_f1_score(input=all_preds_labels, target=all_targets, num_classes=NUMBER_OF_EMOT, average = 'micro').item() # F1-Score
    cohen_kappa = cohen_kappa_score(all_targets, all_preds_labels) # Cohen Kappa coefficient
    global_cosine_sim = global_cosine_sim / (len(val_loader) * batch_size) # all batches have same size

    # ROC AUC score with OvR strategy
    chart_ROC_AUC, roc_auc_per_label, roc_auc_ovr = compute_ROC_AUC_OVR(all_targets, all_preds_distrib, AFFECTNET_CAT_EMOT)

    # Log the confusion matrix
    conf_matrix = confusion_matrix(all_targets.numpy(), all_preds_labels.numpy(), normalize = 'true')
    chart_conf_matrix = vis.create_conf_matrix(conf_matrix)

    if extra_metrics:
        # Compute the precision and recall for the model per each class
        chart_precision_recall = compute_multiclass_precision_recall(all_targets, all_preds_labels)
        chart_f1_score = compute_multiclass_f1_score(all_targets, all_preds_labels)
        # Log the metrics
        run.log({"Val accuracy per epoch": acc1,
                "Val top-2 accuracy per epoch": acc2,
                "Val mean loss per epoch": global_epoch_loss,
                "Val F1-Score per epoch": f1_score,
                "Val Cohen Kappa coefficient per epoch": cohen_kappa,
                "Plot ROC AUC score with OvR strategy":  wandb.Image(chart_ROC_AUC),
                "Area Under the (ROC AUC) Curve per label": roc_auc_per_label,
                "Area Under the (ROC AUC) Curve OvR": roc_auc_ovr,
                "Plot Precision and Recall by class": wandb.Image(chart_precision_recall),
                "Plot F1-Score by class": wandb.Image(chart_f1_score),
                "Confusion Matrix": chart_conf_matrix
                }, step=epoch+1, commit=True)
        metrics = {
            "Accuracy": acc1,
            "Top-2 Accuracy": acc2,
            "Global Val Mean Loss": global_epoch_loss,
            "F1-Score": f1_score,
            "Cohen Kappa coefficient": cohen_kappa,
            "Area Under the (ROC AUC) Curve OvR": roc_auc_ovr}
    else:
        # Log the metrics
        run.log({"Val accuracy per epoch": acc1,
                "Val top-2 accuracy per epoch": acc2,
                "Val mean loss per epoch": global_epoch_loss,
                "Val F1-Score per epoch": f1_score,
                "Val Cohen Kappa coefficient per epoch": cohen_kappa,
                "Plot ROC AUC score with OvR strategy":  wandb.Image(chart_ROC_AUC),
                "Area Under the (ROC AUC) Curve per label": roc_auc_per_label,
                "Area Under the (ROC AUC) Curve OvR": roc_auc_ovr,
                "Confusion Matrix": chart_conf_matrix
                }, step=epoch+1, commit=True)
        metrics = {
            "Accuracy": acc1,
            "Top-2 Accuracy": acc2,
            "Global Val Mean Loss": global_epoch_loss,
            "F1-Score": f1_score,
            "Cohen Kappa coefficient": cohen_kappa,
            "Area Under the (ROC AUC) Curve OvR": roc_auc_ovr}
    
    return metrics



def save_video_test_wandb_metrics(sum_IoU:float , global_sum_loss:float, total_GTs:float, total_detections:float,
                                    total_inference_time:float, total_inference_time_people:float, total_frames:int,
                                    GT_labels:torch.Tensor, preds_labels:torch.Tensor,
                                    acc1:torcheval.metrics, acc2:torcheval.metrics, run: wandb.run) -> dict:
    """Save the validation metrics in Weights and Biases for the model when distillilation is applied.
    Params:
        
    """
    # Compute the metrics
    mean_IoU = sum_IoU / total_GTs
    global_mean_loss = global_sum_loss / total_detections
    inference_time = total_inference_time / total_frames
    inference_time_people = total_inference_time_people / total_detections

    acc1 = acc1.compute().item()
    acc2 = acc2.compute().item()

    f1_score = multiclass_f1_score(input=preds_labels, target=GT_labels, num_classes=NUMBER_OF_EMOT, average = 'macro').item() # F1-Score
    chart_precision_recall = compute_multiclass_precision_recall(GT_labels, preds_labels)
    chart_f1_score = compute_multiclass_f1_score(GT_labels, preds_labels)

    # Log the confusion matrix
    conf_matrix = confusion_matrix(GT_labels.numpy(), preds_labels.numpy(), normalize = 'true')
    chart_conf_matrix = vis.create_conf_matrix(conf_matrix)


    run.log({"Mean IoU above IoU threshold": mean_IoU,
            "Global Mean Loss": global_mean_loss,
            "Total detections": total_detections,
            "Inferece time per frame": inference_time,
            "Inference time per person and frame": inference_time_people,
            "Accuracy": acc1,
            "Top-2 accuracy": acc2,
            "F1-Score": f1_score,
            "Plot Precision and Recall by class": wandb.Image(chart_precision_recall),
            "Plot F1-Score by class": wandb.Image(chart_f1_score),
            "Confusion Matrix": chart_conf_matrix
            }, commit=True)
