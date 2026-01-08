import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_evaluation_results(true_ids, pred_ids, tokenizer, dataset_name=None):
    """
    Plots a normalized confusion matrix and a prediction distribution bar chart.

    Args:
        true_ids (np.array): Ground truth token IDs.
        pred_ids (np.array): Predicted token IDs.
        tokenizer: The tokenizer object (must have convert_ids_to_tokens).
        dataset_name (str): Optional label (e.g., "Test", "Validation") for the title.
    """
    # 1. Map IDs back to Tokens (e.g., 5 -> 'A', 6 -> 'G')
    # Cast numpy integers to python int() to avoid TypeError in tokenizer
    unique_ids = sorted(list(set(true_ids) | set(pred_ids)))
    labels = [tokenizer.convert_ids_to_tokens(int(i)) for i in unique_ids]

    # 2. Confusion Matrix
    cm = confusion_matrix(true_ids, pred_ids, labels=unique_ids)

    # Normalize by row (True Labels) to see percentages
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm.astype('float')), where=row_sums != 0)

    plt.figure(figsize=(14, 6))

    # Dynamic Title Prefix
    prefix = f"{dataset_name} | " if dataset_name else ""

    # --- SUBPLOT 1: Confusion Matrix ---
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f"{prefix}Confusion Matrix (Normalized)")
    plt.ylabel('True Nucleotide')
    plt.xlabel('Predicted Nucleotide')

    # --- SUBPLOT 2: Prediction Distribution ---
    plt.subplot(1, 2, 2)

    # Count occurrences
    unique, counts = np.unique(pred_ids, return_counts=True)
    pred_dict = dict(zip(unique, counts))

    # Align counts with the labels list order
    counts_ordered = [pred_dict.get(i, 0) for i in unique_ids]

    sns.barplot(x=labels, y=counts_ordered, palette="viridis")
    plt.title(f"{prefix}Distribution of Predicted Tokens")
    plt.xlabel("Nucleotide")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()