"""Evaluation utilities for Prototypical Networks:

Metric evaluation, classification report, and confusion matrix heatmap generation.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.models import euclidean_distance


def evaluate_prototypes(
    test_embeddings: torch.Tensor,
    test_labels: np.ndarray,
    prototypes: torch.Tensor,
    class_names: list[str],
    save_cm_path: str = "results/confusion_matrix.png",
) -> dict:
    """Evaluates query/test embeddings against class prototypes using Euclidean distance.

    Args:
        test_embeddings: Tensor of shape [N, D]
        test_labels: Ground truth numerical labels (numpy array)
        prototypes: Tensor of shape [n_classes, D]
        class_names: List of class string names
        save_cm_path: Path to save the confusion matrix plot

    Returns:
        Dictionary of accuracy, classification report, and confusion matrix
    """
    dists = euclidean_distance(test_embeddings, prototypes)
    _, predictions = dists.min(dim=1)
    predictions_np = predictions.cpu().numpy()

    overall_acc = accuracy_score(test_labels, predictions_np)
    report_dict = classification_report(
        test_labels,
        predictions_np,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_str = classification_report(
        test_labels, predictions_np, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(test_labels, predictions_np)

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS - OVERALL ACCURACY: {overall_acc * 100:.2f}%")
    print("=" * 70)

    print("\n--- Per-Class Performance ---")
    for i, name in enumerate(class_names):
        mask = test_labels == i
        if mask.sum() > 0:
            cls_acc = accuracy_score(test_labels[mask], predictions_np[mask])
            print(f"  {name:12s}: {cls_acc * 100:6.2f}% ({mask.sum()} samples)")

    print("\n--- Classification Report ---")
    print(report_str)

    # Plot Confusion Matrix
    if save_cm_path:
        os.makedirs(os.path.dirname(save_cm_path) or ".", exist_ok=True)
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted Application")
        plt.ylabel("True Application")
        plt.title("Confusion Matrix Heatmap (Prototypical ResNet-1D)")
        plt.tight_layout()
        plt.savefig(save_cm_path, dpi=300)
        plt.close()
        print(f"\n[Saved] Confusion matrix figure saved to {save_cm_path}")

    return {
        "overall_accuracy": overall_acc,
        "per_class_report": report_dict,
        "confusion_matrix": cm,
        "predictions": predictions_np,
    }


def analyze_unseen_to_seen_mapping(
    test_embeddings: torch.Tensor,
    y_test_str: np.ndarray,
    seen_prototypes: torch.Tensor,
    seen_classes: list[str],
    unseen_classes: list[str],
):
    """Maps unseen application flows to their nearest seen application prototype

    in the learned geometric embedding space.
    """
    dists = euclidean_distance(test_embeddings, seen_prototypes)
    _, predictions = dists.min(dim=1)
    predictions_np = predictions.cpu().numpy()

    print("\n" + "=" * 70)
    print("UNSEEN -> SEEN APPLICATION PROXIMITY MAPPING")
    print("=" * 70)

    for unseen_app in unseen_classes:
        mask = y_test_str == unseen_app
        if mask.sum() > 0:
            preds_for_app = [seen_classes[p] for p in predictions_np[mask]]
            labels, counts = np.unique(preds_for_app, return_counts=True)
            print(f"\n  Unseen Application: {unseen_app} ({mask.sum()} samples)")
            for name, count in sorted(zip(labels, counts), key=lambda x: -x[1]):
                pct = (count / mask.sum()) * 100
                print(f"    -> Nearest Seen: {name:10s} | {count:4d} flows ({pct:5.1f}%)")

