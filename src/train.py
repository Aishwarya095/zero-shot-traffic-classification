"""Main training and evaluation pipeline for Prototypical ResNet-1D traffic classification."""

import argparse
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.dataset import (
    TrafficDataset,
    extract_flow_embeddings,
    find_or_extract_arff,
    load_arff_dataframe,
    quantize_features,
    train_word2vec_model,
)
from src.evaluate import analyze_unseen_to_seen_mapping, evaluate_prototypes
from src.models import (
    ResNet1D,
    compute_prototypes,
    prototypical_loss,
)


def run_pipeline(
    dataset_path: str = "data/Scenario B-ARFF/TimeBasedFeatures-Dataset-120s-AllinOne.arff",
    label_col: str = "class1",
    quantize: bool = True,
    w2v_dim: int = 100,
    resnet_hidden: int = 128,
    resnet_output: int = 64,
    batch_size: int = 32,
    lr: float = 0.001,
    epochs: int = 15,
    n_support: int = 5,
    mode: str = "supervised",
    save_model_path: str = "results/resnet1d_model.pt",
    save_cm_path: str = "results/confusion_matrix.png",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    # 1. Locate and load ARFF dataset
    arff_path = find_or_extract_arff(dataset_path)
    print(f"[*] Loading dataset from: {arff_path}")
    raw_df = load_arff_dataframe(arff_path)
    print(f"[*] Loaded raw dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    if label_col not in raw_df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. Available columns: {list(raw_df.columns)}"
        )

    # 2. Feature Quantization (Discretization into symbolic tokens)
    if quantize:
        print("[*] Performing symbolic feature quantization (Algorithm 1)...")
        df = quantize_features(raw_df, label_col=label_col, n_bins=4)
    else:
        df = raw_df

    seen_apps = ["CHAT", "MAIL", "FT", "P2P"]
    unseen_apps = ["BROWSING", "STREAMING", "VOIP"]
    all_apps = seen_apps + unseen_apps

    df = df[df[label_col].isin(all_apps)].copy()
    print(f"[*] Filtered target categories: {len(df)} samples across {df[label_col].nunique()} classes.")

    # 3. Train/Test Data Split
    if mode == "supervised":
        print("[*] Mode: Supervised Metric Learning (80% Train / 20% Stratified Test Split)")
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            train_size=0.8,
            stratify=df[label_col],
            random_state=42,
        )
    elif mode == "zero_shot":
        print("[*] Mode: True Zero-Shot Generalization (Train on Seen -> Test on Unseen)")
        train_df = df[df[label_col].isin(seen_apps)].copy()
        test_df = df[df[label_col].isin(unseen_apps)].copy()
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'supervised' or 'zero_shot'.")

    print(f"    -> Training samples: {len(train_df)}")
    print(f"    -> Testing samples:  {len(test_df)}")

    # 4. Word2Vec Semantic Feature Representation
    print(f"[*] Training Word2Vec model (vector_size={w2v_dim})...")
    w2v_model = train_word2vec_model(
        train_df,
        label_col=label_col,
        vector_size=w2v_dim,
        window=5,
        epochs=50,
    )

    print("[*] Extracting flow-level embeddings...")
    X_train, y_train_str = extract_flow_embeddings(
        train_df, w2v_model, label_col=label_col, vector_size=w2v_dim
    )
    X_test, y_test_str = extract_flow_embeddings(
        test_df, w2v_model, label_col=label_col, vector_size=w2v_dim
    )

    # 5. Label Encoding
    le = LabelEncoder()
    if mode == "supervised":
        y_train = le.fit_transform(y_train_str)
        y_test = le.transform(y_test_str)
        class_names = list(le.classes_)
    else:
        y_train = le.fit_transform(y_train_str)
        class_names = list(le.classes_)
        # Unseen classes don't share train indices
        y_test = np.zeros(len(y_test_str), dtype=int)

    n_classes = len(np.unique(y_train))
    print(f"[*] Number of training classes: {n_classes} ({class_names})")

    train_dataset = TrafficDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TrafficDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6. Initialize ResNet-1D Encoder
    model = ResNet1D(
        input_dim=w2v_dim,
        hidden_dim=resnet_hidden,
        output_dim=resnet_output,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] ResNet-1D initialized. Trainable parameters: {total_params:,}")

    # 7. Training Loop with Prototypical Loss
    print(f"[*] Commencing training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            embeddings = model(batch_x)
            loss, acc, _ = prototypical_loss(
                embeddings, batch_y, n_support=n_support, n_classes=n_classes
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    Epoch [{epoch+1:2d}/{epochs:2d}] -> Loss: {avg_loss:.4f} | Batch Acc: {avg_acc*100:.2f}%")

    # Save model weights
    os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print(f"[*] Model weights saved to {save_model_path}")

    # 8. Compute Prototypes and Evaluate
    model.eval()
    with torch.no_grad():
        # Compute final prototypes from all training data
        all_train_x = torch.FloatTensor(X_train).to(device)
        all_train_y = torch.LongTensor(y_train).to(device)
        train_embeddings = model(all_train_x)
        prototypes = compute_prototypes(train_embeddings, all_train_y, n_classes)

        # Extract test embeddings
        all_test_x = torch.FloatTensor(X_test).to(device)
        test_embeddings = model(all_test_x)

    if mode == "supervised":
        results = evaluate_prototypes(
            test_embeddings=test_embeddings,
            test_labels=y_test,
            prototypes=prototypes,
            class_names=class_names,
            save_cm_path=save_cm_path,
        )
    else:
        analyze_unseen_to_seen_mapping(
            test_embeddings=test_embeddings,
            y_test_str=y_test_str,
            seen_prototypes=prototypes,
            seen_classes=class_names,
            unseen_classes=unseen_apps,
        )
        results = {"mode": "zero_shot", "samples": len(y_test_str)}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Prototypical ResNet-1D for Traffic Classification"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/Scenario B-ARFF/TimeBasedFeatures-Dataset-120s-AllinOne.arff",
        help="Path to ARFF dataset file or ZIP archive",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--mode",
        type=str,
        default="supervised",
        choices=["supervised", "zero_shot"],
        help="Mode: 'supervised' (80/20 benchmark) or 'zero_shot' (unseen test transfer)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Apply symbolic feature quantization",
    )
    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mode=args.mode,
        quantize=args.quantize,
    )

