"""Dataset loading, symbolic quantization, and Word2Vec feature extraction

for Encrypted Traffic Classification (ISCX Scenario-B).
"""

import os
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from gensim.models import Word2Vec
from scipy.io import arff


def find_or_extract_arff(dataset_path: str, extract_dir: str = "data") -> str:
    """Finds an ARFF file or extracts it if a ZIP archive is provided.

    Args:
        dataset_path: Path to .arff file or .zip archive.
        extract_dir: Destination folder for extraction.

    Returns:
        Path to the valid ARFF file.
    """
    if os.path.isfile(dataset_path) and dataset_path.lower().endswith(".arff"):
        return dataset_path

    if os.path.isfile(dataset_path) and dataset_path.lower().endswith(".zip"):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Look for extracted .arff file
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(".arff"):
                    return os.path.join(root, file)

    # Search in extract_dir
    if os.path.isdir(extract_dir):
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(".arff"):
                    return os.path.join(root, file)

    raise FileNotFoundError(
        f"Could not find an ARFF dataset at {dataset_path} or within {extract_dir}. "
        "Please provide a valid .arff file or .zip archive."
    )


def load_arff_dataframe(file_path: str) -> pd.DataFrame:
    """Loads an ARFF dataset into a pandas DataFrame, decoding byte strings

    and handling invalid numerical values.
    """
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Decode byte strings to regular strings (common in ARFF formats)
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
                )
            except Exception:
                pass

    # Clean infinities and missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def quantize_features(
    df: pd.DataFrame, label_col: str, n_bins: int = 4
) -> pd.DataFrame:
    """Quantizes continuous flow attributes into discrete symbolic tokens.

    Algorithm 1 in Paper: Continuous flow metrics (e.g. packet length, inter-arrival time)
    are discretized into symbolic tokens (e.g., duration_low, duration_high) so they
    can be embedded semantically using Word2Vec.
    """
    quantized_df = pd.DataFrame()
    feature_cols = [c for c in df.columns if c != label_col]

    bin_labels = ["_low", "_medium", "_high", "_very_high"]
    if n_bins != 4:
        bin_labels = [f"_bin{i}" for i in range(n_bins)]

    for col in feature_cols:
        try:
            # Check if column has non-zero variance
            if df[col].nunique() <= 1:
                quantized_df[col] = f"{col}_const"
            else:
                quantized_df[col] = pd.qcut(
                    df[col],
                    q=n_bins,
                    labels=bin_labels,
                    duplicates="drop",
                )
                quantized_df[col] = col + quantized_df[col].astype(str)
        except Exception:
            # Fallback to uniform interval binning if quantile cut fails
            try:
                quantized_df[col] = pd.cut(
                    df[col],
                    bins=n_bins,
                    labels=bin_labels,
                    duplicates="drop",
                )
                quantized_df[col] = col + quantized_df[col].astype(str)
            except Exception:
                quantized_df[col] = f"{col}_const"

    quantized_df[label_col] = df[label_col].values
    return quantized_df


def train_word2vec_model(
    df: pd.DataFrame,
    label_col: str,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 50,
    seed: int = 42,
) -> Word2Vec:
    """Trains a Word2Vec model over flow attribute sentences."""
    feature_cols = [c for c in df.columns if c != label_col]
    sentences = [
        [str(val) for val in row] for row in df[feature_cols].values
    ]

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        sg=0,  # CBOW
        seed=seed,
    )
    return model


def extract_flow_embeddings(
    df: pd.DataFrame,
    w2v_model: Word2Vec,
    label_col: str,
    vector_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates continuous dense flow-level embeddings by mean-pooling token vectors."""
    feature_cols = [c for c in df.columns if c != label_col]
    X, y = [], []

    for _, row in df.iterrows():
        tokens = [str(val) for val in row[feature_cols].values]
        label = row[label_col]
        vectors = [w2v_model.wv[tok] for tok in tokens if tok in w2v_model.wv]

        if vectors:
            avg_vec = np.mean(vectors, axis=0)
        else:
            avg_vec = np.zeros(vector_size, dtype=np.float32)

        X.append(avg_vec)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


class TrafficDataset(Dataset):
    """PyTorch Dataset for flow embeddings and labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

