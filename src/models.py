"""Deep 1D-ResNet architecture and Prototypical Networks metric learning functions.

Implements flow feature projection into metric space and prototype computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """1D Residual Block with two 1D convolutions, batch normalization,

    ReLU activations, and a skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    """1D Residual Convolutional Network (ResNet-1D) for encoding

    flow-level Word2Vec feature representations into metric embedding space.
    """

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super(ResNet1D, self).__init__()

        self.conv1 = nn.Conv1d(
            1, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(hidden_dim, hidden_dim, num_blocks=2)
        self.layer2 = self._make_layer(
            hidden_dim, hidden_dim * 2, num_blocks=2, stride=2
        )
        self.layer3 = self._make_layer(
            hidden_dim * 2, hidden_dim * 4, num_blocks=2, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        layers = [ResidualBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input [Batch, Dim] -> [Batch, 1, Dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes pairwise squared Euclidean distances between tensor x [N, D]

    and tensor y [M, D]. Output shape: [N, M].
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def compute_prototypes(
    embeddings: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> torch.Tensor:
    """Computes the mean prototype vector for each class in the latent space.

    Args:
        embeddings: Tensor of shape [N, Latent_Dim]
        labels: Tensor of shape [N] with class indices
        n_classes: Total number of classes

    Returns:
        Tensor of shape [n_classes, Latent_Dim]
    """
    latent_dim = embeddings.size(1)
    prototypes = torch.zeros(n_classes, latent_dim, device=embeddings.device)
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            prototypes[i] = embeddings[mask].mean(0)
    return prototypes


def prototypical_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_support: int,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes Prototypical Networks loss with support/query split.

    If a batch does not have sufficient samples per class, falls back
    to nearest centroid classification over all available class samples.
    """
    support_idx = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
    for i in range(n_classes):
        class_mask = labels == i
        class_indices = torch.where(class_mask)[0]
        if len(class_indices) >= n_support:
            support_idx[class_indices[:n_support]] = True

    if support_idx.sum() < n_classes:
        # Fallback to computing prototypes across the batch
        prototypes = compute_prototypes(embeddings, labels, n_classes)
        dists = euclidean_distance(embeddings, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)
        loss = F.nll_loss(log_p_y, labels)
        _, y_pred = log_p_y.max(1)
        acc = (y_pred == labels).float().mean()
        return loss, acc, prototypes

    support_embeddings = embeddings[support_idx]
    support_labels = labels[support_idx]
    query_embeddings = embeddings[~support_idx]
    query_labels = labels[~support_idx]

    prototypes = compute_prototypes(support_embeddings, support_labels, n_classes)
    dists = euclidean_distance(query_embeddings, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, query_labels)
    _, y_pred = log_p_y.max(1)
    acc = (y_pred == query_labels).float().mean()

    return loss, acc, prototypes

