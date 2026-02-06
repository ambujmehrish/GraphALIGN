"""Relationship Graph Construction Module.

Constructs a relationship matrix from pooled graph node features using
cosine similarity between all pairs of nodes.

Input:  [B, N, dgraph] - pooled node features
Output: [B, N, N]      - symmetric relationship matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationshipGraphConstructor(nn.Module):
    """Builds relationship graphs from node features via cosine similarity.

    Properties of the output relationship matrix R:
    - Symmetric: R[i,j] = R[j,i]
    - Normalized to [-1, 1]
    - Diagonal = 1 (self-similarity)
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant for numerical stability in normalization.
        """
        super().__init__()
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Node features [B, N, dgraph].

        Returns:
            Relationship matrix [B, N, N] with cosine similarities.
        """
        # L2-normalize features along the feature dimension
        # [B, N, dgraph]
        features_norm = F.normalize(features, p=2, dim=-1, eps=self.eps)

        # Cosine similarity via batch matrix multiplication
        # [B, N, dgraph] @ [B, dgraph, N] -> [B, N, N]
        R = torch.bmm(features_norm, features_norm.transpose(1, 2))

        # Clamp for numerical stability (should already be in [-1, 1])
        R = R.clamp(-1.0, 1.0)

        return R
