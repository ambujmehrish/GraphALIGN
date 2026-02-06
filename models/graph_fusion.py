"""Cross-Modal Graph Fusion Module.

Fuses two expanded multi-scale graph representations using learnable
bilinear weights over the polynomial orders.

Input:  G_M1 = [B, P+1, N, N], G_M2 = [B, P+1, N, N]
Output: G_fused = [B, N, N]
"""

import torch
import torch.nn as nn


class CrossModalGraphFusion(nn.Module):
    """Fuses expanded graphs from two modalities via learnable bilinear weights.

    G_fused = sum_{p,q} A_{pq} * (R^p_M1 odot R^q_M2)

    where A is a learnable (P+1) x (P+1) weight matrix and odot is
    element-wise (Hadamard) multiplication.
    """

    def __init__(self, order: int = 3):
        """
        Args:
            order: Maximum expansion order P. Weight matrix is (P+1) x (P+1).
        """
        super().__init__()
        self.order = order
        num_terms = order + 1

        # Learnable fusion weights: [(P+1), (P+1)]
        self.fusion_weights = nn.Parameter(torch.empty(num_terms, num_terms))
        # Initialize uniformly so all terms contribute equally at start
        nn.init.constant_(self.fusion_weights, 1.0 / (num_terms * num_terms))

    def forward(
        self, G_M1: torch.Tensor, G_M2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            G_M1: Expanded graphs from modality 1 [B, P+1, N, N].
            G_M2: Expanded graphs from modality 2 [B, P+1, N, N].

        Returns:
            Fused graph [B, N, N].
        """
        B, P1, N, _ = G_M1.shape
        assert P1 == self.order + 1, (
            f"Expected {self.order + 1} expansion orders, got {P1}"
        )
        assert G_M2.shape == G_M1.shape, (
            f"Shape mismatch: G_M1 {G_M1.shape} vs G_M2 {G_M2.shape}"
        )

        # Compute weighted sum of element-wise products
        # G_fused = sum_{p,q} A_{pq} * (G_M1[:, p] * G_M2[:, q])
        #
        # Efficient implementation using einsum:
        # G_M1[:, p] * G_M2[:, q] for all p, q -> [B, P+1, P+1, N, N]
        # Then contract with A[p, q] -> [B, N, N]

        # Compute all pairwise Hadamard products: [B, P+1, P+1, N, N]
        # G_M1: [B, P+1, 1, N, N], G_M2: [B, 1, P+1, N, N]
        pairwise = G_M1.unsqueeze(2) * G_M2.unsqueeze(1)

        # Weight and sum: contract over p and q dimensions
        # fusion_weights: [P+1, P+1] -> [1, P+1, P+1, 1, 1]
        A = self.fusion_weights.view(1, P1, P1, 1, 1)
        G_fused = (A * pairwise).sum(dim=(1, 2))

        return G_fused
