"""Graph Expansion Module.

Expands a relationship matrix into multi-scale representations using
element-wise (Hadamard) powers.

Input:  [B, N, N] relationship matrix R
Output: [B, P+1, N, N] expanded matrices [R^0, R^1, ..., R^P]
"""

import torch
import torch.nn as nn


class GraphExpansion(nn.Module):
    """Polynomial expansion of relationship graphs via Hadamard powers.

    Produces multi-scale graph representations:
    - R^0 = Identity matrix (captures node-level information)
    - R^1 = R (original pairwise relationships)
    - R^p = R element-wise raised to power p (emphasizes strong/weak connections)
    """

    def __init__(self, order: int = 3):
        """
        Args:
            order: Maximum power P for expansion. Produces P+1 matrices.
                   Default 3 -> [R^0, R^1, R^2, R^3].
        """
        super().__init__()
        self.order = order

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        """
        Args:
            R: Relationship matrix [B, N, N].

        Returns:
            Expanded graphs [B, P+1, N, N].
        """
        B, N, _ = R.shape

        expanded = []

        # R^0 = Identity matrix, expanded for batch
        identity = torch.eye(N, device=R.device, dtype=R.dtype)
        identity = identity.unsqueeze(0).expand(B, -1, -1)
        expanded.append(identity)

        # R^1 = R (original)
        expanded.append(R)

        # R^p for p = 2, ..., P (element-wise / Hadamard power)
        R_power = R.clone()
        for _ in range(2, self.order + 1):
            R_power = R_power * R  # Element-wise multiplication
            expanded.append(R_power)

        # Stack along new dimension: [B, P+1, N, N]
        return torch.stack(expanded, dim=1)
