"""Graph-Aware Feature Pooling Module.

Pools variable-length encoded features from any modality into a fixed number
of graph nodes using attention-based pooling with learnable queries.

Input:  [B, L, D]  - encoded features (batch, sequence length, feature dim)
Output: [B, N, dgraph] - pooled graph node features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAwareFeaturePooling(nn.Module):
    """Attention-based pooling that compresses modality features into graph nodes.

    Uses learnable query vectors to attend over the input sequence and produce
    a fixed-size set of node representations suitable for graph construction.
    """

    def __init__(
        self,
        input_dim: int,
        target_length: int = 32,
        graph_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim: Dimension of input features (D).
            target_length: Number of output graph nodes (N). Default 32.
            graph_dim: Dimension of output graph node features (dgraph). Default 256.
            num_heads: Number of attention heads for multi-head attention.
            dropout: Dropout rate applied to attention weights.
        """
        super().__init__()
        self.input_dim = input_dim
        self.target_length = target_length
        self.graph_dim = graph_dim
        self.num_heads = num_heads

        assert input_dim % num_heads == 0, (
            f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.head_dim = input_dim // num_heads

        # Learnable query vectors: [N, D]
        self.queries = nn.Parameter(torch.empty(target_length, input_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Linear projections for keys and values (queries are learnable params)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Output projection: D -> dgraph
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input features [B, L, D].
            mask: Optional attention mask [B, L]. True for valid positions,
                  False for padding. If None, all positions are valid.

        Returns:
            Pooled features [B, N, dgraph].
        """
        B, L, D = x.shape
        assert D == self.input_dim, (
            f"Input dim mismatch: got {D}, expected {self.input_dim}"
        )

        # Compute keys and values from input: [B, L, D]
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Expand learnable queries for the batch: [N, D] -> [B, N, D]
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Reshape for multi-head attention
        # [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        Q = Q.view(B, self.target_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: [B, num_heads, N, L]
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Apply mask if provided
        if mask is not None:
            # mask: [B, L] -> [B, 1, 1, L]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask_expanded, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values: [B, num_heads, N, head_dim]
        pooled = torch.matmul(attn_weights, V)

        # Reshape back: [B, num_heads, N, head_dim] -> [B, N, D]
        pooled = pooled.transpose(1, 2).contiguous().view(B, self.target_length, D)

        # Project to graph dimension: [B, N, D] -> [B, N, dgraph]
        output = self.output_proj(pooled)

        return output
