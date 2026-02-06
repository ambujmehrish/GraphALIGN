"""GraphAlign: End-to-End Graph-Based Multi-Modal Alignment Model.

Integrates graph-based alignment on top of a modality encoder backbone.
Each modality is encoded, then graph pooling, relationship graph construction,
graph expansion, and cross-modal graph fusion are applied for alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .graph_pooling import GraphAwareFeaturePooling
from .relationship_graph import RelationshipGraphConstructor
from .graph_expansion import GraphExpansion
from .graph_fusion import CrossModalGraphFusion


class ModalityEncoder(nn.Module):
    """Generic modality encoder using a shared transformer backbone.

    This serves as the backbone encoder for each modality. In a full
    integration with UNIALIGN, this would be replaced by the ViT backbone
    with modality-specific tokenizers and MoE-LoRA layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, L, input_dim].
            mask: Optional padding mask [B, L].

        Returns:
            Encoded features [B, L, output_dim].
        """
        h = self.input_proj(x)

        if mask is not None:
            # TransformerEncoder expects src_key_padding_mask where True = ignore
            src_key_padding_mask = ~mask
            h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        else:
            h = self.transformer(h)

        return self.norm(self.output_proj(h))


class GraphAlignModel(nn.Module):
    """Full GraphAlign model integrating encoders with graph-based alignment.

    Pipeline per modality:
        raw input -> ModalityEncoder -> GraphAwareFeaturePooling
                  -> RelationshipGraphConstructor -> GraphExpansion

    Cross-modal operations:
        expanded_graphs[m1], expanded_graphs[m2] -> CrossModalGraphFusion

    The model produces relationship graphs and fused graphs that are used
    by the loss functions for training.
    """

    def __init__(
        self,
        modalities: List[str],
        modality_input_dims: Dict[str, int],
        encoder_hidden_dim: int = 512,
        encoder_output_dim: int = 512,
        encoder_num_layers: int = 4,
        encoder_num_heads: int = 8,
        encoder_dropout: float = 0.1,
        graph_target_length: int = 32,
        graph_dim: int = 256,
        graph_pooling_heads: int = 8,
        graph_pooling_dropout: float = 0.0,
        expansion_order: int = 3,
        num_classes: int = 0,
        use_teacher: bool = False,
    ):
        """
        Args:
            modalities: List of modality names, e.g. ['image', 'text', 'point'].
            modality_input_dims: Dict mapping modality -> input feature dimension.
            encoder_hidden_dim: Hidden dimension of modality encoders.
            encoder_output_dim: Output dimension of modality encoders (D).
            encoder_num_layers: Number of transformer layers in each encoder.
            encoder_num_heads: Number of attention heads in each encoder.
            encoder_dropout: Dropout for encoder transformer layers.
            graph_target_length: Number of graph nodes (N).
            graph_dim: Graph node feature dimension (dgraph).
            graph_pooling_heads: Number of attention heads for graph pooling.
            graph_pooling_dropout: Dropout for graph pooling attention.
            expansion_order: Polynomial expansion order (P).
            num_classes: Number of classes for fusion classifier (0 = no classifier).
            use_teacher: Whether to include a teacher model for distillation.
        """
        super().__init__()
        self.modalities = modalities
        self.expansion_order = expansion_order
        self.num_classes = num_classes

        # Per-modality encoders
        self.encoders = nn.ModuleDict()
        for m in modalities:
            self.encoders[m] = ModalityEncoder(
                input_dim=modality_input_dims[m],
                hidden_dim=encoder_hidden_dim,
                output_dim=encoder_output_dim,
                num_layers=encoder_num_layers,
                num_heads=encoder_num_heads,
                dropout=encoder_dropout,
            )

        # Per-modality graph pooling
        self.graph_pool = nn.ModuleDict()
        for m in modalities:
            self.graph_pool[m] = GraphAwareFeaturePooling(
                input_dim=encoder_output_dim,
                target_length=graph_target_length,
                graph_dim=graph_dim,
                num_heads=graph_pooling_heads,
                dropout=graph_pooling_dropout,
            )

        # Shared relationship graph constructor
        self.graph_constructor = RelationshipGraphConstructor()

        # Shared graph expansion
        self.graph_expansion = GraphExpansion(order=expansion_order)

        # Cross-modal graph fusion modules (one per modality pair)
        self.fusion_modules = nn.ModuleDict()
        for i, m1 in enumerate(modalities):
            for m2 in modalities[i + 1:]:
                key = f"{m1}_{m2}"
                self.fusion_modules[key] = CrossModalGraphFusion(
                    order=expansion_order
                )

        # Optional fusion classifier for supervised loss
        if num_classes > 0:
            from losses.graph_losses import FusionClassifier
            self.fusion_classifiers = nn.ModuleDict()
            for i, m1 in enumerate(modalities):
                for m2 in modalities[i + 1:]:
                    key = f"{m1}_{m2}"
                    self.fusion_classifiers[key] = FusionClassifier(
                        graph_nodes=graph_target_length,
                        num_classes=num_classes,
                    )

        # Optional teacher model for knowledge distillation
        self.use_teacher = use_teacher
        if use_teacher:
            self.teacher_encoders = nn.ModuleDict()
            self.teacher_graph_pool = nn.ModuleDict()
            for m in modalities:
                self.teacher_encoders[m] = ModalityEncoder(
                    input_dim=modality_input_dims[m],
                    hidden_dim=encoder_hidden_dim,
                    output_dim=encoder_output_dim,
                    num_layers=encoder_num_layers,
                    num_heads=encoder_num_heads,
                    dropout=encoder_dropout,
                )
                self.teacher_graph_pool[m] = GraphAwareFeaturePooling(
                    input_dim=encoder_output_dim,
                    target_length=graph_target_length,
                    graph_dim=graph_dim,
                    num_heads=graph_pooling_heads,
                )
            # Freeze teacher
            for param in self.teacher_encoders.parameters():
                param.requires_grad = False
            for param in self.teacher_graph_pool.parameters():
                param.requires_grad = False

    def encode(
        self,
        x: torch.Tensor,
        modality: str,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode raw input for a given modality.

        Args:
            x: Raw input [B, L, input_dim].
            modality: Modality name.
            mask: Optional padding mask [B, L].

        Returns:
            Encoded features [B, L, encoder_output_dim].
        """
        return self.encoders[modality](x, mask=mask)

    def build_relationship_graph(
        self, pooled_features: torch.Tensor
    ) -> torch.Tensor:
        """Construct relationship graph from pooled features.

        Args:
            pooled_features: [B, N, dgraph].

        Returns:
            Relationship matrix [B, N, N].
        """
        return self.graph_constructor(pooled_features)

    def expand_graph(self, R: torch.Tensor) -> torch.Tensor:
        """Expand relationship graph to multi-scale representation.

        Args:
            R: [B, N, N] relationship matrix.

        Returns:
            Expanded graphs [B, P+1, N, N].
        """
        return self.graph_expansion(R)

    def fuse_graphs(
        self, G_M1: torch.Tensor, G_M2: torch.Tensor, m1: str, m2: str
    ) -> torch.Tensor:
        """Fuse expanded graphs from two modalities.

        Args:
            G_M1: [B, P+1, N, N] expanded graphs from modality 1.
            G_M2: [B, P+1, N, N] expanded graphs from modality 2.
            m1: Name of modality 1.
            m2: Name of modality 2.

        Returns:
            Fused graph [B, N, N].
        """
        # Ensure consistent key ordering
        key = self._pair_key(m1, m2)
        if key.startswith(m2):
            G_M1, G_M2 = G_M2, G_M1
        return self.fusion_modules[key](G_M1, G_M2)

    def teacher_graph(
        self,
        x: torch.Tensor,
        modality: str,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get relationship graph from the teacher model.

        Args:
            x: Raw input [B, L, input_dim].
            modality: Modality name.
            mask: Optional padding mask [B, L].

        Returns:
            Teacher relationship matrix [B, N, N] (detached).
        """
        with torch.no_grad():
            encoded = self.teacher_encoders[modality](x, mask=mask)
            pooled = self.teacher_graph_pool[modality](encoded, mask=mask)
            R = self.graph_constructor(pooled)
        return R.detach()

    def _pair_key(self, m1: str, m2: str) -> str:
        """Get canonical key for a modality pair."""
        for i, ma in enumerate(self.modalities):
            for mb in self.modalities[i + 1:]:
                if (m1 == ma and m2 == mb) or (m1 == mb and m2 == ma):
                    return f"{ma}_{mb}"
        raise ValueError(f"Invalid modality pair: {m1}, {m2}")

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode, pool, build graphs, expand, fuse.

        Args:
            batch: Dict mapping modality name -> input tensor [B, L_m, D_m].
            modality_masks: Optional dict mapping modality -> mask [B, L_m].

        Returns:
            Dict containing:
                - 'encoded_{m}': encoded features per modality [B, L_m, D]
                - 'pooled_{m}': pooled features per modality [B, N, dgraph]
                - 'graph_{m}': relationship matrices per modality [B, N, N]
                - 'expanded_{m}': expanded graphs per modality [B, P+1, N, N]
                - 'fused_{m1}_{m2}': fused graphs per pair [B, N, N]
                - 'teacher_graph_{m}': teacher graphs if use_teacher [B, N, N]
        """
        outputs = {}
        active_modalities = [m for m in self.modalities if m in batch]

        # Stage 1: Encode all modalities
        for m in active_modalities:
            mask = modality_masks.get(m) if modality_masks else None
            encoded = self.encode(batch[m], m, mask=mask)
            outputs[f"encoded_{m}"] = encoded

        # Stage 2: Graph pooling
        for m in active_modalities:
            mask = modality_masks.get(m) if modality_masks else None
            pooled = self.graph_pool[m](outputs[f"encoded_{m}"], mask=mask)
            outputs[f"pooled_{m}"] = pooled

        # Stage 3: Relationship graph construction
        for m in active_modalities:
            R = self.build_relationship_graph(outputs[f"pooled_{m}"])
            outputs[f"graph_{m}"] = R

        # Stage 4: Graph expansion
        for m in active_modalities:
            expanded = self.expand_graph(outputs[f"graph_{m}"])
            outputs[f"expanded_{m}"] = expanded

        # Stage 5: Cross-modal fusion for all active pairs
        for i, m1 in enumerate(active_modalities):
            for m2 in active_modalities[i + 1:]:
                fused = self.fuse_graphs(
                    outputs[f"expanded_{m1}"],
                    outputs[f"expanded_{m2}"],
                    m1, m2,
                )
                outputs[f"fused_{m1}_{m2}"] = fused

        # Stage 6: Teacher graphs (if available)
        if self.use_teacher:
            for m in active_modalities:
                mask = modality_masks.get(m) if modality_masks else None
                outputs[f"teacher_graph_{m}"] = self.teacher_graph(
                    batch[m], m, mask=mask
                )

        return outputs
