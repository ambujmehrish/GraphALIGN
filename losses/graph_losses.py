"""Graph-based loss functions for GraphAlign.

Implements all 6 loss functions for graph-based multi-modal alignment:
1. Graph Contrastive Learning (L_graph_NCE)
2. Cross-Modal Graph Fusion Loss (L_fusion)
3. Soft Graph Binding (L_soft_bind)
4. Anchor Distillation (L_anch_distil)
5. Graph Knowledge Distillation (L_graph_distill)
6. Graph Regularization (L_graph_reg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def graph_similarity(R_i: torch.Tensor, R_j: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Frobenius-inner-product-based graph similarity.

    sim(R_i, R_j) = <R_i, R_j>_F / (||R_i||_F * ||R_j||_F)

    Args:
        R_i: Graph matrices [..., N, N].
        R_j: Graph matrices [..., N, N].
        eps: Numerical stability constant.

    Returns:
        Scalar similarity per batch element [...].
    """
    # Flatten the last two dims for inner product
    flat_i = R_i.flatten(start_dim=-2)  # [..., N*N]
    flat_j = R_j.flatten(start_dim=-2)  # [..., N*N]

    inner = (flat_i * flat_j).sum(dim=-1)
    norm_i = torch.norm(flat_i, p=2, dim=-1).clamp(min=eps)
    norm_j = torch.norm(flat_j, p=2, dim=-1).clamp(min=eps)

    return inner / (norm_i * norm_j)


def compute_graph_similarity_matrix(
    R_M1: torch.Tensor, R_M2: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Compute pairwise graph similarity matrix between two sets of graphs.

    Args:
        R_M1: [B1, N, N] relationship matrices from modality 1.
        R_M2: [B2, N, N] relationship matrices from modality 2.
        eps: Numerical stability constant.

    Returns:
        Similarity matrix [B1, B2].
    """
    B1 = R_M1.size(0)
    B2 = R_M2.size(0)

    # Flatten: [B, N*N]
    flat1 = R_M1.flatten(start_dim=1)  # [B1, N*N]
    flat2 = R_M2.flatten(start_dim=1)  # [B2, N*N]

    # L2 normalize
    flat1_norm = F.normalize(flat1, p=2, dim=-1, eps=eps)  # [B1, N*N]
    flat2_norm = F.normalize(flat2, p=2, dim=-1, eps=eps)  # [B2, N*N]

    # Pairwise cosine similarity: [B1, B2]
    sim_matrix = torch.mm(flat1_norm, flat2_norm.t())

    return sim_matrix


# ---------------------------------------------------------------------------
# 1. Graph Contrastive Learning (L_graph_NCE)
# ---------------------------------------------------------------------------

def graph_contrastive_loss(
    R_M1: torch.Tensor,
    R_M2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE loss in graph space.

    Positive pairs are corresponding samples (diagonal of similarity matrix).
    All other pairs are negatives.

    Args:
        R_M1: [B, N, N] relationship matrices from modality 1.
        R_M2: [B, N, N] relationship matrices from modality 2.
        temperature: Temperature scaling for logits.

    Returns:
        Scalar loss.
    """
    B = R_M1.size(0)

    sim_matrix = compute_graph_similarity_matrix(R_M1, R_M2) / temperature

    # Symmetric InfoNCE: M1->M2 and M2->M1
    labels = torch.arange(B, device=R_M1.device)
    loss_12 = F.cross_entropy(sim_matrix, labels)
    loss_21 = F.cross_entropy(sim_matrix.t(), labels)

    return (loss_12 + loss_21) / 2.0


# ---------------------------------------------------------------------------
# 2. Cross-Modal Graph Fusion Classification Loss (L_fusion)
# ---------------------------------------------------------------------------

class FusionClassifier(nn.Module):
    """MLP classifier on flattened fused graph for classification loss."""

    def __init__(self, graph_nodes: int, num_classes: int, hidden_dim: int = 512):
        """
        Args:
            graph_nodes: Number of graph nodes N. Input is N*N.
            num_classes: Number of classification categories.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        input_dim = graph_nodes * graph_nodes
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, G_fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            G_fused: Fused graph [B, N, N].

        Returns:
            Logits [B, num_classes].
        """
        h = G_fused.flatten(start_dim=1)  # [B, N*N]
        return self.mlp(h)


def fusion_classification_loss(
    G_fused: torch.Tensor,
    labels: torch.Tensor,
    classifier: nn.Module,
) -> torch.Tensor:
    """Classification loss on the fused graph.

    Args:
        G_fused: Fused graph [B, N, N].
        labels: Class labels [B] (long tensor).
        classifier: FusionClassifier module.

    Returns:
        Scalar cross-entropy loss.
    """
    logits = classifier(G_fused)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# 3. Soft Graph Binding (L_soft_bind)
# ---------------------------------------------------------------------------

def compute_label_similarity(
    y1: torch.Tensor, y2: torch.Tensor
) -> torch.Tensor:
    """Compute label-based similarity matrix.

    S[i,j] = 1 if y1[i] == y2[j] else 0, then row-normalized.

    Args:
        y1: Labels [B1] (long tensor).
        y2: Labels [B2] (long tensor).

    Returns:
        Soft similarity matrix [B1, B2].
    """
    # [B1, 1] == [1, B2] -> [B1, B2]
    match = (y1.unsqueeze(1) == y2.unsqueeze(0)).float()

    # Normalize rows (avoid division by zero for samples with no match)
    row_sums = match.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return match / row_sums


def soft_graph_binding_loss(
    R_M1: torch.Tensor,
    R_M2: torch.Tensor,
    y_M1: torch.Tensor,
    y_M2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Soft graph binding loss using KL divergence.

    Aligns graph-based cross-modal similarity distributions with
    label-based similarity distributions.

    Args:
        R_M1: [B, N, N] relationship matrices from modality 1.
        R_M2: [B, N, N] relationship matrices from modality 2.
        y_M1: [B] labels for modality 1.
        y_M2: [B] labels for modality 2.
        temperature: Temperature for softmax on graph similarities.

    Returns:
        Scalar loss.
    """
    # Graph-based similarity matrices
    S_graph_12 = compute_graph_similarity_matrix(R_M1, R_M2) / temperature
    S_graph_21 = compute_graph_similarity_matrix(R_M2, R_M1) / temperature

    # Label-based target distributions
    S_label_12 = compute_label_similarity(y_M1, y_M2)
    S_label_21 = compute_label_similarity(y_M2, y_M1)

    # KL divergence: D_KL(target || predicted)
    # F.kl_div expects log-probabilities as input, probabilities as target
    loss_12 = F.kl_div(
        F.log_softmax(S_graph_12, dim=-1),
        S_label_12,
        reduction="batchmean",
    )
    loss_21 = F.kl_div(
        F.log_softmax(S_graph_21, dim=-1),
        S_label_21,
        reduction="batchmean",
    )

    return loss_12 + loss_21


# ---------------------------------------------------------------------------
# 4. Anchor Distillation (L_anch_distil)
# ---------------------------------------------------------------------------

def anchor_distillation_loss(
    R_P: torch.Tensor,
    R_T: torch.Tensor,
    R_I: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Anchor distillation loss for 3-modality alignment.

    Ensures that:
    - Point->Text similarity matches Image->Text similarity
    - Point->Image similarity matches Text->Image similarity

    This anchors the point cloud modality using the stronger image-text pair.

    Args:
        R_P: [B, N, N] point cloud relationship matrices.
        R_T: [B, N, N] text relationship matrices.
        R_I: [B, N, N] image relationship matrices.
        temperature: Temperature for softmax.

    Returns:
        Scalar loss.
    """
    # Point->Text should match Image->Text
    sim_PT = compute_graph_similarity_matrix(R_P, R_T) / temperature
    sim_IT = compute_graph_similarity_matrix(R_I, R_T) / temperature

    loss1 = F.kl_div(
        F.log_softmax(sim_PT, dim=-1),
        F.softmax(sim_IT, dim=-1),
        reduction="batchmean",
    )

    # Point->Image should match Text->Image
    sim_PI = compute_graph_similarity_matrix(R_P, R_I) / temperature
    sim_TI = compute_graph_similarity_matrix(R_T, R_I) / temperature

    loss2 = F.kl_div(
        F.log_softmax(sim_PI, dim=-1),
        F.softmax(sim_TI, dim=-1),
        reduction="batchmean",
    )

    return loss1 + loss2


# ---------------------------------------------------------------------------
# 5. Graph Knowledge Distillation (L_graph_distill)
# ---------------------------------------------------------------------------

def graph_knowledge_distillation_loss(
    R_teacher: torch.Tensor,
    R_student: torch.Tensor,
) -> torch.Tensor:
    """Graph knowledge distillation via Frobenius norm of difference.

    Args:
        R_teacher: [B, N, N] teacher relationship matrices (detached).
        R_student: [B, N, N] student relationship matrices.

    Returns:
        Scalar loss (mean Frobenius norm over batch).
    """
    diff = R_teacher.detach() - R_student
    # Frobenius norm per sample, then average over batch
    loss = torch.norm(diff.flatten(start_dim=1), p=2, dim=-1).mean()
    return loss


# ---------------------------------------------------------------------------
# 6. Graph Regularization (L_graph_reg)
# ---------------------------------------------------------------------------

def _compute_modularity(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Approximate modularity score to encourage block-diagonal structure.

    Uses the degree-corrected modularity Q = (1/2m) * sum_{ij} [A_{ij} - k_i k_j / 2m].
    Higher Q means more block-diagonal structure.

    Args:
        R: [B, N, N] relationship matrices (treated as weighted adjacency).
        eps: Numerical stability constant.

    Returns:
        Mean modularity score (scalar).
    """
    # Use absolute values as edge weights
    A = R.abs()

    # Degree vector: k_i = sum_j A_{ij}
    k = A.sum(dim=-1)  # [B, N]

    # Total weight: 2m = sum_{ij} A_{ij}
    m2 = k.sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=eps)  # [B, 1, 1]

    # Expected connections under null model: k_i * k_j / 2m
    expected = torch.bmm(k.unsqueeze(-1), k.unsqueeze(-2)) / m2  # [B, N, N]

    # Modularity matrix
    Q_matrix = A - expected  # [B, N, N]

    # Trace of Q approximates intra-community connections minus expected
    # Using diagonal sum as a proxy for modularity
    modularity = Q_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [B]

    return modularity.mean()


def graph_regularization_loss(
    R: torch.Tensor,
    lambda_sparse: float = 0.01,
    lambda_cluster: float = 0.01,
    lambda_rank: float = 0.01,
) -> torch.Tensor:
    """Combined graph regularization loss.

    Comprises:
    - L_sparse: L1 norm to encourage sparsity
    - L_cluster: Negative modularity to encourage block-diagonal structure
    - L_rank: Nuclear norm to encourage low-rank structure

    Args:
        R: [B, N, N] relationship matrices.
        lambda_sparse: Weight for sparsity term.
        lambda_cluster: Weight for clustering term.
        lambda_rank: Weight for low-rank term.

    Returns:
        Scalar regularization loss.
    """
    # Sparsity: mean L1 norm
    L_sparse = R.abs().mean()

    # Clustering: negative modularity (minimize this = maximize modularity)
    L_cluster = -_compute_modularity(R)

    # Low-rank: nuclear norm (sum of singular values)
    # torch.linalg.svdvals is efficient for just computing singular values
    sv = torch.linalg.svdvals(R)  # [B, min(N,N)]
    L_rank = sv.sum(dim=-1).mean()

    loss = lambda_sparse * L_sparse + lambda_cluster * L_cluster + lambda_rank * L_rank
    return loss
