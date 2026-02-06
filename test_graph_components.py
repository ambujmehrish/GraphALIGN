"""Comprehensive tests for all GraphAlign components.

Tests each module independently with dummy data, verifies tensor shapes,
gradient flow through the entire pipeline, and batch size variations.

Usage:
    python test_graph_components.py
"""

import sys
import traceback
from typing import Dict

import torch
import torch.nn as nn

# Ensure the project root is importable
sys.path.insert(0, ".")

from models.graph_pooling import GraphAwareFeaturePooling
from models.relationship_graph import RelationshipGraphConstructor
from models.graph_expansion import GraphExpansion
from models.graph_fusion import CrossModalGraphFusion
from models.graphalign_model import GraphAlignModel, ModalityEncoder
from losses.graph_losses import (
    graph_similarity,
    compute_graph_similarity_matrix,
    graph_contrastive_loss,
    FusionClassifier,
    fusion_classification_loss,
    compute_label_similarity,
    soft_graph_binding_loss,
    anchor_distillation_loss,
    graph_knowledge_distillation_loss,
    graph_regularization_loss,
)


def separator(name: str):
    print(f"\n{'=' * 60}")
    print(f" TEST: {name}")
    print(f"{'=' * 60}")


def check_shape(tensor: torch.Tensor, expected: tuple, name: str):
    assert tensor.shape == expected, (
        f"{name}: expected shape {expected}, got {tensor.shape}"
    )
    print(f"  [OK] {name}: {tensor.shape}")


def check_gradient(tensor: torch.Tensor, name: str):
    assert tensor.grad is not None, f"{name}: gradient is None!"
    assert not torch.isnan(tensor.grad).any(), f"{name}: gradient has NaN!"
    assert not torch.isinf(tensor.grad).any(), f"{name}: gradient has Inf!"
    print(f"  [OK] {name}: gradient flows (norm={tensor.grad.norm().item():.6f})")


passed = 0
failed = 0
total = 0


def run_test(fn):
    global passed, failed, total
    total += 1
    try:
        fn()
        passed += 1
        print(f"  >>> PASSED\n")
    except Exception as e:
        failed += 1
        print(f"  >>> FAILED: {e}")
        traceback.print_exc()
        print()


# ===========================================================================
# 1. Graph-Aware Feature Pooling
# ===========================================================================

def test_graph_pooling_basic():
    separator("Graph Pooling - Basic")
    B, L, D = 4, 100, 512
    N, dgraph = 32, 256

    pooling = GraphAwareFeaturePooling(
        input_dim=D, target_length=N, graph_dim=dgraph, num_heads=8
    )
    x = torch.randn(B, L, D)
    out = pooling(x)

    check_shape(out, (B, N, dgraph), "pooled output")
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")


def test_graph_pooling_with_mask():
    separator("Graph Pooling - With Mask")
    B, L, D = 4, 100, 512
    N, dgraph = 32, 256

    pooling = GraphAwareFeaturePooling(
        input_dim=D, target_length=N, graph_dim=dgraph
    )
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, 80:] = False  # Mask last 20 positions

    out = pooling(x, mask=mask)
    check_shape(out, (B, N, dgraph), "pooled output with mask")
    assert not torch.isnan(out).any(), "Output contains NaN with mask"


def test_graph_pooling_gradient():
    separator("Graph Pooling - Gradient Flow")
    B, L, D = 2, 50, 256
    N, dgraph = 16, 128

    pooling = GraphAwareFeaturePooling(
        input_dim=D, target_length=N, graph_dim=dgraph, num_heads=4
    )
    x = torch.randn(B, L, D, requires_grad=True)
    out = pooling(x)
    loss = out.sum()
    loss.backward()

    check_gradient(x, "input x")
    for name, param in pooling.named_parameters():
        if param.requires_grad:
            check_gradient(param, f"param {name}")


def test_graph_pooling_batch_sizes():
    separator("Graph Pooling - Variable Batch Sizes")
    D, N, dgraph = 256, 16, 128
    pooling = GraphAwareFeaturePooling(
        input_dim=D, target_length=N, graph_dim=dgraph, num_heads=4
    )

    for B in [1, 2, 8, 16]:
        L = 50
        x = torch.randn(B, L, D)
        out = pooling(x)
        check_shape(out, (B, N, dgraph), f"batch_size={B}")


# ===========================================================================
# 2. Relationship Graph Construction
# ===========================================================================

def test_relationship_graph_basic():
    separator("Relationship Graph - Basic")
    B, N, dgraph = 4, 32, 256

    constructor = RelationshipGraphConstructor()
    features = torch.randn(B, N, dgraph)
    R = constructor(features)

    check_shape(R, (B, N, N), "relationship matrix")

    # Check properties
    # Symmetric
    sym_diff = (R - R.transpose(1, 2)).abs().max().item()
    print(f"  Symmetry error: {sym_diff:.10f}")
    assert sym_diff < 1e-5, f"Not symmetric: max diff = {sym_diff}"

    # Range [-1, 1]
    assert R.min().item() >= -1.0 - 1e-6, f"Min below -1: {R.min().item()}"
    assert R.max().item() <= 1.0 + 1e-6, f"Max above 1: {R.max().item()}"

    # Diagonal should be 1 (cosine similarity of a vector with itself)
    diag = R.diagonal(dim1=-2, dim2=-1)
    diag_err = (diag - 1.0).abs().max().item()
    print(f"  Diagonal error from 1.0: {diag_err:.10f}")
    assert diag_err < 1e-5, f"Diagonal not 1: max error = {diag_err}"


def test_relationship_graph_gradient():
    separator("Relationship Graph - Gradient Flow")
    B, N, dgraph = 2, 16, 128

    constructor = RelationshipGraphConstructor()
    features = torch.randn(B, N, dgraph, requires_grad=True)
    R = constructor(features)
    loss = R.sum()
    loss.backward()

    check_gradient(features, "input features")


# ===========================================================================
# 3. Graph Expansion
# ===========================================================================

def test_graph_expansion_basic():
    separator("Graph Expansion - Basic")
    B, N = 4, 32
    P = 3

    expansion = GraphExpansion(order=P)
    R = torch.randn(B, N, N)
    expanded = expansion(R)

    check_shape(expanded, (B, P + 1, N, N), "expanded graphs")

    # R^0 should be identity
    identity = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
    id_err = (expanded[:, 0] - identity).abs().max().item()
    print(f"  R^0 identity error: {id_err:.10f}")
    assert id_err < 1e-6, f"R^0 is not identity: max error = {id_err}"

    # R^1 should be R
    r1_err = (expanded[:, 1] - R).abs().max().item()
    print(f"  R^1 error from input: {r1_err:.10f}")
    assert r1_err < 1e-6, f"R^1 != R: max error = {r1_err}"

    # R^2 should be R * R (element-wise)
    r2_expected = R * R
    r2_err = (expanded[:, 2] - r2_expected).abs().max().item()
    print(f"  R^2 Hadamard error: {r2_err:.10f}")
    assert r2_err < 1e-5, f"R^2 != R*R: max error = {r2_err}"

    # R^3 should be R * R * R (element-wise)
    r3_expected = R * R * R
    r3_err = (expanded[:, 3] - r3_expected).abs().max().item()
    print(f"  R^3 Hadamard error: {r3_err:.10f}")
    assert r3_err < 1e-5, f"R^3 != R*R*R: max error = {r3_err}"


def test_graph_expansion_gradient():
    separator("Graph Expansion - Gradient Flow")
    B, N, P = 2, 16, 3

    expansion = GraphExpansion(order=P)
    R = torch.randn(B, N, N, requires_grad=True)
    expanded = expansion(R)
    loss = expanded.sum()
    loss.backward()

    check_gradient(R, "input R")


# ===========================================================================
# 4. Cross-Modal Graph Fusion
# ===========================================================================

def test_graph_fusion_basic():
    separator("Graph Fusion - Basic")
    B, N = 4, 32
    P = 3

    fusion = CrossModalGraphFusion(order=P)
    G_M1 = torch.randn(B, P + 1, N, N)
    G_M2 = torch.randn(B, P + 1, N, N)

    G_fused = fusion(G_M1, G_M2)
    check_shape(G_fused, (B, N, N), "fused graph")
    assert not torch.isnan(G_fused).any(), "Fused graph has NaN"


def test_graph_fusion_gradient():
    separator("Graph Fusion - Gradient Flow")
    B, N, P = 2, 16, 3

    fusion = CrossModalGraphFusion(order=P)
    G_M1 = torch.randn(B, P + 1, N, N, requires_grad=True)
    G_M2 = torch.randn(B, P + 1, N, N, requires_grad=True)

    G_fused = fusion(G_M1, G_M2)
    loss = G_fused.sum()
    loss.backward()

    check_gradient(G_M1, "G_M1")
    check_gradient(G_M2, "G_M2")
    check_gradient(fusion.fusion_weights, "fusion_weights")


def test_graph_fusion_learnable_weights():
    separator("Graph Fusion - Learnable Weights")
    P = 3
    fusion = CrossModalGraphFusion(order=P)

    # Check weight shape
    check_shape(fusion.fusion_weights, (P + 1, P + 1), "fusion_weights")
    print(f"  Initial weights:\n{fusion.fusion_weights.data}")

    # Verify weights are registered as parameters
    param_names = [n for n, _ in fusion.named_parameters()]
    assert "fusion_weights" in param_names, "fusion_weights not registered"
    print(f"  [OK] fusion_weights is a registered parameter")


# ===========================================================================
# 5. Loss Functions
# ===========================================================================

def test_graph_similarity():
    separator("Graph Similarity")
    B, N = 4, 16

    R1 = torch.randn(B, N, N)
    R2 = torch.randn(B, N, N)

    sim = graph_similarity(R1, R2)
    check_shape(sim, (B,), "similarity")
    assert (sim >= -1.0 - 1e-5).all(), f"Similarity below -1: {sim.min()}"
    assert (sim <= 1.0 + 1e-5).all(), f"Similarity above 1: {sim.max()}"

    # Self-similarity should be 1
    self_sim = graph_similarity(R1, R1)
    self_err = (self_sim - 1.0).abs().max().item()
    print(f"  Self-similarity error from 1.0: {self_err:.10f}")
    assert self_err < 1e-5, f"Self-similarity != 1: {self_err}"


def test_graph_contrastive_loss():
    separator("Graph Contrastive Loss")
    B, N = 8, 16

    R_M1 = torch.randn(B, N, N, requires_grad=True)
    R_M2 = torch.randn(B, N, N, requires_grad=True)

    loss = graph_contrastive_loss(R_M1, R_M2, temperature=0.07)
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Contrastive loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"

    loss.backward()
    check_gradient(R_M1, "R_M1")
    check_gradient(R_M2, "R_M2")


def test_fusion_classification_loss():
    separator("Fusion Classification Loss")
    B, N, C = 4, 16, 10

    classifier = FusionClassifier(graph_nodes=N, num_classes=C)
    G_fused = torch.randn(B, N, N, requires_grad=True)
    labels = torch.randint(0, C, (B,))

    loss = fusion_classification_loss(G_fused, labels, classifier)
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Classification loss should be positive"

    loss.backward()
    check_gradient(G_fused, "G_fused")


def test_soft_graph_binding_loss():
    separator("Soft Graph Binding Loss")
    B, N = 8, 16

    R_M1 = torch.randn(B, N, N, requires_grad=True)
    R_M2 = torch.randn(B, N, N, requires_grad=True)
    y_M1 = torch.randint(0, 5, (B,))
    y_M2 = torch.randint(0, 5, (B,))

    loss = soft_graph_binding_loss(R_M1, R_M2, y_M1, y_M2)
    print(f"  Loss value: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    loss.backward()
    check_gradient(R_M1, "R_M1")
    check_gradient(R_M2, "R_M2")


def test_label_similarity():
    separator("Label Similarity Matrix")
    y1 = torch.tensor([0, 1, 2, 0, 1])
    y2 = torch.tensor([0, 0, 1, 2, 2])

    S = compute_label_similarity(y1, y2)
    check_shape(S, (5, 5), "label similarity")
    print(f"  Label similarity matrix:\n{S}")

    # Each row should sum to 1 (normalized)
    row_sums = S.sum(dim=-1)
    for i, s in enumerate(row_sums):
        assert abs(s.item() - 1.0) < 1e-5, f"Row {i} sum = {s.item()}, expected 1.0"
    print(f"  [OK] All rows sum to 1.0")


def test_anchor_distillation_loss():
    separator("Anchor Distillation Loss")
    B, N = 8, 16

    R_P = torch.randn(B, N, N, requires_grad=True)
    R_T = torch.randn(B, N, N, requires_grad=True)
    R_I = torch.randn(B, N, N, requires_grad=True)

    loss = anchor_distillation_loss(R_P, R_T, R_I)
    print(f"  Loss value: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    loss.backward()
    check_gradient(R_P, "R_P (point)")
    check_gradient(R_T, "R_T (text)")
    check_gradient(R_I, "R_I (image)")


def test_graph_knowledge_distillation():
    separator("Graph Knowledge Distillation Loss")
    B, N = 4, 16

    R_teacher = torch.randn(B, N, N)
    R_student = torch.randn(B, N, N, requires_grad=True)

    loss = graph_knowledge_distillation_loss(R_teacher, R_student)
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() >= 0, "Distillation loss should be non-negative"

    loss.backward()
    check_gradient(R_student, "R_student")

    # Teacher should not have gradients (detached inside the function)
    assert not R_teacher.requires_grad or R_teacher.grad is None, (
        "Teacher should not accumulate gradients"
    )
    print(f"  [OK] Teacher gradients not accumulated")


def test_graph_regularization_loss():
    separator("Graph Regularization Loss")
    B, N = 4, 16

    R = torch.randn(B, N, N, requires_grad=True)
    loss = graph_regularization_loss(R)
    print(f"  Loss value: {loss.item():.4f}")

    loss.backward()
    check_gradient(R, "R")


# ===========================================================================
# 6. Modality Encoder
# ===========================================================================

def test_modality_encoder():
    separator("Modality Encoder")
    B, L = 4, 50
    input_dim, hidden_dim, output_dim = 768, 512, 512

    encoder = ModalityEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        num_heads=8,
    )
    x = torch.randn(B, L, input_dim, requires_grad=True)
    out = encoder(x)

    check_shape(out, (B, L, output_dim), "encoder output")
    loss = out.sum()
    loss.backward()
    check_gradient(x, "encoder input")


# ===========================================================================
# 7. Full Pipeline (End-to-End)
# ===========================================================================

def test_full_pipeline():
    separator("Full Pipeline - End-to-End")
    B = 4
    modalities = ["image", "text", "point"]
    modality_input_dims = {"image": 768, "text": 768, "point": 768}
    seq_lens = {"image": 50, "text": 30, "point": 64}

    model = GraphAlignModel(
        modalities=modalities,
        modality_input_dims=modality_input_dims,
        encoder_hidden_dim=256,
        encoder_output_dim=256,
        encoder_num_layers=2,
        encoder_num_heads=4,
        graph_target_length=16,
        graph_dim=128,
        graph_pooling_heads=4,
        expansion_order=3,
        num_classes=10,
    )

    # Create batch
    batch = {}
    for m in modalities:
        batch[m] = torch.randn(B, seq_lens[m], modality_input_dims[m])
        batch[f"{m}_labels"] = torch.randint(0, 10, (B,))
    batch["labels"] = torch.randint(0, 10, (B,))

    # Forward
    outputs = model(batch)

    # Check all expected outputs exist
    for m in modalities:
        check_shape(outputs[f"encoded_{m}"], (B, seq_lens[m], 256), f"encoded_{m}")
        check_shape(outputs[f"pooled_{m}"], (B, 16, 128), f"pooled_{m}")
        check_shape(outputs[f"graph_{m}"], (B, 16, 16), f"graph_{m}")
        check_shape(outputs[f"expanded_{m}"], (B, 4, 16, 16), f"expanded_{m}")

    # Check fusion outputs
    for i, m1 in enumerate(modalities):
        for m2 in modalities[i + 1:]:
            key = f"fused_{m1}_{m2}"
            assert key in outputs, f"Missing fused output: {key}"
            check_shape(outputs[key], (B, 16, 16), key)

    print(f"\n  Total outputs: {len(outputs)}")
    print(f"  Output keys: {sorted(outputs.keys())}")


def test_full_pipeline_gradient():
    separator("Full Pipeline - Gradient Flow (End-to-End)")
    B = 2
    modalities = ["image", "text"]
    modality_input_dims = {"image": 256, "text": 256}
    seq_lens = {"image": 20, "text": 15}

    model = GraphAlignModel(
        modalities=modalities,
        modality_input_dims=modality_input_dims,
        encoder_hidden_dim=128,
        encoder_output_dim=128,
        encoder_num_layers=1,
        encoder_num_heads=4,
        graph_target_length=8,
        graph_dim=64,
        graph_pooling_heads=4,
        expansion_order=2,
        num_classes=5,
    )

    batch = {}
    for m in modalities:
        batch[m] = torch.randn(B, seq_lens[m], modality_input_dims[m])
        batch[f"{m}_labels"] = torch.randint(0, 5, (B,))
    batch["labels"] = torch.randint(0, 5, (B,))

    outputs = model(batch)

    # Compute a combined loss
    total_loss = torch.tensor(0.0)

    # Graph contrastive
    total_loss = total_loss + graph_contrastive_loss(
        outputs["graph_image"], outputs["graph_text"]
    )

    # Fusion classification
    pair_key = model._pair_key("image", "text")
    total_loss = total_loss + fusion_classification_loss(
        outputs["fused_image_text"], batch["labels"],
        model.fusion_classifiers[pair_key],
    )

    # Soft binding
    total_loss = total_loss + soft_graph_binding_loss(
        outputs["graph_image"], outputs["graph_text"],
        batch["image_labels"], batch["text_labels"],
    )

    # Regularization
    total_loss = total_loss + graph_regularization_loss(outputs["graph_image"])
    total_loss = total_loss + graph_regularization_loss(outputs["graph_text"])

    print(f"  Total loss: {total_loss.item():.4f}")
    total_loss.backward()

    # Check gradients flow to all modules
    grad_modules = {
        "encoder_image": model.encoders["image"],
        "encoder_text": model.encoders["text"],
        "graph_pool_image": model.graph_pool["image"],
        "graph_pool_text": model.graph_pool["text"],
        "fusion": model.fusion_modules[pair_key],
        "classifier": model.fusion_classifiers[pair_key],
    }

    for mod_name, module in grad_modules.items():
        has_grad = False
        for pname, param in module.named_parameters():
            if param.grad is not None and param.grad.norm().item() > 0:
                has_grad = True
                break
        assert has_grad, f"No gradient in module: {mod_name}"
        print(f"  [OK] Gradients flow through: {mod_name}")


def test_full_pipeline_batch_sizes():
    separator("Full Pipeline - Variable Batch Sizes")
    modalities = ["image", "text"]
    modality_input_dims = {"image": 128, "text": 128}
    seq_lens = {"image": 10, "text": 8}

    model = GraphAlignModel(
        modalities=modalities,
        modality_input_dims=modality_input_dims,
        encoder_hidden_dim=64,
        encoder_output_dim=64,
        encoder_num_layers=1,
        encoder_num_heads=4,
        graph_target_length=4,
        graph_dim=32,
        graph_pooling_heads=4,
        expansion_order=2,
        num_classes=3,
    )

    for B in [1, 2, 4, 8]:
        batch = {}
        for m in modalities:
            batch[m] = torch.randn(B, seq_lens[m], modality_input_dims[m])
        outputs = model(batch)
        check_shape(outputs["graph_image"], (B, 4, 4), f"graph_image B={B}")
        check_shape(outputs["fused_image_text"], (B, 4, 4), f"fused B={B}")


# ===========================================================================
# 8. Mixed Precision Compatibility
# ===========================================================================

def test_mixed_precision():
    separator("Mixed Precision (autocast) Compatibility")

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available, testing CPU autocast only")

    B = 2
    modalities = ["image", "text"]
    modality_input_dims = {"image": 128, "text": 128}
    seq_lens = {"image": 10, "text": 8}

    model = GraphAlignModel(
        modalities=modalities,
        modality_input_dims=modality_input_dims,
        encoder_hidden_dim=64,
        encoder_output_dim=64,
        encoder_num_layers=1,
        encoder_num_heads=4,
        graph_target_length=4,
        graph_dim=32,
        graph_pooling_heads=4,
        expansion_order=2,
        num_classes=3,
    )

    batch = {}
    for m in modalities:
        batch[m] = torch.randn(B, seq_lens[m], modality_input_dims[m])
        batch[f"{m}_labels"] = torch.randint(0, 3, (B,))
    batch["labels"] = torch.randint(0, 3, (B,))

    # Test with CPU autocast (simulates AMP behavior)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        outputs = model(batch)
        loss = graph_contrastive_loss(
            outputs["graph_image"], outputs["graph_text"]
        )

    print(f"  Loss value: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN under autocast"
    print(f"  [OK] Mixed precision forward pass successful")


# ===========================================================================
# 9. Teacher Distillation Pipeline
# ===========================================================================

def test_teacher_distillation():
    separator("Teacher Distillation Pipeline")
    B = 2
    modalities = ["image", "text"]
    modality_input_dims = {"image": 128, "text": 128}
    seq_lens = {"image": 10, "text": 8}

    model = GraphAlignModel(
        modalities=modalities,
        modality_input_dims=modality_input_dims,
        encoder_hidden_dim=64,
        encoder_output_dim=64,
        encoder_num_layers=1,
        encoder_num_heads=4,
        graph_target_length=4,
        graph_dim=32,
        graph_pooling_heads=4,
        expansion_order=2,
        num_classes=3,
        use_teacher=True,
    )

    batch = {}
    for m in modalities:
        batch[m] = torch.randn(B, seq_lens[m], modality_input_dims[m])
    outputs = model(batch)

    # Check teacher graphs exist
    for m in modalities:
        key = f"teacher_graph_{m}"
        assert key in outputs, f"Missing teacher graph: {key}"
        check_shape(outputs[key], (B, 4, 4), key)

    # Compute distillation loss
    for m in modalities:
        loss = graph_knowledge_distillation_loss(
            outputs[f"teacher_graph_{m}"], outputs[f"graph_{m}"]
        )
        print(f"  Distillation loss ({m}): {loss.item():.4f}")
        assert not torch.isnan(loss), f"NaN distillation loss for {m}"

    # Verify teacher params are frozen
    for name, param in model.teacher_encoders.named_parameters():
        assert not param.requires_grad, f"Teacher param not frozen: {name}"
    for name, param in model.teacher_graph_pool.named_parameters():
        assert not param.requires_grad, f"Teacher pool param not frozen: {name}"
    print(f"  [OK] Teacher parameters are frozen")


# ===========================================================================
# 10. Integration with train.py loss computation
# ===========================================================================

def test_compute_total_loss():
    separator("Compute Total Loss (train.py integration)")

    # Import here to avoid circular imports at module level
    from train import compute_total_loss

    B = 4
    modalities = ["image", "text", "point"]
    modality_input_dims = {"image": 128, "text": 128, "point": 128}
    seq_lens = {"image": 10, "text": 8, "point": 12}

    model = GraphAlignModel(
        modalities=modalities,
        modality_input_dims=modality_input_dims,
        encoder_hidden_dim=64,
        encoder_output_dim=64,
        encoder_num_layers=1,
        encoder_num_heads=4,
        graph_target_length=4,
        graph_dim=32,
        graph_pooling_heads=4,
        expansion_order=2,
        num_classes=5,
    )

    batch = {}
    for m in modalities:
        batch[m] = torch.randn(B, seq_lens[m], modality_input_dims[m])
        batch[f"{m}_labels"] = torch.randint(0, 5, (B,))
    batch["labels"] = torch.randint(0, 5, (B,))

    outputs = model(batch)

    config = {
        "training": {
            "temperature": 0.07,
            "lambda_1_graph_nce": 1.0,
            "lambda_2_fusion": 1.0,
            "lambda_3_soft_bind": 1.0,
            "lambda_4_anchor": 1.0,
            "lambda_5_teacher": 1.0,
            "lambda_6_reg": 0.01,
            "lambda_sparse": 0.01,
            "lambda_cluster": 0.01,
            "lambda_rank": 0.01,
        }
    }

    total_loss, losses_dict = compute_total_loss(
        model, outputs, batch, modalities, config
    )

    print(f"  Total loss: {total_loss.item():.4f}")
    for k, v in sorted(losses_dict.items()):
        print(f"    {k}: {v:.4f}")

    assert not torch.isnan(total_loss), "Total loss is NaN"
    assert total_loss.item() > 0, "Total loss should be positive"

    # Verify gradients flow
    total_loss.backward()
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.norm().item() > 0:
            grad_count += 1
    print(f"  [OK] {grad_count} parameters received gradients")
    assert grad_count > 0, "No parameters received gradients!"


# ===========================================================================
# Run all tests
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" GraphAlign Component Tests")
    print("=" * 60)

    # Graph Pooling tests
    run_test(test_graph_pooling_basic)
    run_test(test_graph_pooling_with_mask)
    run_test(test_graph_pooling_gradient)
    run_test(test_graph_pooling_batch_sizes)

    # Relationship Graph tests
    run_test(test_relationship_graph_basic)
    run_test(test_relationship_graph_gradient)

    # Graph Expansion tests
    run_test(test_graph_expansion_basic)
    run_test(test_graph_expansion_gradient)

    # Graph Fusion tests
    run_test(test_graph_fusion_basic)
    run_test(test_graph_fusion_gradient)
    run_test(test_graph_fusion_learnable_weights)

    # Loss function tests
    run_test(test_graph_similarity)
    run_test(test_graph_contrastive_loss)
    run_test(test_fusion_classification_loss)
    run_test(test_soft_graph_binding_loss)
    run_test(test_label_similarity)
    run_test(test_anchor_distillation_loss)
    run_test(test_graph_knowledge_distillation)
    run_test(test_graph_regularization_loss)

    # Modality Encoder test
    run_test(test_modality_encoder)

    # Full pipeline tests
    run_test(test_full_pipeline)
    run_test(test_full_pipeline_gradient)
    run_test(test_full_pipeline_batch_sizes)

    # Mixed precision test
    run_test(test_mixed_precision)

    # Teacher distillation test
    run_test(test_teacher_distillation)

    # Integration test with train.py
    run_test(test_compute_total_loss)

    # Summary
    print("\n" + "=" * 60)
    print(f" RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
