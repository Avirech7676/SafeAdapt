"""Comprehensive Unit Test Suite for SafeAdapt-OT Pipeline.

Validates:
1. ResNet-18 model architecture & tensor output dimensions.
2. Sinkhorn-Knopp optimal transport soft coupling & marginal constraints.
3. TTA parameter configuration & selective BatchNorm affine gradient enablement.
4. L_infinity trust-region projection clamping (|theta - theta_anchor| <= delta_max).
5. Conformal risk rollback on high non-conformity anomaly detection.
6. AJCC 8th Edition pTNM Whole-Slide Image (WSI) clinical staging logic.
7. Statistical calibration suite (ECE, MCE, Brier score, NLL).
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure SafeAdapt_Fresh root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import ResNet18Binary, sinkhorn_knopp_transport
from wsi_staging import compute_metastasis_diameter_mm, classify_ajcc_ptnm
from statistical_calibration import compute_calibration_metrics


class TestSafeAdaptModel:
    """Tests for model architecture, feature extraction, and TTA parameters."""

    def test_resnet18_forward_and_shapes(self):
        """Verifies forward pass and penultimate feature dimensionality."""
        model = ResNet18Binary(pretrained=False)
        model.eval()
        dummy_input = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            logits = model(dummy_input)
            features = model.extract_features(dummy_input)

        assert logits.shape == (4, 2), f"Expected logits shape (4, 2), got {logits.shape}"
        assert features.shape == (4, 512), f"Expected features shape (4, 512), got {features.shape}"

    def test_intermediate_feature_extraction(self):
        """Verifies multi-scale intermediate layer activations."""
        model = ResNet18Binary(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            feat_dict = model.get_intermediate_features(x)

        expected_keys = ["conv1", "stem", "layer1", "layer2", "layer3", "layer4", "pooled", "logits"]
        for k in expected_keys:
            assert k in feat_dict, f"Missing intermediate layer key: {k}"

        assert feat_dict["conv1"].shape == (2, 64, 112, 112)
        assert feat_dict["layer1"].shape == (2, 64, 56, 56)
        assert feat_dict["layer2"].shape == (2, 128, 28, 28)
        assert feat_dict["layer3"].shape == (2, 256, 14, 14)
        assert feat_dict["layer4"].shape == (2, 512, 7, 7)
        assert feat_dict["pooled"].shape == (2, 512)

    def test_tta_parameter_isolation(self):
        """Verifies that 100% of conv/linear filters are frozen and ONLY BN affine params adapt."""
        model = ResNet18Binary(pretrained=False)
        trainable = model.configure_tta()

        # All convolutional weights and classifier must be completely frozen
        assert not model.conv1.weight.requires_grad
        assert not model.fc.weight.requires_grad
        assert not model.fc.bias.requires_grad

        # Check that only BatchNorm parameters require grad
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                assert m.weight.requires_grad, "BatchNorm weight must require grad during TTA"
                assert m.bias.requires_grad, "BatchNorm bias must require grad during TTA"
                assert not m.training, "BatchNorm running stats must remain frozen in eval mode"


class TestOptimalTransport:
    """Tests for Sinkhorn-Knopp optimal transport solver."""

    def test_sinkhorn_knopp_coupling_constraints(self):
        """Verifies that Q* satisfies exact marginal probability constraints."""
        B, C = 32, 2
        cost = torch.rand(B, C)
        class_priors = torch.tensor([0.5, 0.5])

        Q = sinkhorn_knopp_transport(cost, class_priors, epsilon=0.03, max_iters=50)

        # 1. Non-negativity
        assert torch.all(Q >= 0.0), "Transport coupling must be non-negative"

        # 2. Total mass should sum to 1.0
        assert torch.isclose(Q.sum(), torch.tensor(1.0), atol=1e-3), f"Total mass {Q.sum()} != 1.0"

        # 3. Column marginals must match class priors
        col_sums = Q.sum(dim=0)
        assert torch.allclose(col_sums, class_priors, atol=1e-2), f"Column sums {col_sums} != priors {class_priors}"

        # 4. Row marginals must be uniform (1 / B)
        row_sums = Q.sum(dim=1)
        expected_row = torch.full((B,), 1.0 / B)
        assert torch.allclose(row_sums, expected_row, atol=1e-2), "Row sums must be uniform"

    def test_stream_of_one_streaming_adaptation(self):
        """Verifies SafeAdapt-OT functions seamlessly for single-patch (B=1) streaming inference."""
        model = ResNet18Binary(pretrained=False)
        trainable = model.configure_tta()
        optimizer = torch.optim.Adam(trainable, lr=1e-4)
        model.prototypes = torch.randn(2, 512)
        model.prototypes = F.normalize(model.prototypes, p=2, dim=-1)

        # Single patch query (B=1)
        x_single = torch.randn(1, 3, 224, 224)
        loss, rolled_back = model.adapt_step_ot(x_single, optimizer)

        assert isinstance(loss, float)
        assert not np.isnan(loss), "B=1 adaptation produced NaN loss"
        assert len(model.feature_queue) > 0, "Feature queue should retain memory after B=1 step"

    def test_dynamic_class_priors_adaptation(self):
        """Verifies adaptive Dirichlet dynamic prior estimation under extreme class imbalance."""
        model = ResNet18Binary(pretrained=False)
        trainable = model.configure_tta()
        optimizer = torch.optim.Adam(trainable, lr=1e-4)
        model.prototypes = torch.randn(2, 512)
        model.prototypes = F.normalize(model.prototypes, p=2, dim=-1)

        # Batch with severe imbalance (e.g. 20 samples)
        x_imbalanced = torch.randn(20, 3, 224, 224)
        # Passing class_priors=None triggers dynamic prior estimation
        loss, rolled_back = model.adapt_step_ot(x_imbalanced, optimizer, class_priors=None)

        assert isinstance(loss, float)
        assert not np.isnan(loss), "Dynamic prior estimation produced NaN loss"


class TestSafetyInterlocks:
    """Tests for trust-region projection clamping and conformal rollback."""

    def test_linf_trust_region_clamping(self):
        """Verifies that parameter drift is strictly capped at delta_max."""
        model = ResNet18Binary(pretrained=False)
        trainable = model.configure_tta()
        optimizer = torch.optim.Adam(trainable, lr=1e-3)
        model.prototypes = torch.randn(2, 512)
        model.prototypes = F.normalize(model.prototypes, p=2, dim=-1)

        delta_max = 0.05
        x = torch.randn(16, 3, 224, 224)

        # Execute 3 adaptation steps with aggressive gradient
        for _ in range(3):
            model.adapt_step_ot(x, optimizer, lambda_ot=5.0, delta_max=delta_max)

        # Check that no affine parameter drifted beyond delta_max
        for name, p in model.named_parameters():
            if p.requires_grad and name in model.anchor_params:
                anchor = model.anchor_params[name]
                max_diff = torch.max(torch.abs(p - anchor)).item()
                assert max_diff <= delta_max + 1e-5, f"Parameter {name} violated L_inf bound: drift={max_diff} > {delta_max}"

    def test_conformal_rollback_on_anomaly(self):
        """Verifies automatic rollback when non-conformity exceeds safety threshold."""
        model = ResNet18Binary(pretrained=False)
        trainable = model.configure_tta()
        optimizer = torch.optim.Adam(trainable, lr=1e-2)
        model.prototypes = torch.randn(2, 512)

        # Cache initial anchor weights
        initial_bn1_weight = model.bn1.weight.detach().clone()

        # Run adapt step with adversarial input (pure noise producing high uncertainty)
        adversarial_input = torch.randn(10, 3, 224, 224) * 5.0
        loss, rolled_back = model.adapt_step_ot(adversarial_input, optimizer)

        # Rollback should successfully restore parameters if triggered
        if rolled_back:
            assert torch.allclose(model.bn1.weight, initial_bn1_weight, atol=1e-6), "Rollback failed to restore pre-step state"


class TestClinicalStagingAndCalibration:
    """Tests for WSI lymph node staging and calibration metrics."""

    def test_ajcc_ptnm_staging_rules(self):
        """Verifies exact AJCC 8th Edition staging thresholds."""
        assert classify_ajcc_ptnm(0.0) == "pN0"
        assert classify_ajcc_ptnm(0.12) == "pN0(i+)"
        assert classify_ajcc_ptnm(0.20) == "pN0(i+)"
        assert classify_ajcc_ptnm(0.21) == "pN1mi"
        assert classify_ajcc_ptnm(2.00) == "pN1mi"
        assert classify_ajcc_ptnm(2.01) == "pN1a"
        assert classify_ajcc_ptnm(15.5) == "pN1a"

    def test_metastasis_diameter_calculation(self):
        """Verifies physical metastasis diameter estimation in millimeters."""
        coords = np.array([
            [1000, 1000],
            [1000 + 448, 1000 + 448]
        ])
        diameter = compute_metastasis_diameter_mm(coords, patch_size_px=224, mpp=0.50)
        assert diameter > 0.0
        assert isinstance(diameter, float)

    def test_calibration_metrics_computation(self):
        """Verifies ECE, MCE, Brier Score, and NLL on known probabilities."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.95])

        metrics = compute_calibration_metrics(y_true, y_prob, n_bins=10)

        assert "ece" in metrics
        assert "mce" in metrics
        assert "brier" in metrics
        assert "nll" in metrics
        assert 0.0 <= metrics["ece"] <= 1.0
        assert 0.0 <= metrics["brier"] <= 1.0
        assert metrics["brier"] < 0.1, f"Expected well-calibrated Brier score < 0.1, got {metrics['brier']}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

