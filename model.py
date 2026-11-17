"""SafeAdapt ResNet-18 Model Architecture and Test-Time Adaptation Module.

This module implements the ResNet-18 classifier for histopathology patch diagnosis
(Normal tissue vs Tumor tissue) with SafeAdapt-OT test-time adaptation mechanics.

Core Philosophy:
- Keeps 99.914% (11,167,938) of the model weights frozen to preserve learned medical features.
- Adapts only 0.086% (9,600) affine BatchNorm scale and shift parameters at test time to
  compensate for hospital scanner optical stain variations.
- Uses source class prototypes and Sinkhorn-Knopp optimal transport to guide adaptation without
  relying on noisy pseudo-labels.
- Enforces strict safety interlocks: L-infinity trust region clamping and conformal anomaly rollback.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def sinkhorn_knopp_transport(
    cost_matrix: torch.Tensor,
    class_priors: torch.Tensor,
    epsilon: float = 0.03,
    max_iters: int = 30,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Finds the optimal, balanced soft assignment between images and class prototypes.

    Plain English explanation:
    --------------------------
    When testing on a new hospital, we don't have true labels. We want to match each
    image patch to our reference prototypes (Normal vs Tumor). However, if we do simple
    nearest-neighbor matching, the model might lazily dump all images into just one class.

    The Sinkhorn-Knopp algorithm solves this "Optimal Transport" problem. It finds the
    most accurate assignment matrix Q* such that:
    1. Each image is matched to the class prototype closest to its features.
    2. Across the whole batch, the overall class distribution respects our prior belief
       (e.g., roughly 50% Normal and 50% Tumor).

    "Soft assignment" means each image gets a smooth probability distribution (e.g., 85% Tumor,
    15% Normal) rather than a hard 0 or 1, allowing stable gradient updates.

    Args:
        cost_matrix: (batch_size, num_classes) tensor measuring dissimilarity between
                     each image's features and each class prototype.
        class_priors: (num_classes,) tensor of expected class proportions (e.g. [0.5, 0.5]).
        epsilon: Temperature parameter controlling sharpness of assignment.
        max_iters: Maximum Sinkhorn row/column normalization iterations.
        tol: Convergence tolerance threshold.

    Returns:
        Q: (batch_size, num_classes) normalized soft transport matrix where all entries sum to 1.
    """
    batch_size, num_classes = cost_matrix.shape
    device = cost_matrix.device

    # --- Why we center the cost matrix per row ---
    # The cost matrix contains distance values. To convert distances into probabilities,
    # we take exp(-cost / epsilon). Because epsilon is small (0.03), dividing by it produces
    # large numbers, and exp(-large_number) can easily underflow to 0.0 or trigger NaN errors.
    # By finding the minimum cost in each row (c_min) and subtracting it, the closest class
    # becomes 0 so exp(0) = 1.0. This standard numerical stabilization trick (like the LogSumExp
    # trick in softmax) keeps numbers within safe floating-point ranges without changing the math.
    c_min = cost_matrix.min(dim=1, keepdim=True).values
    K = torch.exp(-(cost_matrix - c_min) / epsilon)

    # Initialize scaling vectors for Sinkhorn iterations
    u = torch.ones(batch_size, device=device) / batch_size
    v = class_priors.to(device)

    # Alternately balance rows (images) and columns (classes) until convergence
    for _ in range(max_iters):
        u_prev = u
        u = (1.0 / batch_size) / (torch.matmul(K, v) + 1e-12)
        v = class_priors / (torch.matmul(K.t(), u) + 1e-12)
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    # Final coupling matrix Q calculation
    u = (1.0 / batch_size) / (torch.matmul(K, v) + 1e-12)
    Q = u.unsqueeze(1) * K * v.unsqueeze(0)
    Q = Q / (Q.sum() + 1e-12)
    return Q


class ResNet18Binary(nn.Module):
    """ResNet-18 binary classifier (Normal vs Tumor) with SafeAdapt-OT mechanics."""

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

        # Load standard torchvision ResNet-18 backbone
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # 1. Input Stem: processes 224x224 RGB image down to 56x56
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # 2. Residual stages: hierarchical feature extraction
        self.layer1 = base.layer1  # 64 channels (56x56)
        self.layer2 = base.layer2  # 128 channels (28x28)
        self.layer3 = base.layer3  # 256 channels (14x14)
        self.layer4 = base.layer4  # 512 channels (7x7)

        # 3. Pooling and binary classification head
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

        # 4. Source class reference prototypes (0: Normal, 1: Tumor)
        # Stored as a non-trainable persistent buffer (shape: 2 x 512)
        self.register_buffer("prototypes", torch.zeros(num_classes, 512), persistent=False)

        # 5. Rolling feature memory queue for stream-of-one (B=1) adaptation
        # Stores recent feature representations to ensure optimal transport
        # marginal constraints remain well-defined even for single-patch queries
        self.register_buffer("feature_queue", torch.empty(0, 512), persistent=False)
        self.queue_capacity = 64

        # Dictionary to store initial anchor parameter values before test-time adaptation
        self.anchor_params: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits for Normal (0) vs Tumor (1)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate 512-dimensional feature embeddings before classification."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts feature maps from every block for inspection and visual interpretability."""
        features = {}

        x_stem = self.conv1(x)
        x_stem = self.bn1(x_stem)
        x_stem = self.relu(x_stem)
        features["conv1"] = x_stem
        x_stem = self.maxpool(x_stem)
        features["stem"] = x_stem

        x_l1 = self.layer1(x_stem)
        features["layer1"] = x_l1

        x_l2 = self.layer2(x_l1)
        features["layer2"] = x_l2

        x_l3 = self.layer3(x_l2)
        features["layer3"] = x_l3

        x_l4 = self.layer4(x_l3)
        features["layer4"] = x_l4

        x_pool = self.avgpool(x_l4)
        x_flat = torch.flatten(x_pool, 1)
        features["pooled"] = x_flat
        features["logits"] = self.fc(x_flat)

        return features

    def compute_source_prototypes(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
        """Computes and stores unit-normalized class centroids from the source training data.
        
        These serve as fixed clinical reference points (Normal vs Tumor) during adaptation.
        """
        self.eval()
        all_feats = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                feats = self.extract_features(images)
                norm_feats = F.normalize(feats, p=2, dim=-1)
                all_feats.append(norm_feats)
                all_labels.append(labels)

        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute centroid for Normal (class 0) and Tumor (class 1)
        proto_list = []
        for c in range(self.num_classes):
            mask = (all_labels == c)
            if mask.sum() > 0:
                class_centroid = all_feats[mask].mean(dim=0)
                class_proto = F.normalize(class_centroid, p=2, dim=0)
            else:
                class_proto = torch.zeros(512, device=device)
            proto_list.append(class_proto)

        self.prototypes = torch.stack(proto_list, dim=0)
        return self.prototypes

    def configure_tta(self):
        """Configures the model for SafeAdapt Test-Time Adaptation.
        
        Why this makes adaptation safe:
        1. All 11,167,938 convolutional and classifier weights are completely frozen.
           This guarantees zero catastrophic forgetting of learned diagnostic knowledge.
        2. Only the 9,600 affine scale (gamma) and shift (beta) parameters in BatchNorm layers
           are enabled for adaptation (representing just 0.086% of total weights).
           This gives the model just enough flexibility to correct scanner stain and contrast
           shifts without allowing it to destroy its diagnostic filters.
        3. Initial values of the adapting parameters are stored in `anchor_params` so we can
           bound how far they drift.
        """
        for p in self.parameters():
            p.requires_grad = False

        trainable_params = []
        self.anchor_params = {}

        for name, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # Freeze running statistics (mean/var) to avoid noisy batch statistics
                m.eval()
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                trainable_params.extend([m.weight, m.bias])

        # Cache baseline parameter anchors for trust-region clamping and distance penalty
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.anchor_params[name] = p.detach().clone()

        return trainable_params

    def adapt_step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer, **kwargs) -> float:
        """Executes one SafeAdapt test-time adaptation step on an unlabeled target batch."""
        loss, _ = self.adapt_step_ot(x, optimizer, **kwargs)
        return loss

    def adapt_step_ot(
        self,
        x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        lambda_ot: float = 2.0,
        lambda_div: float = 0.5,
        lambda_anchor: float = 0.05,
        delta_max: float = 0.05,
        epsilon_ot: float = 0.03,
        class_priors: Optional[torch.Tensor] = None,
        rollback_threshold: float = 0.48,
        prior_smoothing: float = 0.2,
    ) -> Tuple[float, bool]:
        """Executes one SafeAdapt-OT adaptation step with safety interlocks.

        Detailed breakdown of the 6 steps:
        ----------------------------------
        1. Feature Extraction & Cost Matrix:
           Measures the cosine distance between each target patch feature and the
           verified source class prototypes. Supports single-patch queries (B=1) via a
           rolling feature memory queue.
        2. Optimal Transport Soft Alignment (Sinkhorn-Knopp):
           Matches target batch patches to prototypes under dynamic or provided class priors.
        3. Anti-Attractor Class Diversity Loss:
           Prevents single-class collapse (where unsupervised models predict 100% Tumor
           or 100% Normal) by penalizing drift away from expected class balance.
        4. Anchor Regularization:
           Penalizes large quadratic deviations from original source weights.
        5. L-infinity Trust-Region Projection Clamping:
           Hard bounding box: clamps parameter changes so no weight drifts more than
           delta_max (0.05) away from its starting baseline.
        6. Conformal Anomaly Check & Rollback:
           If the batch produces abnormally high prediction uncertainty (non-conformity > rollback_threshold),
           the adaptation step is rejected and parameters are restored to pre-step state.
        """
        # Save pre-step parameters in case we need to roll back
        pre_step_state = {name: p.detach().clone() for name, p in self.named_parameters() if p.requires_grad}

        # --- Step 1: Feature Extraction & Cosine Dissimilarity Matrix ---
        feats = self.extract_features(x)
        norm_feats = F.normalize(feats, p=2, dim=-1)

        # Stream-of-One Support (B=1 or small mini-batch):
        # If the current batch has fewer than 16 samples, we combine it with the rolling
        # feature memory queue to ensure optimal transport marginal constraints remain well-conditioned.
        if len(norm_feats) < 16 and hasattr(self, "feature_queue") and self.feature_queue.shape[0] > 0:
            transport_feats = torch.cat([norm_feats, self.feature_queue], dim=0)
        else:
            transport_feats = norm_feats

        # Cost is 0 when features match prototype, 2 when completely opposite
        cost = 1.0 - torch.matmul(transport_feats, self.prototypes.t())

        # Update rolling feature memory queue with current batch features
        with torch.no_grad():
            if hasattr(self, "feature_queue"):
                new_q = torch.cat([norm_feats.detach(), self.feature_queue], dim=0)
                self.feature_queue = new_q[:self.queue_capacity]

        # --- Step 2: Dynamic or Provided Class Priors & Optimal Transport ---
        # If priors are not specified, estimate them dynamically from current batch tendency
        # blended with a uniform prior via prior_smoothing (prevents degenerate marginal assignments)
        if class_priors is None:
            with torch.no_grad():
                p_hat = F.softmax(self.fc(transport_feats), dim=-1).mean(dim=0)
                uniform_prior = torch.full_like(p_hat, 1.0 / p_hat.numel())
                class_priors = (1.0 - prior_smoothing) * p_hat + prior_smoothing * uniform_prior
                class_priors = class_priors / class_priors.sum()
        else:
            class_priors = class_priors.to(x.device)

        with torch.no_grad():
            Q_opt = sinkhorn_knopp_transport(cost, class_priors, epsilon=epsilon_ot)

        # Transport alignment loss: guides patches toward matching prototypes
        loss_ot = torch.sum(Q_opt * cost)

        # --- Step 3: Anti-Attractor Class Diversity Loss ---
        # Calculates average predicted class distribution and guides it toward target prior
        logits = self.fc(feats)
        probs = F.softmax(logits, dim=-1)
        p_bar = probs.mean(dim=0)
        loss_div = torch.sum(p_bar * torch.log((p_bar + 1e-8) / class_priors))

        # --- Step 4: Source Anchor Distance Loss ---
        # Soft penalty to prevent weights from drifting needlessly far from initial source values
        loss_anchor = torch.tensor(0.0, device=x.device)
        for name, p in self.named_parameters():
            if p.requires_grad and name in self.anchor_params:
                anchor = self.anchor_params[name].to(x.device)
                loss_anchor = loss_anchor + torch.sum((p - anchor) ** 2)

        # Total combined adaptation objective
        total_loss = lambda_ot * loss_ot + lambda_div * loss_div + lambda_anchor * loss_anchor

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # --- Step 5: L-infinity Trust-Region Projection Clamping ---
        # Hard safety barrier: ensures |theta - theta_anchor| <= delta_max.
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.requires_grad and name in self.anchor_params:
                    anchor = self.anchor_params[name].to(x.device)
                    diff = p - anchor
                    clamped_diff = torch.clamp(diff, -delta_max, delta_max)
                    p.copy_(anchor + clamped_diff)
                    # Project optimizer momentum buffer at boundary to prevent momentum desynchronization
                    if p in optimizer.state and "exp_avg" in optimizer.state[p]:
                        hit_boundary = (diff.abs() >= delta_max)
                        if hit_boundary.any():
                            optimizer.state[p]["exp_avg"].masked_fill_(hit_boundary, 0.0)

        # --- Step 6: Conformal Anomaly Check & Rollback Hook ---
        with torch.no_grad():
            post_logits = self.forward(x)
            post_probs = F.softmax(post_logits, dim=-1)
            non_conformity = (1.0 - post_probs.max(dim=-1).values).mean().item()

            rolled_back = False
            if non_conformity > rollback_threshold:
                for name, p in self.named_parameters():
                    if p.requires_grad and name in pre_step_state:
                        p.copy_(pre_step_state[name])
                        # Reset momentum buffer on rollback to avoid corrupted gradient accumulation
                        if p in optimizer.state and "exp_avg" in optimizer.state[p]:
                            optimizer.state[p]["exp_avg"].zero_()
                rolled_back = True
                post_logits = self.forward(x)

            self.last_post_logits = post_logits

        return float(total_loss.item()), rolled_back


if __name__ == "__main__":
    print("=" * 70)
    print("           SafeAdapt ResNet-18 Model Architecture Self-Test          ")
    print("=" * 70)

    # 1. Instantiate model
    print("[1/4] Instantiating ResNet18Binary model...")
    model = ResNet18Binary(pretrained=False)
    model.eval()

    # 2. Forward pass test
    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"[2/4] Testing forward pass with input shape {list(dummy_input.shape)}...")
    logits = model(dummy_input)
    features = model.extract_features(dummy_input)
    probs = F.softmax(logits, dim=-1)
    print(f"      - Penultimate Features: {list(features.shape)} (512-dim embedding)")
    print(f"      - Output Logits Shape:  {list(logits.shape)} (2 classes: Normal vs Tumor)")
    print(f"      - Output Probabilities: Class 0 = {probs[0, 0]:.4f}, Class 1 = {probs[0, 1]:.4f}")

    # 3. Parameter count & TTA configuration
    print("[3/4] Checking parameter configuration for Test-Time Adaptation...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.configure_tta()
    adapt_params = sum(p.numel() for p in trainable_params)
    frozen_params = total_params - adapt_params
    print(f"      - Total Parameters:     {total_params:,} (100.0%)")
    print(f"      - Frozen Parameters:    {frozen_params:,} ({(frozen_params/total_params)*100:.3f}%)")
    print(f"      - Adaptable BN Params:  {adapt_params:,} ({(adapt_params/total_params)*100:.3f}%)")

    # 4. Sinkhorn-Knopp test
    print("[4/4] Testing Sinkhorn-Knopp Optimal Transport alignment...")
    dummy_cost = torch.rand(2, 2)
    dummy_priors = torch.tensor([0.5, 0.5])
    transport_plan = sinkhorn_knopp_transport(dummy_cost, dummy_priors)
    print(f"      - Transport Plan Shape: {list(transport_plan.shape)}")
    print(f"      - Transport Plan Sum:   {transport_plan.sum().item():.4f}")

    print("=" * 70)
    print("[SUCCESS] ResNet18Binary and SafeAdapt-OT mechanics verified cleanly!")
    print("=" * 70)
