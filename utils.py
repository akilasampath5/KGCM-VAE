"""
utils.py
--------
Utility functions for Framework-KGCM-VAE: training, evaluation metrics,
and causal effect estimation.

Reference:
    Sampath et al., "Knowledge-Guided Causal Modeling (KGCM-VAE) for
    Sea Ice Thickness and Atmospheric Forcing Variables." Under review, 2025.
"""

import numpy as np
import torch
import torch.nn.functional as F
from models import compute_mmd_stable


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def kgcm_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    kl_weight: float  = 0.001,
    mmd_weight: float = 1.0,
    use_mmd: bool     = True,
) -> tuple:
    """
    Combined KGCM-VAE training loss.

    Loss = MSE (reconstruction) + KL divergence + MMD (treatment balancing)

    Args:
        y_pred    : Predicted outcomes, shape (B, T, 1).
        y_true    : Ground truth outcomes, shape (B, T, 1).
        mu        : Latent mean, shape (B, T, latent_dim).
        logvar    : Latent log variance, shape (B, T, latent_dim).
        z         : Latent samples, shape (B, T, latent_dim).
        t         : Treatment indicator, shape (B, T, 1).
        kl_weight : Weight for KL divergence term.
        mmd_weight: Weight for MMD balancing term.
        use_mmd   : Whether to include MMD loss.

    Returns:
        (total_loss, mse_loss, kl_loss, mmd_loss) as scalars.
    """
    mse_loss = F.mse_loss(y_pred, y_true)
    kl_loss  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    mmd_loss = torch.tensor(0.0, device=y_pred.device)
    if use_mmd:
        t_flat   = t[:, 0, 0]
        z_flat   = z[:, -1, :]
        mask_t0  = t_flat < 0.5
        mask_t1  = t_flat >= 0.5
        z_t0     = z_flat[mask_t0]
        z_t1     = z_flat[mask_t1]
        mmd_loss = compute_mmd_stable(z_t0, z_t1, device=y_pred.device)

    total = mse_loss + kl_weight * kl_loss + mmd_weight * mmd_loss
    return total, mse_loss, kl_loss, mmd_loss


# ---------------------------------------------------------------------------
# Causal Effect Estimation
# ---------------------------------------------------------------------------

def estimate_ate(
    model: torch.nn.Module,
    X: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
) -> float:
    """
    Estimate Average Treatment Effect (ATE).

        ATE = E[Y(1) - Y(0)]

    Args:
        model: Trained DCMVAE model.
        X    : Covariate tensor, shape (B, T, features).
        t0   : Control treatment tensor.
        t1   : Treated treatment tensor.

    Returns:
        Scalar ATE estimate.
    """
    model.eval()
    with torch.no_grad():
        y0, _, _, _ = model(X, t0, mode="eval")
        y1, _, _, _ = model(X, t1, mode="eval")
    return (y1 - y0).mean().item()


def compute_pehe(
    y0_pred: np.ndarray,
    y1_pred: np.ndarray,
    y0_true: np.ndarray,
    y1_true: np.ndarray,
) -> float:
    """
    Precision in Estimation of Heterogeneous Effects (PEHE).

        PEHE = sqrt(E[(ITE_pred - ITE_true)^2])

    Args:
        y0_pred, y1_pred: Predicted potential outcomes.
        y0_true, y1_true: True potential outcomes.

    Returns:
        Scalar PEHE score (lower is better).
    """
    ite_pred = y1_pred - y0_pred
    ite_true = y1_true - y0_true
    return np.sqrt(np.mean((ite_pred - ite_true) ** 2))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate model on test set.

    Args:
        model      : Trained DCMVAE model.
        test_loader: Test DataLoader with (X, t_factual, t_counter, y0, y1).
        device     : Torch device.

    Returns:
        Dictionary with MSE, ATE, and PEHE metrics.
    """
    model.eval()
    all_y0_pred, all_y1_pred = [], []
    all_y0_true, all_y1_true = [], []
    mse_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            X, t_f, t_c, y0, y1 = [b.to(device) for b in batch]

            y0_pred, _, _, _ = model(X, t_f, mode="eval")
            y1_pred, _, _, _ = model(X, t_c, mode="eval")

            mse_total += F.mse_loss(y0_pred, y0).item()
            n_batches += 1

            all_y0_pred.append(y0_pred.cpu().numpy())
            all_y1_pred.append(y1_pred.cpu().numpy())
            all_y0_true.append(y0.cpu().numpy())
            all_y1_true.append(y1.cpu().numpy())

    y0_pred_np = np.concatenate(all_y0_pred)
    y1_pred_np = np.concatenate(all_y1_pred)
    y0_true_np = np.concatenate(all_y0_true)
    y1_true_np = np.concatenate(all_y1_true)

    ate_pred = (y1_pred_np - y0_pred_np).mean()
    ate_true = (y1_true_np - y0_true_np).mean()
    pehe     = compute_pehe(y0_pred_np, y1_pred_np, y0_true_np, y1_true_np)

    return {
        "MSE"     : mse_total / n_batches,
        "ATE_pred": float(ate_pred),
        "ATE_true": float(ate_true),
        "PEHE"    : float(pehe),
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print evaluation metrics."""
    prefix = f"[{label}] " if label else ""
    for k, v in metrics.items():
        print(f"{prefix}{k}: {v:.4f}")
