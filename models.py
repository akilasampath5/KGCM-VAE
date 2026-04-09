"""
models.py
---------
Framework-KGCM-VAE: Knowledge-Guided Causal Modeling with Variational Autoencoders
for causal inference in Arctic sea ice thickness dynamics.

Models included:
    - DCMVAE          : Deep Causal Mixture VAE (main KGCM-VAE model)
    - IHDP_TimeSeries : Data module for causal time series
    - ArcticDataLoader: Real-world Arctic data loader

Utilities:
    - gaussian_rbf_matrix : RBF kernel for MMD computation
    - compute_mmd_stable  : Stable multi-scale MMD loss

Reference:
    Sampath et al., "Knowledge-Guided Causal Modeling (KGCM-VAE) for
    Sea Ice Thickness and Atmospheric Forcing Variables." Under review, 2025.
"""

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Default Hyperparameters
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 30
BATCH_SIZE      = 32
HIDDEN_DIM      = 128
LATENT_DIM      = 64
NUM_EPOCHS      = 150
LEARNING_RATE   = 5e-4
KL_WEIGHT       = 0.001
MMD_WEIGHT      = 1.0
WINDOW_SIZE     = 5
TREATMENT_LAG   = 3


# ---------------------------------------------------------------------------
# MMD Utilities
# ---------------------------------------------------------------------------

def gaussian_rbf_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute Gaussian RBF kernel matrix between x and y.

    Args:
        x, y : Input tensors of shape (N, D) and (M, D).
        sigma: Kernel bandwidth.

    Returns:
        Kernel matrix of shape (N, M).
    """
    x_norm = (x ** 2).sum(1).unsqueeze(1)
    y_norm = (y ** 2).sum(1).unsqueeze(0)
    dists  = x_norm + y_norm - 2 * (x @ y.t())
    return torch.exp(-dists / (2 * sigma ** 2 + 1e-12))


def compute_mmd_stable(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Multi-scale Maximum Mean Discrepancy (MMD) for treatment group balancing.

    Uses three RBF bandwidths (0.5, 1.0, 2.0) for stability.
    Applied to latent representations of factual and counterfactual groups.

    Args:
        x, y  : Latent representations for two treatment groups.
        device: Torch device.

    Returns:
        Scalar MMD loss.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if x is None or y is None or x.size(0) <= 1 or y.size(0) <= 1:
        return torch.tensor(0.0, device=device)

    mmd = 0.0
    for sigma in [0.5, 1.0, 2.0]:
        K_xx   = gaussian_rbf_matrix(x, x, sigma)
        K_yy   = gaussian_rbf_matrix(y, y, sigma)
        K_xy   = gaussian_rbf_matrix(x, y, sigma)
        n, m   = x.size(0), y.size(0)
        sum_xx = (K_xx.sum() - torch.diag(K_xx).sum()) / (n * (n - 1))
        sum_yy = (K_yy.sum() - torch.diag(K_yy).sum()) / (m * (m - 1))
        sum_xy = K_xy.mean()
        mmd   += sum_xx + sum_yy - 2.0 * sum_xy
    return mmd / 3.0


# ---------------------------------------------------------------------------
# 1. DCMVAE — Main Model
# ---------------------------------------------------------------------------

class DCMVAE(nn.Module):
    """
    Deep Causal Mixture VAE (KGCM-VAE main model).

    Architecture:
        - Bidirectional GRU encoder over covariates + treatment
        - VAE reparameterization (mu, logvar) for latent representation
        - Treatment projection head amplifies treatment signal
        - Outcome head conditioned on latent z + projected treatment
        - Optional MMD loss for treatment group balancing

    The latent space learns treatment-dependent representations,
    enabling counterfactual inference for causal effect estimation.

    Args:
        dim_x_features: Number of input covariate features.
        hidden_dim    : GRU hidden state size.
        latent_dim    : VAE latent dimension.
        use_mmd       : Whether to apply MMD balancing loss.
    """

    def __init__(
        self,
        dim_x_features: int = 5,
        hidden_dim: int     = HIDDEN_DIM,
        latent_dim: int     = LATENT_DIM,
        use_mmd: bool       = True,
    ):
        super(DCMVAE, self).__init__()
        self.use_mmd = use_mmd

        # Bidirectional GRU encoder
        self.encoder_rnn = nn.GRU(
            dim_x_features + 1, hidden_dim,
            batch_first=True, bidirectional=True
        )
        self.fc_mu     = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Treatment projection: 1 → 16 dims
        self.t_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
        )

        # Outcome head conditioned on latent z + projected treatment
        self.outcome_head = nn.Sequential(
            nn.Linear(latent_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        VAE reparameterization trick.

        During training samples z ~ N(mu, sigma^2).
        During inference uses the mean (z = mu).
        """
        if mode == "train":
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(mu) * std
        return mu

    def forward(
        self,
        X: torch.Tensor,
        t: torch.Tensor,
        mode: str = "train",
    ) -> tuple:
        """
        Forward pass.

        Args:
            X   : Covariate tensor, shape (B, T, dim_x_features).
            t   : Treatment tensor, shape (B, T, 1).
            mode: "train" for sampling, "eval" for mean.

        Returns:
            y_pred : Predicted outcome, shape (B, T, 1).
            mu     : Latent mean, shape (B, T, latent_dim).
            logvar : Latent log variance, shape (B, T, latent_dim).
            z      : Sampled latent vector, shape (B, T, latent_dim).
        """
        x_and_t = torch.cat([X, t], dim=-1)
        h, _    = self.encoder_rnn(x_and_t)

        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z      = self.reparameterize(mu, logvar, mode)

        t_emb   = self.t_proj(t)
        z_and_t = torch.cat([z, t_emb], dim=-1)
        y_pred  = self.outcome_head(z_and_t)

        return y_pred, mu, logvar, z


# ---------------------------------------------------------------------------
# 2. IHDP_TimeSeries — Synthetic Data Module
# ---------------------------------------------------------------------------

class IHDP_TimeSeries:
    """
    Synthetic causal time series data module for KGCM-VAE experiments.

    Generates factual and counterfactual outcomes with regime-dependent
    treatment effects based on Arctic sea ice velocity dynamics.

    Args:
        csv_path       : Path to ERA5 CSV file.
        batch_size     : DataLoader batch size.
        sequence_length: Sequence window length.
        treatment_lag  : Number of timesteps to lag treatment variable.
    """

    def __init__(
        self,
        csv_path: str,
        batch_size: int,
        sequence_length: int,
        treatment_lag: int = TREATMENT_LAG,
    ):
        self.csv_path        = csv_path
        self.batch_size      = batch_size
        self.sequence_length = sequence_length
        self.treatment_lag   = treatment_lag
        self._load_and_preprocess_data()

    @staticmethod
    def apply_moving_window(series: np.ndarray, window_size: int) -> np.ndarray:
        """Apply rolling mean smoothing."""
        return (
            pd.Series(series.flatten())
            .rolling(window=window_size, min_periods=1)
            .mean()
            .values
            .reshape(-1, 1)
        )

    def _compute_lag(self, T: np.ndarray, lag: int) -> np.ndarray:
        """Shift treatment by lag steps; fill leading entries with zero."""
        Tlag          = np.zeros_like(T)
        Tlag[lag:]    = T[:-lag]
        return Tlag

    def _load_and_preprocess_data(self) -> None:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
        else:
            df = pd.DataFrame(
                np.random.randn(1621, 5),
                columns=["uoe", "von", "total_vel", "zos", "sithick"]
            )

        x_base = df[["uoe", "von", "total_vel"]].values
        y_base = df[["sithick"]].values
        ssh    = df["zos"].values.reshape(-1, 1)
        vel    = df["total_vel"].values.reshape(-1, 1)
        hidden = np.sin(np.linspace(0, 30 * np.pi, len(df))).reshape(-1, 1)

        T0_smooth = self.apply_moving_window(ssh, WINDOW_SIZE)
        T0_np     = T0_smooth + 2.0 * hidden + np.random.normal(0, 0.1, ssh.shape)

        v0      = np.mean(vel)
        sigmoid = 1 / (1 + np.exp(-(-5.0) * (vel - v0)))
        T1_np   = ((1.0 + 1.5 * sigmoid) * T0_np).reshape(-1, 1)

        T0_lag_np = self._compute_lag(T0_np, self.treatment_lag)
        X_RAW     = np.concatenate([x_base, T0_np, T0_lag_np], axis=1)

        num_seq = len(df) // self.sequence_length
        limit   = num_seq * self.sequence_length

        scaler       = StandardScaler()
        X_scaled     = scaler.fit_transform(X_RAW[:limit])
        self.xall    = X_scaled.reshape(num_seq, self.sequence_length, X_RAW.shape[1])

        self.t_factual = T0_np[:limit].reshape(num_seq, self.sequence_length, 1)
        self.t_counter = T1_np[:limit].reshape(num_seq, self.sequence_length, 1)
        self.y_factual = y_base[:limit].reshape(num_seq, self.sequence_length, 1)

        hidden_seq = hidden[:limit].reshape(num_seq, self.sequence_length, 1)
        T0_seq     = T0_np[:limit].reshape(num_seq, self.sequence_length, 1)
        T1_seq     = T1_np[:limit].reshape(num_seq, self.sequence_length, 1)
        delta      = -6.0 * np.abs(hidden_seq) * np.tanh(2.0 * (T1_seq - T0_seq))

        self.y0_cf = self.y_factual
        self.y1_cf = self.y_factual + delta

    def get_dataloaders(self) -> tuple:
        """
        Create train and test DataLoaders.

        Returns:
            (train_loader, test_loader)
        """
        indices        = np.arange(len(self.xall))
        tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=42)

        train_ds = TensorDataset(
            torch.FloatTensor(self.xall[tr_idx]),
            torch.FloatTensor(self.t_factual[tr_idx]),
            torch.FloatTensor(self.y_factual[tr_idx]),
        )
        test_ds = TensorDataset(
            torch.FloatTensor(self.xall[te_idx]),
            torch.FloatTensor(self.t_factual[te_idx]),
            torch.FloatTensor(self.t_counter[te_idx]),
            torch.FloatTensor(self.y0_cf[te_idx]),
            torch.FloatTensor(self.y1_cf[te_idx]),
        )
        return (
            DataLoader(train_ds, self.batch_size, shuffle=True),
            DataLoader(test_ds,  self.batch_size, shuffle=False),
        )


# ---------------------------------------------------------------------------
# 3. ArcticDataLoader — Real-World Data Module
# ---------------------------------------------------------------------------

class ArcticDataLoader:
    """
    Real-world Arctic data loader for causal inference experiments.

    Uses ERA5 ocean variables (velocity, SSH, sea ice thickness) to
    construct factual and counterfactual treatment scenarios.

    Args:
        csv_path       : Path to arctic_s2s_multivar CSV file.
        batch_size     : DataLoader batch size.
        sequence_length: Sequence window length.
        treatment_lag  : Number of timesteps to lag treatment.
    """

    def __init__(
        self,
        csv_path: str,
        batch_size: int,
        sequence_length: int,
        treatment_lag: int = 1,
    ):
        self.csv_path        = csv_path
        self.batch_size      = batch_size
        self.sequence_length = sequence_length
        self.treatment_lag   = treatment_lag
        self._load_and_preprocess_data()

    @staticmethod
    def apply_moving_window(series: np.ndarray, window_size: int) -> np.ndarray:
        return (
            pd.Series(series.flatten())
            .rolling(window=window_size, min_periods=1)
            .mean()
            .values
            .reshape(-1, 1)
        )

    def _compute_lag(self, T: np.ndarray, lag: int) -> np.ndarray:
        Tlag       = np.zeros_like(T)
        Tlag[lag:] = T[:-lag]
        return Tlag

    def _load_and_preprocess_data(self) -> None:
        df      = pd.read_csv(self.csv_path)
        xall_np = df[["uoe", "von", "total_vel", "zos"]].values
        yall_np = df[["sithick"]].values
        ssh     = df["zos"].values.reshape(-1, 1)
        vel     = df["total_vel"].values.reshape(-1, 1)

        T0_smooth = self.apply_moving_window(ssh, WINDOW_SIZE)
        T0_np     = T0_smooth + np.random.normal(0, 0.1, ssh.shape)

        v0      = np.mean(vel)
        sigmoid = 1 / (1 + np.exp(-(-5.0) * (ssh - v0)))
        T1_np   = ((1.0 + 1.5 * sigmoid) * T0_np).reshape(-1, 1)

        T0_lag_np = self._compute_lag(T0_np, self.treatment_lag)
        T1_lag_np = self._compute_lag(T1_np, self.treatment_lag)

        X_IN = np.concatenate([xall_np, T0_np, T0_lag_np], axis=1)
        num_seq = len(df) // self.sequence_length
        limit   = num_seq * self.sequence_length

        scaler         = StandardScaler()
        X_scaled       = scaler.fit_transform(X_IN[:limit])
        self.xall      = X_scaled.reshape(num_seq, self.sequence_length, X_IN.shape[1])
        self.t_factual = T0_np[:limit].reshape(num_seq, self.sequence_length, 1)
        self.t_counter = T1_np[:limit].reshape(num_seq, self.sequence_length, 1)
        self.y_factual = yall_np[:limit].reshape(num_seq, self.sequence_length, 1)
        self.y0_cf     = self.y_factual
        self.y1_cf     = self.y_factual

    def get_dataloaders(self) -> tuple:
        indices        = np.arange(len(self.xall))
        tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=42)

        train_ds = TensorDataset(
            torch.FloatTensor(self.xall[tr_idx]),
            torch.FloatTensor(self.t_factual[tr_idx]),
            torch.FloatTensor(self.y_factual[tr_idx]),
        )
        test_ds = TensorDataset(
            torch.FloatTensor(self.xall[te_idx]),
            torch.FloatTensor(self.t_factual[te_idx]),
            torch.FloatTensor(self.t_counter[te_idx]),
            torch.FloatTensor(self.y0_cf[te_idx]),
            torch.FloatTensor(self.y1_cf[te_idx]),
        )
        return (
            DataLoader(train_ds, self.batch_size, shuffle=True),
            DataLoader(test_ds,  self.batch_size, shuffle=False),
        )
