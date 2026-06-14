# KGCM-VAE: Knowledge-Guided Time-Varying Causal Inference for Arctic Sea Ice Dynamics

KGCM-VAE integrates causal discovery with variational autoencoders to quantify cause-and-effect relationships between sea ice thickness and atmospheric forcing variables. By embedding domain knowledge as structural constraints, the framework enables causal inference in dynamic physical systems.

## Dataset

The Arctic multivariate dataset is available at:
**[https://zenodo.org/records/15665532](https://zenodo.org/records/15665532)**

## Repository Structure

```
KGCM-VAE/
├── README.md
├── requirements.txt
├── src/
│   ├── models.py       # DCMVAE model, data modules, MMD utilities
│   └── utils.py        # Loss functions, ATE/PEHE evaluation metrics
└── notebooks/
    ├── Submission_KGCMVAE_Abs.ipynb
    ├── Submission_KGCM_VAE_benchmark.ipynb
    └── Submission_sit_ssh_realworld_final.ipynb
```

## Overview

| Component | Description |
|-----------|-------------|
| `DCMVAE` | Deep Causal Mixture VAE — main model |
| `IHDP_TimeSeries` | Synthetic causal time series data module |
| `ArcticDataLoader` | Real-world Arctic ERA5 data module |
| `compute_mmd_stable` | Multi-scale MMD for treatment balancing |
| `kgcm_loss` | MSE + KL + MMD combined loss |
| `compute_pehe` | PEHE causal effect evaluation metric |

## Key Features

- **Causal inference** via counterfactual outcome estimation
- **VAE reparameterization** for latent treatment-dependent representations
- **MMD balancing** to align factual and counterfactual latent distributions
- **Domain knowledge constraints** embedded as structural priors
- **PEHE and ATE** metrics for causal effect evaluation

## Installation

```bash
git clone https://github.com/akilasampath5/KGCM-VAE.git
cd KGCM-VAE
pip install -r requirements.txt
```

## Quick Start

```python
from src.models import DCMVAE, IHDP_TimeSeries, set_seed
from src.utils import kgcm_loss, evaluate

set_seed(42)

# Load data
data   = IHDP_TimeSeries("arctic_s2s_multivar_2020_2024.csv", batch_size=32, sequence_length=30)
train_loader, test_loader = data.get_dataloaders()

# Initialize model
model  = DCMVAE(dim_x_features=5, hidden_dim=128, latent_dim=64, use_mmd=True)
```

See notebooks for full training and evaluation examples.

## Keywords

Causal Inference · Variational Autoencoder · Sea Ice Thickness · Arctic ·
MMD Balancing · Counterfactual Estimation · Physics-Guided ML · ERA5
