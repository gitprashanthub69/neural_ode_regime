"""
model.py — Neural ODE model for market regime detection.
Architecture:
  - Encoder: GRU reads feature sequence → initial latent state
  - ODE Function: 3-layer MLP (64→64→latent_dim) with tanh activation
  - Decoder: Linear layer latent_dim → 3 regime classes (bull/bear/crisis)
Uses torchdiffeq odeint with adjoint method for memory-efficient backprop.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


LATENT_DIM = 32
NUM_REGIMES = 3  # bull, bear, crisis


class ODEFunc(nn.Module):
    """
    Neural ODE dynamics function.
    3-layer MLP: latent_dim → 64 → 64 → latent_dim with tanh activations.
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
        )
        # Initialize weights for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, t, z):
        """ODE dynamics: dz/dt = f(z, t)"""
        return self.net(z)


class NeuralODERegimeDetector(nn.Module):
    """
    Full Neural ODE model for regime detection.

    Forward pass:
      1. GRU encoder reads input feature sequence → initial latent state z0
      2. Neural ODE integrates z0 forward in time → z(T)
      3. Decoder maps z(T) → regime class logits
    """

    def __init__(self, input_dim, latent_dim=LATENT_DIM, num_regimes=NUM_REGIMES):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder: GRU to read feature sequence into initial latent state
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Neural ODE function
        self.ode_func = ODEFunc(latent_dim)

        # Decoder: map latent state to regime classes
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_regimes),
        )

    def forward(self, x, integration_time=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim) — feature sequences
            integration_time: time points for ODE integration (default: [0, 1])

        Returns:
            logits: (batch_size, num_regimes) — regime class logits
        """
        # Encode sequence into initial latent state
        _, h_n = self.encoder(x)  # h_n: (num_layers, batch, latent_dim)
        z0 = h_n[-1]  # Take last layer hidden state: (batch, latent_dim)

        # Set integration time
        if integration_time is None:
            integration_time = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)

        # Solve ODE: z(t) = z0 + ∫ f(z, t) dt
        # odeint returns (time_points, batch, latent_dim)
        z_t = odeint(self.ode_func, z0, integration_time, method="dopri5")

        # Take final time point
        z_final = z_t[-1]  # (batch, latent_dim)

        # Decode to regime logits
        logits = self.decoder(z_final)

        return logits

    def get_latent_trajectory(self, x, n_points=10):
        """Get full latent trajectory for visualization."""
        with torch.no_grad():
            _, h_n = self.encoder(x)
            z0 = h_n[-1]
            t = torch.linspace(0, 1, n_points, device=x.device, dtype=x.dtype)
            z_t = odeint(self.ode_func, z0, t, method="dopri5")
            return z_t, t
