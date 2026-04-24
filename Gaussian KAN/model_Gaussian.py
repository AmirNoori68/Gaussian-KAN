# ==========================================================
# Gaussian KAN model used in the paper
# ==========================================================
#
# This file contains only the core model.
# It is enough to reproduce the Gaussian KAN architecture used in the
# experiments, provided that the same data scaling, architecture, grid size,
# epsilon, optimizer, and random seed are used.
#
# Gaussian basis:
#
#     phi_g(x) = exp( - (x - c_g)^2 / eps^2 )
#
# where the centers are uniformly distributed on [0,1]:
#
#     c_g in [0,1],   g = 1,...,G.
#
# Important:
# The first-layer inputs must be mapped to [0,1]^d before being passed
# to this model. In the experiments, physical points x_phys in [a,b]^d
# were mapped by
#
#     x = (x_phys - a) / (b - a).
#
# This matters because the Gaussian centers below are fixed on [0,1].
#
# Main experimental settings used in the paper included:
#     ARCH     = (2, 12, 12, 1)
#     NUM_GRID = 20
#     eps      = [1/(G-1),2/(G-1)]
#     dtype    = torch.float32
#     sampling = Halton points for training and mesh grid for evaluation
#
# The coefficient initialization is also part of the model definition:
#
#     coeffs ~ Normal(0, 1 / (input_dim * num_grid)).
#

# ==========================================================

import torch
import torch.nn as nn


class GaussianKANLayer(nn.Module):
    """
    One Gaussian KAN layer.

    Input:
        x : tensor of shape (N, input_dim)

    Output:
        y : tensor of shape (N, output_dim)

    Each edge function is represented by

        psi_{j,i}(x_i) = sum_g a_{i,j,g} phi_g(x_i),

    with

        phi_g(x_i) = exp( - (x_i - c_g)^2 / eps^2 ).

    The same grid centers and the same epsilon are used for all edges
    in this layer.
    """

    def __init__(self, input_dim, output_dim, num_grid, eps=1.0, device="cpu"):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grid = num_grid
        self.eps = eps

        self.coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, num_grid, device=device)
        )

        nn.init.normal_(
            self.coeffs,
            mean=0.0,
            std=1.0 / (input_dim * num_grid),
        )

        self.register_buffer(
            "centers",
            torch.linspace(0.0, 1.0, num_grid, device=device),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim, 1).expand(-1, -1, self.num_grid)

        r2 = (x - self.centers) ** 2
        phi = torch.exp(-r2 / (self.eps ** 2))

        y = torch.einsum("nig,iog->no", phi, self.coeffs)
        return y


class GKAN(nn.Module):
    """
    Multi-layer Gaussian KAN.

    Example:
        model = GKAN(
            a=(2, 12, 12, 1),
            num_grid=20,
            eps=0.054,
            device=device,
        )

    Notes:
        - The input should already be scaled to [0,1]^d.
        - The same eps is used in all layers.
        - Centers are fixed and uniformly distributed on [0,1].
        - Only the coefficients are trainable.
    """

    def __init__(self, a, *, num_grid, eps, device="cpu"):
        super().__init__()

        self.arch = tuple(a)
        self.num_grid = num_grid
        self.eps = eps

        self.layers = nn.ModuleList(
            [
                GaussianKANLayer(
                    input_dim=i,
                    output_dim=j,
                    num_grid=num_grid,
                    eps=eps,
                    device=device,
                )
                for i, j in zip(a[:-1], a[1:])
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x