# ==========================================================
# First-layer Gaussian KAN diagnostics
# ==========================================================
#
# This file contains only the diagnostic code used to measure the
# conditioning of the first Gaussian KAN layer.
#
# The diagnostic is computed from the training inputs after they have
# been mapped to [0,1]^d. It is not computed from the test grid.
#
# The reason for using only the first layer is that this layer is the
# only layer whose inputs are the original physical coordinates
# after scaling. Therefore, its Gaussian feature matrix directly
# describes how well the input domain is resolved by the chosen
# centers and epsilon.
#
# For a Gaussian KAN layer, each input coordinate is expanded by the
# one-dimensional Gaussian feature map
#
#     phi_g(x_p) = exp( - (x_p - c_g)^2 / eps^2 ),
#
# where
#
#     c_g in [0,1],   g = 1,...,G.
#
# For each input coordinate p, we build the 1D feature matrix
#
#     Phi_p[n,g] = phi_g(x_n,p),
#
# using all training samples x_n,p. Thus Phi_p has shape
#
#     (number of training points) x (number of Gaussian centers).
#
# The condition number and numerical rank are computed separately for
# each coordinate p. The reported critical coordinate is the one with
# the smallest numerical rank. If several coordinates have the same
# smallest rank, we choose the one with the largest condition number.
#
# This is useful because rank loss in the first-layer feature matrix
# means that the Gaussian features cannot distinguish the training
# samples well in that coordinate. Later layers cannot recover
# distinguishability that has already been lost at the input-feature
# level.
#
# In the experiments, this diagnostic was evaluated after training
# using the same training points used to fit the model. Since the
# centers and epsilon are fixed in this model, the first-layer
# diagnostic itself depends on the training inputs, G, epsilon, and
# floating-point precision, not on the learned coefficients.
# ==========================================================

import torch
import numpy as np


# ==========================================================
# Gaussian feature matrix
# ==========================================================

@torch.no_grad()
def gaussian_phi_1d(x, centers, eps):
    """
    Build the 1D Gaussian feature matrix.

    Parameters
    ----------
    x : torch.Tensor, shape (N,)
        One coordinate of the training inputs.

    centers : torch.Tensor, shape (G,)
        Uniform Gaussian centers in [0,1].

    eps : float
        Gaussian length-scale.

    Returns
    -------
    Phi : torch.Tensor, shape (N, G)
        Phi[n,g] = exp(-(x_n - c_g)^2 / eps^2).
    """

    return torch.exp(-((x[:, None] - centers[None, :]) ** 2) / (eps ** 2))


# ==========================================================
# SVD-based diagnostics
# ==========================================================

@torch.no_grad()
def svd_diagnostics(A):
    """
    Compute condition number, singular values, tolerance, and numerical rank.

    The numerical rank is computed using the standard floating-point
    tolerance

        tol = max(m,n) * sigma_max * eps_machine,

    where eps_machine depends on the tensor dtype.
    """

    s = torch.linalg.svdvals(A)

    sigma_max = s.max()
    sigma_min = s.min()

    if sigma_min == 0:
        cond = float("inf")
    else:
        cond = float((sigma_max / sigma_min).item())

    m, n = A.shape
    eps_machine = torch.finfo(A.dtype).eps
    tol = max(m, n) * sigma_max * eps_machine

    rank = int((s >= tol).sum().item())

    return {
        "cond_phi": cond,
        "sigma_max": float(sigma_max.item()),
        "sigma_min": float(sigma_min.item()),
        "tol": float(tol.item()),
        "rank": rank,
    }


# ==========================================================
# First-layer diagnostics only
# ==========================================================

@torch.no_grad()
def first_layer_conditioning(model, x_train):
    """
    Compute first-layer conditioning diagnostics.

    Parameters
    ----------
    model : GKAN
        Trained or untrained Gaussian KAN model.
        Only model.layers[0] is used.

    x_train : torch.Tensor, shape (N, d)
        Training inputs after mapping to [0,1]^d.

    Returns
    -------
    stats : dict
        Contains diagnostics for each input coordinate and the selected
        critical coordinate.

    Notes
    -----
    The critical coordinate is selected by:

        1. smallest numerical rank;
        2. if tied, largest condition number.

    This is different from simply choosing the largest condition number,
    because rank loss is the more severe numerical issue.
    """

    dtype = torch.float32

    layer = model.layers[0]

    x_train = x_train.to(dtype)
    centers = layer.centers.to(dtype)
    eps = float(layer.eps)

    input_dim = x_train.shape[1]

    cond_all = []
    sigma_max_all = []
    sigma_min_all = []
    tol_all = []
    rank_all = []

    for p in range(input_dim):

        Phi = gaussian_phi_1d(
            x_train[:, p],
            centers,
            eps,
        )

        diag = svd_diagnostics(Phi)

        cond_all.append(diag["cond_phi"])
        sigma_max_all.append(diag["sigma_max"])
        sigma_min_all.append(diag["sigma_min"])
        tol_all.append(diag["tol"])
        rank_all.append(diag["rank"])

    cond_all = np.array(cond_all, dtype=float)
    sigma_max_all = np.array(sigma_max_all, dtype=float)
    sigma_min_all = np.array(sigma_min_all, dtype=float)
    tol_all = np.array(tol_all, dtype=float)
    rank_all = np.array(rank_all, dtype=int)

    # ------------------------------------------------------
    # Critical coordinate:
    # smallest rank first, then largest condition number
    # ------------------------------------------------------
    min_rank = np.min(rank_all)
    candidates = np.where(rank_all == min_rank)[0]

    if len(candidates) == 1:
        critical_coordinate = int(candidates[0])
    else:
        critical_coordinate = int(candidates[np.argmax(cond_all[candidates])])

    return {
        "eps": eps,
        "num_grid": int(layer.num_grid),
        "input_dim": int(input_dim),

        "cond_phi_all": cond_all,
        "rank_all": rank_all,
        "sigma_max_all": sigma_max_all,
        "sigma_min_all": sigma_min_all,
        "tol_all": tol_all,

        "critical_coordinate": critical_coordinate,

        "critical_cond_phi": float(cond_all[critical_coordinate]),
        "critical_rank": int(rank_all[critical_coordinate]),
        "critical_sigma_max": float(sigma_max_all[critical_coordinate]),
        "critical_sigma_min": float(sigma_min_all[critical_coordinate]),
        "critical_tol": float(tol_all[critical_coordinate]),

        "rank_min": int(np.min(rank_all)),
        "cond_phi_max": float(np.max(cond_all)),
        "cond_phi_mean": float(np.mean(cond_all)),
    }


# ==========================================================
# Example usage
# ==========================================================
#
# from model_Gaussian import GKAN
#
# torch.set_default_dtype(torch.float32)
# torch.manual_seed(seed)
#
# model = GKAN(
#     a=(2, 12, 12, 1),
#     num_grid=20,
#     eps=2/(G-1),
#     device=device,
# )
#
# # x_train must already be mapped to [0,1]^d:
# # x_train = (x_train_phys - a) / (b - a)
#
# stats = first_layer_conditioning(model, x_train)
#
# print(stats["critical_coordinate"])
# print(stats["critical_cond_phi"])
# print(stats["critical_rank"])
# ==========================================================