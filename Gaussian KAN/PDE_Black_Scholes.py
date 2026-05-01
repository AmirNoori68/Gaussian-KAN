import torch
import math


# --------------------------------------------------
# Black-Scholes Digital Option settings
# --------------------------------------------------
MAX_S = 1.0
MAX_T = 1.0

RISK_FREE_RATE = 0.05
VOLATILITY = 0.2
STRIKE_PRICE = 0.5


# --------------------------------------------------
# Standard normal CDF using torch
# --------------------------------------------------
def normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# --------------------------------------------------
# Exact solution: digital cash-or-nothing call
#
# u(S,t) = exp(-r (T-t)) * N(d2)
#
# d2 = [ln(S/K) + (r - 0.5 sigma^2)(T-t)]
#      / [sigma sqrt(T-t)]
#
# Terminal condition:
#   u(S,T) = 1 if S > K else 0
#
# Boundary conditions:
#   u(0,t)     = 0
#   u(S_max,t) = exact_u(S_max,t)
# --------------------------------------------------
def exact_u(xy):
    S = xy[:, 0:1]
    t = xy[:, 1:2]

    r = RISK_FREE_RATE
    sigma = VOLATILITY
    K = STRIKE_PRICE
    T = MAX_T

    tau = T - t

    payoff = (S > K).to(S.dtype)

    tau_safe = torch.clamp(tau, min=1e-12)
    S_safe = torch.clamp(S, min=1e-12)

    d2 = (
        torch.log(S_safe / K)
        + (r - 0.5 * sigma * sigma) * tau_safe
    ) / (sigma * torch.sqrt(tau_safe))

    value = torch.exp(-r * tau_safe) * normal_cdf(d2)

    # terminal condition at t = T
    is_terminal = tau <= 1e-10
    value = torch.where(is_terminal, payoff, value)

    # left boundary S = 0
    is_left = S <= 1e-12
    value = torch.where(is_left, torch.zeros_like(value), value)

    return value


# --------------------------------------------------
# Sampling: interior points
# --------------------------------------------------
def sample_interior(n, device):
    data = torch.empty(n, 2, device=device)

    data[:, 0].uniform_(0.0, MAX_S)
    data[:, 1].uniform_(0.0, MAX_T)

    return data


# --------------------------------------------------
# Sampling: boundary points
#
# Three boundary pieces:
#   1) S = 0
#   2) t = T
#   3) S = S_max
# --------------------------------------------------
def sample_boundary(n, device):
    data = torch.empty(n, 2, device=device)
    soln = torch.zeros(n, 1, device=device)

    data[:, 0].uniform_(0.0, MAX_S)
    data[:, 1].uniform_(0.0, MAX_T)

    idx = torch.arange(n, device=device)
    mode = idx % 3

    # S = 0
    mask_left = mode == 0
    data[mask_left, 0] = 0.0
    soln[mask_left, 0] = 0.0

    # t = T terminal payoff
    mask_terminal = mode == 1
    data[mask_terminal, 1] = MAX_T
    soln[mask_terminal] = (
        data[mask_terminal, 0:1] > STRIKE_PRICE
    ).to(data.dtype)

    # S = S_max
    mask_right = mode == 2
    data[mask_right, 0] = MAX_S
    soln[mask_right] = exact_u(data[mask_right])

    return data, soln


# --------------------------------------------------
# Evaluation grid
# --------------------------------------------------
def sample_grid(n, device):
    s = torch.linspace(0.0, MAX_S, n, device=device)
    t = torch.linspace(0.0, MAX_T, n, device=device)

    S, Tm = torch.meshgrid(s, t, indexing="ij")
    xy = torch.stack([S.reshape(-1), Tm.reshape(-1)], dim=1)

    return S, Tm, xy