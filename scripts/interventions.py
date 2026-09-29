"""Small learned edits of the hidden state at one layer, used as alternatives to the paper's map (see e91_intervention_compare.py):

  FlowLowRank  a FLAS-style iterated edit of the same size as the map: a per-token nonlinear low-rank velocity field
               v(z, t) = B silu(A z + e(t)) on the RMS-normalized state, integrated with N forward-Euler steps over flow time T
  FlowMLP      the FLAS FlowBlock without the concept encoder and without self-attention (as in later FLAS versions): a time
               embedding added to the state (optional), then a gated MLP of the model's intermediate width with RMSNorm before and
               after and a per-channel residual gate initialized to 0.1; velocity = output minus input; N Euler steps over time T

Both act per token and read the flow time from the attribute `T` (training samples T ~ U[0.5, 2]; evaluation uses T = 2, as in FLAS).
Reference: FLAS, flow-based activation steering (Jin et al., 2026, arXiv:2605.05892).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms(x):
    return x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)


def time_embedding(t, dim, device):
    """sinusoidal embedding of a scalar flow time (base 10000, dim/2 frequencies), as in FLAS"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / half)
    ang = t * freqs
    return torch.cat([ang.sin(), ang.cos()])


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class FlowLowRank(nn.Module):
    def __init__(self, d, r=8, N=3, freq_dim=32):
        super().__init__()
        self.N, self.T, self.freq_dim = N, 2.0, freq_dim
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / math.sqrt(d))
        nn.init.zeros_(self.B.weight)
        self.t1, self.t2 = nn.Linear(freq_dim, r), nn.Linear(r, r)
        nn.init.zeros_(self.t2.weight)
        nn.init.zeros_(self.t2.bias)

    def forward(self, h):
        x = h.float()
        n = _rms(x)
        z = x / n
        for k in range(self.N):
            t = torch.tensor(k * self.T / self.N, device=x.device)
            e = self.t2(F.silu(self.t1(time_embedding(t, self.freq_dim, x.device))))
            z = z + (self.T / self.N) * self.B(F.silu(self.A(z) + e))
        return (z * n).to(h.dtype)


class FlowMLP(nn.Module):
    def __init__(self, d, mlp, N=3, time=True, freq_dim=128):
        super().__init__()
        self.N, self.T, self.time, self.freq_dim = N, 2.0, time, freq_dim
        if time:
            self.t1, self.t2 = nn.Linear(freq_dim, d), nn.Linear(d, d)
            nn.init.zeros_(self.t2.weight)
            nn.init.zeros_(self.t2.bias)
        self.gate, self.up = nn.Linear(d, mlp, bias=False), nn.Linear(d, mlp, bias=False)
        self.down = nn.Linear(mlp, d, bias=False)
        self.na, self.nb = RMSNorm(d), RMSNorm(d)
        self.g = nn.Parameter(torch.full((d,), 0.1))

    def velocity(self, x, t):
        y = x
        if self.time:
            y = y + self.t2(F.silu(self.t1(time_embedding(t, self.freq_dim, x.device))))
        z = self.na(y)
        y = y + self.g * self.nb(self.down(F.silu(self.gate(z)) * self.up(z)))
        return y - x

    def forward(self, h):
        x = h.float()
        for k in range(self.N):
            x = x + (self.T / self.N) * self.velocity(x, torch.tensor(k * self.T / self.N, device=x.device))
        return x.to(h.dtype)
