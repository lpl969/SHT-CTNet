#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple, Optional

import numpy as np
import torch

try:
    from scipy.special import lpmv
except Exception:
    lpmv = None
    print("[sht.py] Warning: scipy not found. Install with `pip install scipy`.")





def cartesian_to_spherical(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z) + 1e-12
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    return theta, phi


def make_uca_positions(C: int, radius: float = 0.035, z: float = 0.0) -> np.ndarray:
    xyz = np.zeros((C, 3), dtype=np.float64)
    for i in range(C):
        ang = 2.0 * math.pi * i / C
        xyz[i, 0] = radius * math.cos(ang)
        xyz[i, 1] = radius * math.sin(ang)
        xyz[i, 2] = z
    return xyz


def n_sh_from_order(L: int) -> int:
    return (L + 1) ** 2





def _real_sph_harm_sn3d(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    if lpmv is None:
        raise RuntimeError("scipy is required: please `pip install scipy`.")

    x = np.cos(theta)
    am = abs(m)

    P_lm = lpmv(am, l, x)


    K = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - am) / math.factorial(l + am))

    if m > 0:
        Y = math.sqrt(2.0) * K * P_lm * np.cos(am * phi)
    elif m < 0:
        Y = math.sqrt(2.0) * K * P_lm * np.sin(am * phi)
    else:
        Y = K * P_lm

    return Y


def build_A_real_sn3d_from_angles(thetas: np.ndarray, phis: np.ndarray, L: int) -> np.ndarray:
    thetas = np.asarray(thetas, dtype=np.float64)
    phis = np.asarray(phis, dtype=np.float64)
    C = thetas.shape[0]
    M = n_sh_from_order(L)
    A = np.zeros((C, M), dtype=np.float64)
    col = 0
    for l in range(L + 1):
        for m in range(-l, l + 1):
            A[:, col] = _real_sph_harm_sn3d(l, m, thetas, phis)
            col += 1
    return A


def build_A_real_sn3d_from_xyz(xyz: np.ndarray, L: int) -> np.ndarray:
    thetas, phis = cartesian_to_spherical(xyz)
    return build_A_real_sn3d_from_angles(thetas, phis, L)





def regularized_pinv(A: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    AtA = A.T @ A
    M = AtA.shape[0]
    A_dag = np.linalg.solve(AtA + lam * np.eye(M, dtype=np.float64), A.T)
    return A_dag





def build_sht_from_xyz(
    xyz: np.ndarray,
    L: int,
    lam: float = 1e-3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    A_np = build_A_real_sn3d_from_xyz(xyz, L)
    A_dag_np = regularized_pinv(A_np, lam)
    device = device if device is not None else torch.device("cpu")
    A = torch.from_numpy(A_np).to(device=device, dtype=dtype)
    A_dag = torch.from_numpy(A_dag_np).to(device=device, dtype=dtype)
    return A, A_dag


def build_sht_from_angles(
    thetas: np.ndarray,
    phis: np.ndarray,
    L: int,
    lam: float = 1e-3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    A_np = build_A_real_sn3d_from_angles(thetas, phis, L)
    A_dag_np = regularized_pinv(A_np, lam)
    device = device if device is not None else torch.device("cpu")
    A = torch.from_numpy(A_np).to(device=device, dtype=dtype)
    A_dag = torch.from_numpy(A_dag_np).to(device=device, dtype=dtype)
    return A, A_dag





class SHTProjection(torch.nn.Module):
    def __init__(self, A_dag_real: torch.Tensor):
        super().__init__()
        if A_dag_real.ndim != 2:
            raise ValueError("A_dag_real must be (M, C)")
        self.register_buffer("A_dag", A_dag_real.to(torch.float32))

    @torch.no_grad()
    def _project(self, xr: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        yr = torch.einsum("mc,bcft->bmft", self.A_dag, xr)
        yi = torch.einsum("mc,bcft->bmft", self.A_dag, xi)
        return yr, yi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Input must be 4-D tensor (B, 2C, F, T).")
        B, C2, F, T = x.shape
        if C2 % 2 != 0:
            raise ValueError("Channel dim must be even: first half real, second half imag.")
        C = C2 // 2
        xr = x[:, :C, :, :]
        xi = x[:, C:, :, :]
        yr, yi = self._project(xr, xi)
        return torch.cat([yr, yi], dim=1)

