"""Tiled regional kernels and the FP32 residual-kernel solve.

Regional dot products are reused for all requested bandwidths inside each tile.
"""
from __future__ import annotations

import numpy as np
import torch


def exp_kernel(sim, beta):
    return (-float(beta) * (1 - sim)).exp()


def global_kernel(x, z, beta):
    return exp_kernel(x @ z.T, beta)


def self_mass(x, betas):
    sim = x @ x.transpose(1, 2)
    return [exp_kernel(sim, beta).mean((1, 2)) for beta in betas]


def set_kernels(x, z, betas, chunk_x=128, chunk_z=128, symmetric=False):
    """RKHS-normalized mean of all region pairs, with bounded tile memory."""
    if x.ndim != 3 or z.ndim != 3 or x.shape[1:] != z.shape[1:]:
        raise ValueError('Expected matching [images, regions, channels] descriptors')
    if x.dtype != z.dtype or x.device != z.device or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('Kernel operands must share floating precision and device')
    if symmetric and (x.data_ptr() != z.data_ptr() or chunk_x != chunk_z):
        raise ValueError('Symmetric tiling requires identical operands and square tiles')
    betas = tuple(float(beta) for beta in betas)
    dx = self_mass(x, betas)
    dz = dx if symmetric else self_mass(z, betas)
    result = [torch.empty((len(x), len(z)), dtype=x.dtype, device=x.device) for _ in betas]
    regions, channels = x.shape[1:]
    for i in range(0, len(x), chunk_x):
        xx = x[i:i + chunk_x]
        for j in range(i if symmetric else 0, len(z), chunk_z):
            zz = z[j:j + chunk_z]
            sim = (xx.reshape(-1, channels) @ zz.reshape(-1, channels).T).reshape(
                len(xx), regions, len(zz), regions)
            for k, beta in enumerate(betas):
                mass = exp_kernel(sim, beta).mean((1, 3))
                value = mass / torch.sqrt(dx[k][i:i + len(xx), None] * dz[k][None, j:j + len(zz)])
                result[k][i:i + len(xx), j:j + len(zz)] = value
                if symmetric and j != i:
                    result[k][j:j + len(zz), i:i + len(xx)] = value.T
    return result


def pyramid_kernels(xs, zs, betas, tile=128, symmetric=False):
    if len(xs) != 3 or len(zs) != 3:
        raise ValueError('Exactly three pyramid levels are required')
    result = None
    for x, z in zip(xs, zs):
        layer = set_kernels(x, z, betas, tile, tile, symmetric=symmetric)
        if result is None:
            result = layer
        else:
            for total, part in zip(result, layer):
                total.add_(part)
    return [part.div_(3) for part in result]


def regional_kernels(query, support, branch, betas, tile=128, symmetric=False):
    if branch == 'patch':
        sim = query['patch'] @ support['patch'].T
        return [exp_kernel(sim, beta) for beta in betas]
    if branch not in ('fpn', 'lateral'):
        raise ValueError(branch)
    return pyramid_kernels(query[branch], support[branch], betas, tile, symmetric)


def prior(features, text):
    # Keep upstream's einsum layout, including its floating-point operation order.
    return torch.einsum('sd, cd -> sc', features, text.T)


def residual(features, labels, text):
    return torch.nn.functional.one_hot(labels, num_classes=text.shape[1]).to(features.dtype) - prior(features, text)


def system_matrix(gram, lam):
    # Upstream iterates CPU FP32 scalar tensors; 1 / lambda is rounded before
    # multiplying the GPU Gram. A Python reciprocal or G / lambda differs.
    reciprocal = (1 / torch.tensor(float(lam), dtype=torch.float32)).item()
    return reciprocal * gram + torch.eye(len(gram), dtype=gram.dtype, device=gram.device)


def solve(gram, rhs, lam):
    return torch.linalg.solve(system_matrix(gram, lam), rhs)


def numerical_residual(gram, rhs, weights, lam, row_chunk=512):
    """FP64 accumulated residual and normwise backward error of the FP32 system."""
    reciprocal = (1 / torch.tensor(float(lam), dtype=torch.float32)).item()
    error2 = rhs2 = matrix2 = 0.
    double_weights = weights.double()
    for i in range(0, len(gram), row_chunk):
        # The coefficient matrix must be rounded in FP32 before promotion.
        matrix = (reciprocal * gram[i:i + row_chunk]).clone()
        rows = torch.arange(len(matrix), device=gram.device)
        matrix[rows, rows + i] += 1
        matrix = matrix.double()
        target = rhs[i:i + row_chunk].double()
        error2 += float(((matrix @ double_weights - target) ** 2).sum())
        rhs2 += float((target ** 2).sum())
        matrix2 += float((matrix ** 2).sum())
    error, rhsnorm = error2 ** .5, rhs2 ** .5
    backward = error / (matrix2 ** .5 * float(torch.linalg.vector_norm(double_weights)) + rhsnorm)
    return dict(relative_rhs_residual=error / max(rhsnorm, 1e-30), backward_error=backward)


def numpy_set(x, z, beta):
    """Small independent CPU reference; deliberately does not use tiled code."""
    x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
    def diagonal(a):
        return np.exp(-beta * (1 - np.einsum('ird,isd->irs', a, a))).mean((1, 2))
    mass = np.zeros((len(x), len(z)), np.float64)
    for i in range(x.shape[1]):
        for j in range(z.shape[1]):
            mass += np.exp(-beta * (1 - x[:, i] @ z[:, j].T)) / (x.shape[1] * z.shape[1])
    return mass / np.sqrt(diagonal(x)[:, None] * diagonal(z)[None])
