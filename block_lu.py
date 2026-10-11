"""Exact dense block LU with CUDA arithmetic and host/mmap matrix storage.

Partial pivoting factors a full-height panel on CUDA. Only trailing columns and
the right-hand side need the panel's row permutation: earlier L columns are not
used again, because forward elimination of every RHS occurs during factoring.
There is no symmetry assumption, eigenvalue truncation, or iterative solver.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import torch


def require(value, message):
    if not value:
        raise RuntimeError(message)


class MatrixStore:
    """CPU tensors are storage views only; every floating operation is CUDA."""
    def __init__(self, tensor, path=None, backing=None):
        require(tensor.device.type == 'cpu' and tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1],
                'Square host matrix storage required')
        require(tensor.dtype in (torch.float32, torch.float64), 'Unsupported storage precision')
        self.tensor, self.path, self.backing = tensor, path, backing
        self.n, self.dtype = len(tensor), tensor.dtype

    @classmethod
    def create(cls, n, dtype=torch.float32, path=None):
        if path is None:
            return cls(torch.empty((n, n), dtype=dtype, device='cpu'))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backing = np.memmap(path, mode='w+', dtype=np.float32 if dtype == torch.float32 else np.float64,
                            shape=(n, n))
        return cls(torch.from_numpy(backing), path, backing)

    def read(self, rows, columns, device, dtype=None):
        return self.tensor[rows, columns].to(device=device, dtype=dtype or self.dtype)

    def write(self, rows, columns, value):
        require(value.is_cuda, 'Matrix arithmetic/output must originate on CUDA')
        # copy_ performs transfer/assignment, never CPU mathematical transforms.
        self.tensor[rows, columns].copy_(value, non_blocking=False)

    def flush(self):
        if self.backing is not None:
            self.backing.flush()


def capacity(n, classes, tile=2048):
    scalar = n * n * 4
    # Estimates, not measured peaks. Panel input/output and a trailing block
    # may overlap; reserve 2GiB for allocator and CUDA/library workspace.
    gpu_fp32 = 6 * n * tile * 4 + 3 * n * classes * 4 + 2 * 1024**3
    return dict(entries=n, classes=classes, tile=tile,
        gram_fp32_bytes=scalar, gram_plus_work_fp32_bytes=2 * scalar,
        gram_plus_work_fp64_reference_bytes=3 * scalar,
        conservative_gpu_fp32_bytes=gpu_fp32,
        conservative_gpu_fp64_bytes=2 * (gpu_fp32 - 2 * 1024**3) + 2 * 1024**3,
        lu_flops_approx=2 * n**3 / 3,
        pcie_matrix_bytes_approx=8 * n**3 / (3 * tile),
        note='Dense direct algorithm; byte/flop estimates exclude feature extraction and do not predict actual throughput.')


def panel_permutation(pivots, rows, device):
    """Integer pivot bookkeeping on CPU; all matrix permutation occurs on CUDA."""
    permutation = list(range(rows))
    for index, target in enumerate(pivots.cpu().tolist()):
        target = int(target) - 1  # LAPACK's one-based panel pivots.
        require(index <= target < rows, 'Invalid panel pivot')
        permutation[index], permutation[target] = permutation[target], permutation[index]
    return torch.tensor(permutation, dtype=torch.long, device=device)


def factor_solve(work, rhs, device=None, tile=2048, factor_dtype=None, progress=None):
    """Overwrite work with block-eliminated matrix; return the GPU solution.

    work must already contain the complete intended rounded coefficient matrix.
    factor_dtype must equal storage dtype, preventing round-back of Schur updates.
    """
    device = torch.device(device or rhs.device)
    dtype = factor_dtype or work.dtype
    require(device.type == 'cuda' and rhs.is_cuda and rhs.device == device, 'CUDA-only direct solver')
    require(dtype == work.dtype and dtype in (torch.float32, torch.float64), 'Work precision must match factors')
    require(tile > 0 and rhs.ndim == 2 and len(rhs) == work.n, 'Invalid dense system dimensions')
    n, started = work.n, time.perf_counter()
    target = rhs.to(dtype=dtype).clone()
    pivoted_rows, panel_count = 0, 0
    for begin in range(0, n, tile):
        end = min(begin + tile, n)
        width, rows = end - begin, n - begin
        panel = work.read(slice(begin, n), slice(begin, end), device, dtype)
        previous_library = torch.backends.cuda.preferred_linalg_library()
        try:
            torch.backends.cuda.preferred_linalg_library('cusolver')
            lu, pivots, info = torch.linalg.lu_factor_ex(panel, pivot=True, check_errors=False)
        finally:
            torch.backends.cuda.preferred_linalg_library(previous_library)
        require(int(info) == 0, f'Singular direct LU panel at {begin}; info={int(info)}')
        del panel
        permutation = panel_permutation(pivots, rows, device)
        pivoted_rows += int((permutation != torch.arange(rows, device=device)).sum())
        target[begin:] = target[begin:].index_select(0, permutation)
        lower = lu[:width, :width]
        solved_rhs = torch.linalg.solve_triangular(lower, target[begin:end], upper=False, unitriangular=True)
        target[begin:end] = solved_rhs
        if end < n:
            target[end:].addmm_(lu[width:], solved_rhs, beta=1., alpha=-1.)
        for column in range(end, n, tile):
            column_end = min(column + tile, n)
            block = work.read(slice(begin, n), slice(column, column_end), device, dtype)
            block = block.index_select(0, permutation)
            upper = torch.linalg.solve_triangular(lower, block[:width], upper=False, unitriangular=True)
            block[:width] = upper
            if end < n:
                block[width:].addmm_(lu[width:], upper, beta=1., alpha=-1.)
            work.write(slice(begin, n), slice(column, column_end), block)
            del block, upper
        work.write(slice(begin, n), slice(begin, end), lu)
        del lu, lower, pivots, permutation, solved_rhs
        panel_count += 1
        if progress is not None:
            progress(dict(stage='factor', completed=end, entries=n,
                          panels=panel_count, seconds=time.perf_counter()-started))
    # Upper triangular block substitution. target holds the already forward-
    # eliminated RHS; there is no CPU forward/backward solve.
    for begin in reversed(range(0, n, tile)):
        end = min(begin + tile, n)
        value = target[begin:end].clone()
        for column in range(end, n, tile):
            column_end = min(column + tile, n)
            block = work.read(slice(begin, end), slice(column, column_end), device, dtype)
            value.addmm_(block, target[column:column_end], beta=1., alpha=-1.)
        diagonal = work.read(slice(begin, end), slice(begin, end), device, dtype)
        target[begin:end] = torch.linalg.solve_triangular(diagonal, value, upper=True)
    require(bool(torch.isfinite(target).all()), 'Nonfinite out-of-core direct solution')
    torch.cuda.synchronize(device)
    return target, dict(method='full_height_panel_partial_pivot_LU', approximate=False,
        CPU_matrix_arithmetic=False, host_integer_pivot_bookkeeping=True,
        entries=n, rhs_columns=rhs.shape[1], tile=tile, factor_dtype=str(dtype),
        panels=panel_count, pivoted_row_positions=pivoted_rows, panel_library_preference='cusolver',
        seconds=time.perf_counter()-started)


def coefficient_store(gram, lam, device, path=None, dtype=torch.float32, rows=512):
    """Literal FP32 I + reciprocal(lambda)*G; optional FP64 storage promotion."""
    result = MatrixStore.create(gram.n, dtype, path)
    reciprocal = (1 / torch.tensor(float(lam), dtype=torch.float32)).item()
    for begin in range(0, gram.n, rows):
        end = min(begin + rows, gram.n)
        value = gram.read(slice(begin, end), slice(None), device, torch.float32)
        value.mul_(reciprocal)
        index = torch.arange(end-begin, device=device)
        value[index, index+begin] += 1
        result.write(slice(begin, end), slice(None), value.to(dtype=dtype))
    return result


def residual(gram, lam, rhs, alpha, rows=512):
    device = rhs.device
    reciprocal = (1 / torch.tensor(float(lam), dtype=torch.float32)).item()
    error2 = torch.zeros((), device=device, dtype=torch.float64)
    matrix2 = error2.clone()
    target, solution = rhs.double(), alpha.double()
    for begin in range(0, gram.n, rows):
        end = min(begin+rows, gram.n)
        block = gram.read(slice(begin, end), slice(None), device, torch.float32)
        block.mul_(reciprocal)
        index = torch.arange(end-begin, device=device)
        block[index, index+begin] += 1
        double = block.double()
        error2 += ((double @ solution - target[begin:end])**2).sum()
        matrix2 += (double**2).sum()
    error = error2.sqrt()
    rhs_norm, alpha_norm = torch.linalg.vector_norm(target), torch.linalg.vector_norm(solution)
    return dict(relative_rhs_residual=float(error / rhs_norm.clamp_min(1e-30)),
                backward_error=float(error / (matrix2.sqrt()*alpha_norm+rhs_norm).clamp_min(1e-30)))


def global_gram(features, beta, path=None, rows=512):
    require(features.is_cuda and features.dtype == torch.float32, 'GPU FP32 features required')
    result = MatrixStore.create(len(features), torch.float32, path)
    for begin in range(0, len(features), rows):
        end = min(begin+rows, len(features))
        value = features[begin:end] @ features.T
        torch.sub(1, value, out=value)
        value.mul_(-float(beta)).exp_()
        result.write(slice(begin, end), slice(None), value)
    return result


def native_gram(features, beta, set_kernel, path=None, rows=512, tile=128):
    """Use 128x128 regional matrix products and transfer larger row slabs."""
    require(features.is_cuda and features.dtype == torch.float32, 'GPU FP32 native features required')
    result = MatrixStore.create(len(features), torch.float32, path)
    for begin in range(0, len(features), rows):
        end = min(begin+rows, len(features))
        block = set_kernel(features[begin:end], features, [beta], tile, tile)[0]
        result.write(slice(begin, end), slice(None), block)
    return result


def center_support(gram, device, floor=1e-10, negative_tolerance=2e-6, rows=512):
    """GPU FP64 support statistics, then in-place FP32 centered host storage."""
    n = gram.n
    means = torch.empty(n, dtype=torch.float64, device=device)
    diagonal = torch.empty_like(means)
    for begin in range(0, n, rows):
        end = min(begin+rows, n)
        block = gram.read(slice(begin, end), slice(None), device, torch.float32)
        means[begin:end] = block.sum(1, dtype=torch.float64) / n
        index = torch.arange(end-begin, device=device)
        diagonal[begin:end] = block[index, index+begin].double()
    mean = means.mean()
    variance = diagonal - 2 * means + mean
    require(bool(torch.isfinite(variance).all()) and float(variance.min()) >= -negative_tolerance,
            'Invalid centered support variance')
    scale = variance.clamp_min(floor).sqrt()
    record = dict(minimum_variance=float(variance.min()), maximum_variance=float(variance.max()),
        mean_variance=float(variance.mean()), floored_rows=int((variance < floor).sum()),
        negative_rows=int((variance < 0).sum()), rows=n, variance_floor=floor)
    for begin in range(0, n, rows):
        end = min(begin+rows, n)
        value = gram.read(slice(begin, end), slice(None), device, torch.float64)
        value = value - means[begin:end, None] - means[None, :] + mean
        value = (value / (scale[begin:end, None] * scale[None, :])).float()
        gram.write(slice(begin, end), slice(None), value)
    return dict(means=means, mean=mean, scale=scale), record


@torch.inference_mode()
def self_test(device, size=257, tile=64):
    device = torch.device(device)
    require(device.type == 'cuda' and torch.cuda.is_available(), 'CUDA required')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator(device=device).manual_seed(490026)
    records = []
    with tempfile.TemporaryDirectory(prefix='lamp_lu_check_') as directory:
        for dtype in (torch.float32, torch.float64):
            # A nonsymmetric, well-conditioned matrix requiring row pivoting.
            raw = torch.randn((size, size), generator=generator, device=device, dtype=dtype) * .01
            raw.diagonal().add_(2.)
            permutation = torch.randperm(size, generator=generator, device=device)
            matrix = raw.index_select(0, permutation)
            rhs = torch.randn((size, 7), generator=generator, device=device, dtype=dtype)
            work = MatrixStore.create(size, dtype, Path(directory) / f'{dtype}.bin')
            work.write(slice(None), slice(None), matrix)
            alpha, record = factor_solve(work, rhs, device, tile, dtype)
            reference = torch.linalg.solve(matrix, rhs)
            relative = torch.linalg.vector_norm(matrix @ alpha-rhs) / torch.linalg.vector_norm(rhs)
            difference = torch.linalg.vector_norm(alpha-reference) / torch.linalg.vector_norm(reference)
            tolerance = 3e-5 if dtype == torch.float32 else 3e-12
            require(float(relative) < tolerance and float(difference) < tolerance, 'Blocked pivot LU reference failed')
            require(record['pivoted_row_positions'] > 0, 'Pivot test did not exercise row exchanges')
            record.update(relative_rhs_residual=float(relative), relative_reference_difference=float(difference))
            records.append(record)
    return dict(passed=True, records=records, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                no_dataset_or_test_features_read=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--size', type=int, default=257)
    parser.add_argument('--tile', type=int, default=64)
    parser.add_argument('--estimate', type=int, help='Support entries; metadata estimate only')
    parser.add_argument('--classes', type=int, default=1000)
    args = parser.parse_args()
    if args.estimate is not None:
        print(json.dumps(capacity(args.estimate, args.classes, args.tile), indent=2))
    else:
        require(args.self_test, 'Choose --self-test or --estimate')
        print(json.dumps(self_test(args.device, args.size, args.tile), indent=2), flush=True)
