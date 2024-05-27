import numpy as np
import torch
from torch_linear_assignment import batch_linear_assignment
from typing import Optional


def matrix_power(matrix: torch.Tensor, s: float) -> torch.Tensor:
    """
    Computes the power of a square matrix using eigen decomposition.

    Parameters:
    matrix (torch.Tensor): Batch x N x N
    s (float): Scalar power

    Returns:
    torch.Tensor: Batch x N x N matrix raised to the power s
    """
    L, V = torch.linalg.eig(matrix)
    Lambda_s = torch.diag_embed(L**s)
    result = V @ Lambda_s @ torch.linalg.inv(V)
    if torch.imag(result).abs().max() > 1e-5:
        print("Imaginary part is too large")
    return torch.real(result)


def matching(orth_matrix: torch.Tensor, constraint: Optional[torch.Tensor] = None, maximize: bool = True) -> torch.Tensor:
    """
    Solves the linear assignment problem using the provided orthogonal matrix.

    Parameters:
    orth_matrix (torch.Tensor): Batch x N x N orthogonal matrix
    constraint (Optional[torch.Tensor]): Batch x N x N constraint matrix
    maximize (bool): Whether to maximize or minimize the assignment

    Returns:
    torch.Tensor: Batch x N x N permutation matrix
    """
    matrix = orth_matrix.clone()
    matrix -= matrix.min()
    if torch.is_tensor(constraint):
        matrix *= constraint
    if maximize:
        matrix = matrix.max() - matrix
        col_idxs = batch_linear_assignment(matrix)
    else:
        col_idxs = batch_linear_assignment(matrix)
    return create_perm_matrix(col_idxs)

def create_perm_matrix(indx: torch.Tensor) -> torch.Tensor:
    """
    Creates a permutation matrix from given indices.

    Parameters:
    indx (torch.Tensor): Batch x N indices

    Returns:
    torch.Tensor: Batch x N x N permutation matrix
    """
    device = indx.device
    batch_size, size = indx.shape
    perm_matrix = torch.zeros(batch_size, size, size, device=device, dtype=torch.float)
    rows = torch.arange(size, device=device).repeat(batch_size, 1)
    perm_matrix[torch.arange(batch_size, device=device).unsqueeze(1), rows, indx] = 1
    return perm_matrix

    
def generate_random_orthogonal_matrix(batch_size: int, n: int) -> torch.Tensor:
    """
    Generates a batch of random orthogonal matrices.

    Parameters:
    batch_size (int): Number of matrices to generate
    n (int): Size of each orthogonal matrix

    Returns:
    torch.Tensor: Batch x N x N orthogonal matrices
    """
    n_minus_one = n - 1 if n % 2 != 0 else n

    theta_min = np.pi / 8
    theta_max = 3 * np.pi / 8

    angles = np.random.uniform(theta_min, theta_max, (batch_size, n_minus_one // 2))

    Q_batch = torch.zeros((batch_size, n, n), dtype=torch.float32)
    for b in range(batch_size):
        Q = torch.eye(n, dtype=torch.float32)  # Initialize to identity matrix
        for i in range(0, n_minus_one, 2):
            theta = angles[b, i // 2]
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            Q[i, i] = cos_theta
            Q[i, i + 1] = -sin_theta
            Q[i + 1, i] = sin_theta
            Q[i + 1, i + 1] = cos_theta

        if n % 2 != 0:
            Q[-1, -1] = 1
        Q_batch[b] = Q

    return Q_batch