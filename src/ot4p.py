from src.utils import matching, matrix_power, generate_random_orthogonal_matrix
import torch
import torch.nn as nn


class OT4P(nn.Module):
    def __init__(self, size: int, batch: int = 1):
        """
        Initialize the OT4P model.

        Parameters:
        size (int): The size of the orthogonal matrix.
        batch (int): The batch size.
        """
        super(OT4P, self).__init__()
        self.register_buffer('base', generate_random_orthogonal_matrix(batch, size))
        self.register_buffer('constraint', None)

    def ver2orth(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Map the vector to the orthogonal matrix.

        Parameters:
        matrix (torch.Tensor): Batch x N x N tensor.

        Returns:
        torch.Tensor: Orthogonal matrix.
        """
        skew = matrix.triu(1)
        return torch.matrix_exp(skew - skew.transpose(-2, -1))
    
    def orth2perm(self, orth_matrix: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        Moves the orthogonal matrix toward the closest permutation matrix.

        Parameters:
        orth_matrix (torch.Tensor): Batch x N x N orthogonal matrix.
        tau (float): Temperature parameter.

        Returns:
        torch.Tensor: Orthogonal matrix close to permutation matrix.
        """
        perm_matrix = matching(orth_matrix, self.constraint)
        
        dets = torch.linalg.det(perm_matrix)
        adjust = (-2 * (dets < 0).float() + 1).unsqueeze(1)
        perm_matrix[:, :, -1].mul_(adjust)
        
        matrix_powered = matrix_power(perm_matrix.transpose(-2, -1) @ orth_matrix, tau)
        new_orth_matrix = perm_matrix @ matrix_powered
        
        new_orth_matrix[:, :, -1].mul_(adjust)
        return new_orth_matrix    

    def get_trans_matrix(self, orth_matrix: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        Compute the equivalent transformation of the mapping 'orth2perm'.

        Parameters:
        orth_matrix (torch.Tensor): Batch x N x N orthogonal matrix.
        tau (float): Temperature parameter.

        Returns:
        torch.Tensor: Transformation matrix.
        """
        with torch.no_grad():
            perm_matrix = self.orth2perm(orth_matrix, tau)
            trans_matrix = perm_matrix @ orth_matrix.transpose(-2, -1)
        return trans_matrix
        
    def forward(self, matrix: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        Map the unconstrained vector space to the orthogonal group,
        where the temperature parameter, in the limit, 
        concentrates orthogonal matrices near permutation matrices.

        Parameters:
        matrix (torch.Tensor): Batch x N x N tensor.
        tau (float): Temperature parameter.

        Returns:
        torch.Tensor: Orthogonal matrix close to permutation matrix.
        """
        is_2d = matrix.dim() == 2
        if is_2d:
            matrix = matrix.unsqueeze(0)
        
        orth_matrix = self.base @ self.ver2orth(matrix)
        trans_matrix = self.get_trans_matrix(orth_matrix, tau)
        perm_matrix = trans_matrix @ orth_matrix
        
        if is_2d:
            perm_matrix = perm_matrix.squeeze(0) 
        return perm_matrix

    def update_base(self, param):
        """
        param: Batch x N x N or N x N
        Update the base orthogonal matrix.
        """
        with torch.no_grad():
            if param.dim() == 2:
                matrix = param.unsqueeze(0)
            orth_matrix = self.base @ self.ver2orth(matrix)
            self.base.copy_(orth_matrix)
            param.set_(torch.zeros_like(param))