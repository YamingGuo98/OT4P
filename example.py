"""
A minimal example demonstrating the use of OT4P

We provide a minimal example demonstrating the use of OT4P. Given X and Y = PXP^{\top}, 
where P is a permutation matrix, the objective is to find the true permutation matrix P.
This problem can be defined as:
\min_P ||PXP^{\top} - Y||^2.
We use OT4P to solve the above problem from three different perspectives:
deterministic optimization, stochastic optimization, and constrained optimization.
"""

def loss_fn(Y, X, P):
    """
    Compute the loss function.

    Parameters:
        Y (torch.Tensor): Target matrix.
        X (torch.Tensor): Input matrix.
        P (torch.Tensor): Soft permutation matrix.

    Returns:
        torch.Tensor: Loss value.
    """
    return torch.mean(torch.pow(P @ X @ P.transpose(-2, -1) - Y, 2))


def train_model(X, Y, model, optimizer, weightP, log_weightP_var=None, max_iter=500):
    """
    Train the model using the OT4P.

    Parameters:
        X (torch.Tensor): Input matrix.
        Y (torch.Tensor): Target matrix.
        model (OT4P): OT4P model instance.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        weightP (torch.nn.Parameter): Weight parameter for deterministic optimization.
        log_weightP_var (torch.nn.Parameter, optional): Log variance of weight parameter for stochastic optimization.
        max_iter (int): Maximum number of iterations.
    Returns:
        None
    """

    for i in range(max_iter):
        optimizer.zero_grad()

        if log_weightP_var is None:
            perm_matrix = model(weightP, tau=0.5)
        else:
            # Re-parameterization trick
            mean = weightP.unsqueeze(0).expand(5, -1, -1)
            std = torch.exp(log_weightP_var / 2).unsqueeze(0).expand(5, -1, -1)
            sample = mean + std * torch.randn_like(mean) * 0.01
            perm_matrix = model(sample, tau=0.5)

        loss_train = loss_fn(Y, X, perm_matrix)
        loss_train.backward()
        optimizer.step()

        # Compute validation loss
        with torch.no_grad():
            perm_matrix_val = model(weightP, tau=0)
            loss_val = loss_fn(Y, X, perm_matrix_val)

        # Print training and validation losses
        print(f"Iteration {i+1}: Training Loss = {loss_train.item():.6f}, Validation Loss = {loss_val.item():.6f}")

        # Update base of the model
        model.update_base(weightP)

        # Determine optimization type
        optimization_type = "Deterministic"
        if log_weightP_var is not None:
            optimization_type = "Stochastic"
        if model.constraint is not None:
            optimization_type = "Constrained"
        # Check convergence
        if loss_val < 1e-5:
            print(f"{optimization_type} optimization converges at iteration {i+1}")
            break

def main():
    size = 100
    X = torch.randn(size, size)
    trueP = torch.eye(size)[torch.randperm(size)]
    Y = trueP @ X @ trueP.T

    # Deterministic optimization
    weightP = torch.nn.Parameter(torch.randn(size, size), requires_grad=True)
    model = OT4P(size)
    optimizer = torch.optim.AdamW([weightP], lr=1e-1)
    print("Starting Deterministic Optimization...")
    train_model(X, Y, model, optimizer, weightP, log_weightP_var=None)

    # Stochastic optimization
    weightP = torch.nn.Parameter(torch.randn(size, size), requires_grad=True)
    log_weightP_var = torch.nn.Parameter(torch.randn(size, size), requires_grad=True)
    model = OT4P(size)
    optimizer = torch.optim.AdamW([weightP], lr=1e-1)
    print("Starting Stochastic Optimization...")
    train_model(X, Y, model, optimizer, weightP, log_weightP_var=log_weightP_var)

    # Constrained optimization
    constraint_matrix = torch.ones((size, size))
    num_selected = int(size * 0.05)
    selected_rows = torch.randperm(size)[:num_selected]
    for row in selected_rows:
        col_index = trueP[row].nonzero().item()
        constraint_matrix[row, :] = 0
        constraint_matrix[:, col_index] = 0
        constraint_matrix[row, col_index] = 1

    weightP = torch.nn.Parameter(torch.randn(size, size), requires_grad=True)
    model = OT4P(size)
    model.constraint = constraint_matrix.unsqueeze(0)
    optimizer = torch.optim.AdamW([weightP], lr=1e-1)
    print("Starting Constrained Optimization...")
    train_model(X, Y, model, optimizer, weightP, log_weightP_var=None)


if __name__ == "__main__":
    import torch
    from src.ot4p import OT4P
    main()