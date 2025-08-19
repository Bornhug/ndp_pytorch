# gp_torch_gpytorch.py
from typing import Callable, Dict, Tuple
import torch, gpytorch

class _WrappedExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_params, kernel_params):
        super().__init__(train_x, train_y, likelihood)
        # Mean & kernel roughly mimic those in your JAX prior
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        # Set hyper-parameters from the incoming `params` dict
        self.mean_module.constant       = torch.nn.Parameter(
            torch.tensor(mean_params["μ"])
        )
        self.covar_module.base_kernel.lengthscale = torch.nn.Parameter(
            torch.tensor(kernel_params["ℓ"])
        )
        self.covar_module.outputscale   = torch.nn.Parameter(
            torch.tensor(kernel_params["σ_f"]**2)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def predict(
    prior,                     # still accepted but unused – gpytorch carries the model
    params: Dict,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    diag: bool = False,
) -> Callable[[torch.Tensor], gpytorch.distributions.MultivariateNormal]:

    train_x, train_y = train_data
    train_y = train_y.squeeze(-1)

    # Build likelihood with fixed noise variance
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.nn.Parameter(torch.tensor(params["noise_variance"]))

    model = _WrappedExactGP(
        train_x, train_y, likelihood,
        params["mean_function"], params["kernel"]
    )
    model.eval(); likelihood.eval()     # no training loop here

    @torch.no_grad()
    def _predict(test_x: torch.Tensor):
        posterior = likelihood(model(test_x))
        if diag:
            # Returns only the diagonal (variance) in a cheap lazy way
            cov = torch.diag_embed(posterior.variance)   # (n,n) diagonal tensor
            return gpytorch.distributions.MultivariateNormal(
                posterior.mean, cov
            )
        return posterior        # already a MultivariateNormal

    return _predict
