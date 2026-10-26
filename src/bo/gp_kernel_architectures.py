"""GPyTorch covariance architectures on observed-training-normalized inputs.

No local distance, radial covariance, or tail formula is evaluated here.
The linear adapter supplies the original x-.5 coordinate convention only.
"""
from __future__ import annotations

import math
import torch
from gpytorch.kernels import (
    AdditiveKernel, LinearKernel, MaternKernel, ProductKernel, RBFKernel,
    RQKernel, ScaleKernel,
)

GP_ARCHITECTURES = (
    'rbf', 'matern15', 'matern25', 'matern05', 'rq', 'additive_rbf',
    'additive_matern15', 'rbf_linear', 'matern15_linear', 'rbf_times_linear',
)


class CheckedRQKernel(RQKernel):
    """Delegate RQ covariance to GPyTorch, rejecting invalid optimizer states.

    Native RQ can return finite covariance even when softplus parameters have
    underflowed to zero. Reject those states so our existing fit retry runs.
    No positive lower bound or statistical prior is introduced.
    """
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if (not torch.isfinite(self.lengthscale).all() or not (self.lengthscale > 0).all()
                or not torch.isfinite(self.alpha).all() or not (self.alpha > 0).all()):
            raise RuntimeError('RQ lengthscale and alpha must be finite and positive.')
        if not torch.isfinite(x1).all() or not torch.isfinite(x2).all():
            raise ValueError('RQ input coordinates must be finite.')
        covariance = super().forward(x1, x2, diag=diag,
                                     last_dim_is_batch=last_dim_is_batch, **params)
        if not torch.isfinite(covariance).all():
            raise RuntimeError('RQ covariance must be finite.')
        return covariance


class CenteredLinearKernel(LinearKernel):
    """Coordinate adapter; upstream LinearKernel computes the covariance."""
    def forward(self, x1, x2, **params):
        return super().forward(x1-.5, x2-.5, **params)


def _mean_additive(kind, input_dim, ard):
    components = [
        (RBFKernel(active_dims=(index,)) if kind == 'rbf'
         else MaternKernel(nu=1.5, active_dims=(index,)))
        for index in range(input_dim)
    ]
    if not ard:
        for component in components[1:]:
            component.raw_lengthscale = components[0].raw_lengthscale
    # Mean, rather than sum: this constant is not an extra learned amplitude.
    # double() before initialize preserves the input_dim reciprocal precision.
    result = ScaleKernel(AdditiveKernel(*components)).double()
    result.initialize(outputscale=torch.tensor(1. / input_dim, dtype=torch.double))
    result.raw_outputscale.requires_grad_(False)
    return result


def build_kernel_architecture(architecture, *, input_dim, ard=True):
    """Build the same statistical architecture using upstream covariances."""
    if architecture not in GP_ARCHITECTURES:
        raise ValueError(f'Unknown GP architecture {architecture!r}.')
    if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim < 1:
        raise ValueError('input_dim must be a positive integer.')
    if not isinstance(ard, bool):
        raise ValueError('ard must be a boolean.')
    options = {'ard_num_dims': input_dim if ard else None}
    if architecture in ('rbf', 'matern15', 'matern25', 'matern05'):
        return (RBFKernel(**options) if architecture == 'rbf' else
                MaternKernel(nu={'matern05': .5, 'matern15': 1.5, 'matern25': 2.5}[architecture], **options))
    if architecture == 'rq':
        kernel = CheckedRQKernel(**options).double()
        kernel.initialize(alpha=1.)
        return kernel
    if architecture.startswith('additive_'):
        return _mean_additive(architecture.removeprefix('additive_'), input_dim, ard)
    stationary = (MaternKernel(nu=1.5, **options) if architecture == 'matern15_linear'
                  else RBFKernel(**options))
    linear = CenteredLinearKernel().double()
    linear.initialize(variance=1.)
    if architecture == 'rbf_times_linear':
        linear.raw_variance.requires_grad_(False)
        return ProductKernel(stationary, linear)
    return AdditiveKernel(stationary, linear)


def lengthscale_components(kernel):
    """Enumerate real lengthscale owners, excluding sum/product wrappers."""
    return tuple((name or 'base_kernel', module) for name, module in kernel.named_modules()
                 if getattr(module, 'has_lengthscale', False))


def initialize_lengthscales(kernel, value):
    """Apply each original deterministic fit start to all stationary components."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Lengthscale start must be finite and positive.')
    components = lengthscale_components(kernel)
    if not components:
        raise ValueError('Kernel has no lengthscale components.')
    for _, module in components:
        module.initialize(lengthscale=value)


def _values(tensor):
    result = tensor.detach().cpu().reshape(-1).tolist()
    if not all(math.isfinite(value) for value in result):
        raise ValueError('Kernel hyperparameters must be finite for diagnostics.')
    return result


def kernel_hyperparameters(model):
    """Return JSON-compatible component diagnostics without changing the model."""
    root = model.covar_module
    scale = _values(root.outputscale)[0] if isinstance(root, ScaleKernel) else None
    kernel = root.base_kernel if isinstance(root, ScaleKernel) else root
    components = []
    for name, module in kernel.named_modules():
        if not getattr(module, 'has_lengthscale', False) and not isinstance(module, LinearKernel):
            continue
        has_lengthscale = getattr(module, 'has_lengthscale', False)
        has_alpha, has_linear = isinstance(module, RQKernel), isinstance(module, LinearKernel)
        components.append({
            'name': name or 'base_kernel', 'type': type(module).__name__,
            'lengthscale': _values(module.lengthscale) if has_lengthscale else None,
            'alpha': _values(module.alpha) if has_alpha else None,
            'linear_variance': _values(module.variance) if has_linear else None,
            'lengthscale_learned': bool(module.raw_lengthscale.requires_grad) if has_lengthscale else False,
            'alpha_learned': bool(module.raw_alpha.requires_grad) if has_alpha else False,
            'linear_variance_learned': bool(module.raw_variance.requires_grad) if has_linear else False,
        })
    return {'outputscale': scale, 'components': components}
