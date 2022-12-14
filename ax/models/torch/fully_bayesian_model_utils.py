# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import pyro  # @manual=fbsource//third-party/pypi/pyro-ppl:pyro-ppl
import torch
from ax.models.torch.botorch_defaults import _get_model
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from botorch.models.gp_regression import SquareRootSingleTaskGP
from torch import Tensor


def _get_rbf_kernel(num_samples: int, dim: int) -> ScaleKernel:
    return ScaleKernel(
        base_kernel=RBFKernel(ard_num_dims=dim, batch_shape=torch.Size([num_samples])),
        batch_shape=torch.Size([num_samples]),
    )


def _get_rbf_noscale_kernel(num_samples: int, dim: int) -> RBFKernel:
    return RBFKernel(ard_num_dims=dim, batch_shape=torch.Size([num_samples]))


def _get_single_task_gpytorch_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    state_dict: Optional[Dict[str, Tensor]] = None,
    num_samples: int = 512,
    thinning: int = 16,
    use_input_warping: bool = False,
    gp_kernel: str = "matern",
    **kwargs: Any,
) -> ModelListGP:
    r"""Instantiates a batched GPyTorchModel(ModelListGP) based on the given data.
    The model fitting is based on MCMC and is run separately using pyro. The MCMC
    samples will be loaded into the model instantiated here afterwards.

    Returns:
        A ModelListGP.
    """
    if len(task_features) > 0:
        raise NotImplementedError("Currently do not support MT-GP models with MCMC!")
    if len(fidelity_features) > 0:
        raise NotImplementedError(
            "Fidelity MF-GP models are not currently supported with MCMC!"
        )
    num_mcmc_samples = num_samples // thinning
    covar_modules = [
        _get_rbf_kernel(num_samples=num_mcmc_samples, dim=Xs[0].shape[-1])
        if gp_kernel == "rbf"
        else None
        for _ in range(len(Xs))
    ]

    models = [
        _get_model(
            X=X.unsqueeze(0).expand(num_mcmc_samples, X.shape[0], -1),
            Y=Y.unsqueeze(0).expand(num_mcmc_samples, Y.shape[0], -1),
            Yvar=Yvar.unsqueeze(0).expand(num_mcmc_samples, Yvar.shape[0], -1),
            fidelity_features=fidelity_features,
            use_input_warping=use_input_warping,
            covar_module=covar_module,
            **kwargs,
        )
        for X, Y, Yvar, covar_module in zip(Xs, Ys, Yvars, covar_modules)
    ]
    model = ModelListGP(*models)
    model.to(Xs[0])
    return model


def _get_active_learning_gpytorch_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    state_dict: Optional[Dict[str, Tensor]] = None,
    num_samples: int = 512,
    thinning: int = 16,
    use_input_warping: bool = False,
    gp_kernel: str = "rbf",
    **kwargs: Any,
) -> ModelListGP:
    r"""Instantiates a batched GPyTorchModel(ModelListGP) based on the given data.
    The model fitting is based on MCMC and is run separately using pyro. The MCMC
    samples will be loaded into the model instantiated here afterwards.

    Returns:
        A ModelListGP.
    """
    if len(task_features) > 0:
        raise NotImplementedError("Currently do not support MT-GP models with MCMC!")
    if len(fidelity_features) > 0:
        raise NotImplementedError(
            "Fidelity MF-GP models are not currently supported with MCMC!"
        )
    num_mcmc_samples = num_samples // thinning
    covar_modules = [
        _get_rbf_noscale_kernel(num_samples=num_mcmc_samples, dim=Xs[0].shape[-1])
        if gp_kernel == "rbf"
        else None
        for _ in range(len(Xs))
    ]

    models = [
        _get_model(
            X=X.unsqueeze(0).expand(num_mcmc_samples, X.shape[0], -1),
            Y=Y.unsqueeze(0).expand(num_mcmc_samples, Y.shape[0], -1),
            Yvar=Yvar.unsqueeze(0).expand(num_mcmc_samples, Yvar.shape[0], -1),
            fidelity_features=fidelity_features,
            use_input_warping=use_input_warping,
            covar_module=covar_module,
            mean_module=ZeroMean(),
            **kwargs,
        )
        for X, Y, Yvar, covar_module in zip(Xs, Ys, Yvars, covar_modules)
    ]
    for model in models:
        # to make model.subset_output() work
        model._subset_batch_dict = {
            "likelihood.noise_covar.raw_noise": -2,
            "covar_module.raw_lengthscale": -3,
        }
    model = ModelListGP(*models)
    model.to(Xs[0])
    return model


def _get_square_root_gpytorch_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    state_dict: Optional[Dict[str, Tensor]] = None,
    num_samples: int = 512,
    thinning: int = 16,
    use_input_warping: bool = False,
    gp_kernel: str = "matern",
    **kwargs: Any,
) -> ModelListGP:
    r"""Instantiates a batched GPyTorchModel(ModelListGP) based on the given data.
    The model fitting is based on MCMC and is run separately using pyro. The MCMC
    samples will be loaded into the model instantiated here afterwards.

    Returns:
        A ModelListGP.
    """
    if len(task_features) > 0:
        raise NotImplementedError("Currently do not support MT-GP models with MCMC!")
    if len(fidelity_features) > 0:
        raise NotImplementedError(
            "Fidelity MF-GP models are not currently supported with MCMC!"
        )
    num_mcmc_samples = num_samples // thinning
    covar_modules = [
        _get_rbf_kernel(num_samples=num_mcmc_samples, dim=Xs[0].shape[-1])
        if gp_kernel == "rbf"
        else None
        for _ in range(len(Xs))
    ]

    models = [
        _get_square_root_model(
            X=X.unsqueeze(0).expand(num_mcmc_samples, X.shape[0], -1),
            Y=Y.unsqueeze(0).expand(num_mcmc_samples, Y.shape[0], -1),
            Yvar=Yvar.unsqueeze(0).expand(num_mcmc_samples, Yvar.shape[0], -1),
            fidelity_features=fidelity_features,
            use_input_warping=use_input_warping,
            covar_module=covar_module,
            **kwargs,
        )
        for X, Y, Yvar, covar_module in zip(Xs, Ys, Yvars, covar_modules)
    ]
    model = ModelListGP(*models)
    model.to(Xs[0])
    return model


def _get_square_root_model(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    task_feature: Optional[int] = None,
    fidelity_features: Optional[List[int]] = None,
    use_input_warping: bool = False,
    **kwargs: Any,
) -> GPyTorchModel:
    """Instantiate a model of type depending on the input data.

    Args:
        X: A `n x d` tensor of input features.
        Y: A `n x m` tensor of input observations.
        Yvar: A `n x m` tensor of input variances (NaN if unobserved).
        task_feature: The index of the column pertaining to the task feature
            (if present).
        fidelity_features: List of columns of X that are fidelity parameters.

    Returns:
        A GPyTorchModel (unfitted).
    """
    Yvar = Yvar.clamp_min(MIN_OBSERVED_NOISE_LEVEL)
    is_nan = torch.isnan(Yvar)
    any_nan_Yvar = torch.any(is_nan)
    all_nan_Yvar = torch.all(is_nan)
    if any_nan_Yvar and not all_nan_Yvar:
        if task_feature:
            # TODO (jej): Replace with inferred noise before making perf judgements.
            Yvar[Yvar != Yvar] = MIN_OBSERVED_NOISE_LEVEL
        else:
            raise ValueError(
                "Mix of known and unknown variances indicates valuation function "
                "errors. Variances should all be specified, or none should be."
            )
    if use_input_warping:
        warp_tf = get_warping_transform(
            d=X.shape[-1],
            task_feature=task_feature,
            batch_shape=X.shape[:-2],
        )
    else:
        warp_tf = None
    if fidelity_features is not None:
        raise ValueError('SquareRootGP is not available with multi-fidelity.')
    elif task_feature is None and all_nan_Yvar:
        gp = SquareRootSingleTaskGP(
            train_X=X, train_Y=Y, input_transform=warp_tf, **kwargs)
    elif task_feature is None:
        raise ValueError('SquareRootGP is not available with fixed noise.')
    return gp


def pyro_sample_outputscale(
    concentration: float = 2.0,
    rate: float = 0.15,
    **tkwargs: Any,
) -> Tensor:

    return pyro.sample(
        "outputscale",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`
        pyro.distributions.Gamma(
            torch.tensor(concentration, **tkwargs),
            torch.tensor(rate, **tkwargs),
        ),
    )


def pyro_sample_mean(**tkwargs: Any) -> Tensor:

    return pyro.sample(
        "mean",
        # pyre-fixme[16]: Module `distributions` has no attribute `Normal`.
        pyro.distributions.Normal(
            torch.tensor(0.0, **tkwargs),
            torch.tensor(1.0, **tkwargs),
        ),
    )


def pyro_sample_noise(**tkwargs: Any) -> Tensor:

    # this prefers small noise but has heavy tails
    return pyro.sample(
        "noise",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.Gamma(
            torch.tensor(0.9, **tkwargs),
            torch.tensor(10.0, **tkwargs),
        ),
    )


def pyro_sample_saas_lengthscales(
    dim: int,
    alpha: float = 0.1,
    **tkwargs: Any,
) -> Tensor:

    tausq = pyro.sample(
        "kernel_tausq",
        # pyre-fixme[16]: Module `distributions` has no attribute `HalfCauchy`.
        pyro.distributions.HalfCauchy(torch.tensor(alpha, **tkwargs)),
    )
    inv_length_sq = pyro.sample(
        "_kernel_inv_length_sq",
        # pyre-fixme[16]: Module `distributions` has no attribute `HalfCauchy`.
        pyro.distributions.HalfCauchy(torch.ones(dim, **tkwargs)),
    )
    inv_length_sq = pyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)
    lengthscale = pyro.deterministic(
        "lengthscale",
        (1.0 / inv_length_sq).sqrt(),  # pyre-ignore [16]
    )
    return lengthscale


def pyro_sample_input_warping(
    dim: int,
    **tkwargs: Any,
) -> Tuple[Tensor, Tensor]:

    c0 = pyro.sample(
        "c0",
        # pyre-fixme[16]: Module `distributions` has no attribute `LogNormal`.
        pyro.distributions.LogNormal(
            torch.tensor([0.0] * dim, **tkwargs),
            torch.tensor([0.75**0.5] * dim, **tkwargs),
        ),
    )
    c1 = pyro.sample(
        "c1",
        # pyre-fixme[16]: Module `distributions` has no attribute `LogNormal`.
        pyro.distributions.LogNormal(
            torch.tensor([0.0] * dim, **tkwargs),
            torch.tensor([0.75**0.5] * dim, **tkwargs),
        ),
    )
    return c0, c1


def postprocess_saas_samples(samples: Dict[str, Tensor]) -> Dict[str, Tensor]:
    inv_length_sq = (
        samples["kernel_tausq"].unsqueeze(-1) * samples["_kernel_inv_length_sq"]
    )
    samples["lengthscale"] = (1.0 / inv_length_sq).sqrt()  # pyre-ignore [16]
    del samples["kernel_tausq"], samples["_kernel_inv_length_sq"]
    # this prints the summary

    return samples


def postprocess_squareroot_gp_samples(samples: Dict[str, Tensor]) -> Dict[str, Tensor]:
    if not 'sqrt_eta' in samples.keys():
        raise ValueError('For a square root GP, there must be samples of "sqrt_eta".'
                         f'Now, there is only {list(samples.keys())}')
    return samples


def postprocess_bayesian_al_samples(samples: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return samples


# pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use `typing.Dict`
#  to avoid runtime subscripting errors.
def load_mcmc_samples_to_model(model: GPyTorchModel, mcmc_samples: Dict) -> None:
    """Load MCMC samples into GPyTorchModel."""

    if "noise" in mcmc_samples:
        model.likelihood.noise_covar.noise = (
            mcmc_samples["noise"]
            .detach()
            .clone()
            .view(model.likelihood.noise_covar.noise.shape)  # pyre-ignore
            .clamp_min(MIN_INFERRED_NOISE_LEVEL)
        )
    if hasattr(model.covar_module, 'base_kernel'):
        model.covar_module.base_kernel.lengthscale = (
            mcmc_samples["lengthscale"]
            .detach()
            .clone()
            .view(model.covar_module.base_kernel.lengthscale.shape)  # pyre-ignore
        )
    else:
        model.covar_module.lengthscale = (
            mcmc_samples["lengthscale"]
            .detach()
            .clone()
            .view(model.covar_module.lengthscale.shape)  # pyre-ignore
        )
    if "outputscale" in mcmc_samples:
        model.covar_module.outputscale = (  # pyre-ignore
            mcmc_samples["outputscale"]
            .detach()
            .clone()
            # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
            #  `outputscale`.
            .view(model.covar_module.outputscale.shape)
        )
    if "mean" in mcmc_samples:
        model.mean_module.constant.data = (
            mcmc_samples["mean"]
            .detach()
            .clone()
            .view(model.mean_module.constant.shape)  # pyre-ignore
        )
    if "sqrt_eta" in mcmc_samples:
        model.sqrt_eta = (
            mcmc_samples["mean"]
            .detach()
            .clone()
            # pyre-ignore
        )
        print('Remember to check the shape of sqrt eta in FBMU!')
    if "c0" in mcmc_samples:
        model.input_transform._set_concentration(  # pyre-ignore
            i=0,
            value=mcmc_samples["c0"]
            .detach()
            .clone()
            .view(model.input_transform.concentration0.shape),  # pyre-ignore
        )
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `_set_concentration`.
        model.input_transform._set_concentration(
            i=1,
            value=mcmc_samples["c1"]
            .detach()
            .clone()
            .view(model.input_transform.concentration1.shape),  # pyre-ignore
        )


def pyro_sample_sqrt_eta(mu: float = 0, var: float = 0.1, **tkwargs: Any) -> Tensor:
    return pyro.sample(
        "sqrt_eta",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.LogNormal(
            torch.tensor(mu, **tkwargs),
            torch.tensor(var ** 0.5, **tkwargs),
        ),
    )


def pyro_sample_al_noise(mu: float = 0, var: float = 3.0, **tkwargs: Any) -> Tensor:
    return pyro.sample(
        "noise",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.LogNormal(
            torch.tensor(mu, **tkwargs),
            torch.tensor(var ** 0.5, **tkwargs),
        ),
    )


def pyro_sample_al_lengthscales(dim, mu: float = 0, var: float = 3.0, **tkwargs: Any) -> Tensor:
    return pyro.sample(
        "lengthscale",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.LogNormal(
            torch.ones(dim, **tkwargs) * mu,
            torch.ones(dim, **tkwargs) * var ** 0.5
        ),
    )


PRIOR_REGISTRY = {
    'SAAS': {
        'parameter_priors':
        {
            'outputscale_func': pyro_sample_outputscale,
            'mean_func': pyro_sample_mean,
            'noise_func': pyro_sample_noise,
            'lengthscale_func': pyro_sample_saas_lengthscales,
            'input_warping_func': pyro_sample_input_warping,
        },
        'postprocessing': postprocess_saas_samples
    },
    'BAL': {
        'parameter_priors':
        {
            'outputscale_func': None,
            'mean_func': None,
            'noise_func': pyro_sample_al_noise,
            'lengthscale_func': pyro_sample_al_lengthscales,
            'input_warping_func': None,
        },
        'postprocessing': postprocess_bayesian_al_samples
    },
    'BO': {
        'parameter_priors':
        {
            'outputscale_func': None,
            'mean_func': None,
            'noise_func': pyro_sample_al_noise,
            'lengthscale_func': pyro_sample_al_lengthscales,
            'input_warping_func': None,
        },
        'postprocessing': postprocess_bayesian_al_samples
    },
    'SquareRoot': {
        'parameter_priors':
        {
            'outputscale_func': None,
            'mean_func': None,
            'noise_func': pyro_sample_al_noise,
            'lengthscale_func': pyro_sample_al_lengthscales,
            'sqrt_eta_func': pyro_sample_sqrt_eta,
            'input_warping_func': None,
        },
        'postprocessing': postprocess_bayesian_al_samples
    },
}
