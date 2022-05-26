# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)

import torch as t
from enum import Enum


class IMAGE_LOSS_TYPE(Enum):
    Gaussian = 0
    Rice = 1


def calculate_image_nll(target, predictions, noise_level, image_loss_type=IMAGE_LOSS_TYPE.Rice):
    if image_loss_type is IMAGE_LOSS_TYPE.Rice:
        nll = -rice_log_prob(target, predictions, noise_level)
    elif image_loss_type is IMAGE_LOSS_TYPE.Gaussian:
        nll = -gauss_log_prob(target, predictions, noise_level)
    else:
        raise NotImplementedError()

    # Take the mean over samples
    nll = t.mean(nll, dim=0)

    return nll


def rice_log_prob(target, predictions, noise_level):
    from rice import Rice
    from numpy import prod
    n_samples = predictions.shape[0]
    n_im = predictions.shape[-1]
    d = prod(predictions.shape[1:-1])
    y_samples = predictions + 1e-3
    r = Rice(t.reshape(y_samples, (n_samples, d * n_im)), noise_level, 200)
    nll = r.log_prob(t.reshape(target, (1, d * n_im)))
    nll = t.reshape(nll, (n_samples, d, n_im))

    # print("percentage of nans", t.sum(t.isnan(nll)) / (N_SAMPLES * N * 64 * 64 * 64))
    return nll


def gauss_log_prob_no_reduce(target, predictions, sigma_sq):
    return -0.5 * t.log(sigma_sq * 2 * 3.14159) - 0.5 * t.square(target - predictions) / sigma_sq


def gauss_log_prob(target, predictions, sigma_sq):
    return -0.5 * t.log(sigma_sq * 2 * 3.14159) - 0.5 * t.sum(t.square(target - predictions), dim=1) / sigma_sq
