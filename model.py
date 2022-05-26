# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Contains the generative model for the image data
from enum import Enum
import torch as t
import numpy as np


class SodiumModelType(Enum):
    SHARED_TISSUE_VALUE = 0
    VOXELWISE_WITH_TISSUE_PRIOR = 1


class SodiumModel:
    """
    Contains the generative model for the Sodium data
    """

    def __init__(self, segmentations, no_ims, tissue_mean_priors, sigma, model_type, im_loss_type, psf=None,
                 translate_std=1e-2, rotate_std=1e-1):

        assert (isinstance(model_type, SodiumModelType))
        self.model_type = model_type

        self.device = SodiumModel.get_device()

        if psf is not None:
            self.psf = psf.detach().requires_grad_(False)
        else:
            self.psf = None

        self.segmentations = segmentations

        self.brain_mask = segmentations[-1].detach().requires_grad_(False)
        self.no_ims = no_ims
        self.im_loss_type = im_loss_type

        self.D = 64 * 64 * 64

        self.tissue_distributions = self.tissue_mean_priors = self.noise_parameter = self.translation_var = \
            self.rotation_var = self._angles_prior = self._translations_prior = self._scale = self.angles = \
            self.translations = None

        self.create_variables(tissue_mean_priors, translate_std, rotate_std, sigma)

    def create_variables(self, tissue_mean_priors, translate_std, rotate_std, sigma):
        """ Construct all the variables that we want to infer """

        # The average tissue distribution has an inferred distribution, which starts with the prior values
        self.tissue_distributions = t.Tensor(
            np.stack([tissue_mean_priors[0], np.log(tissue_mean_priors[1]) * 2.0], axis=0)).to(
            self.device).requires_grad_(True)

        # Set the prior for the tissue means and variances
        self.tissue_mean_priors = t.Tensor(
            np.stack([tissue_mean_priors[0], (tissue_mean_priors[1] / 10.0) ** 2], axis=0)).to(
            self.device).requires_grad_(False)

        self.noise_parameter = t.Tensor([[sigma]]).to(self.device).requires_grad_(True)

        self.translation_var = np.square(translate_std)
        self.rotation_var = np.square(rotate_std)

        self._angles_prior = t.Tensor(t.cat([t.zeros(1, 4), t.ones(1, 4) * self.rotation_var], dim=-1)).to(
            self.device).requires_grad_(False)
        self._translations_prior = t.Tensor(t.cat([t.zeros(1, 3), t.ones(1, 3) * self.translation_var], dim=-1)).to(
            self.device).requires_grad_(False)

        self._scale = t.Tensor(t.zeros(1)).to(self.device).requires_grad_(False)

        # Construct a list of variables for the angles and translations
        initial_axis = t.randn(1, 3) * np.sqrt(self.rotation_var)
        self.angles = [t.Tensor(t.cat([initial_axis, t.zeros(1, 1)], dim=-1)
                                ).to(self.device).requires_grad_(True) for i in range(self.no_ims)]

        self.translations = [t.Tensor(t.zeros(1, 3)).to(self.device).requires_grad_(True) for i in range(self.no_ims)]

    @staticmethod
    def get_device():
        if t.cuda.is_available():
            device = t.device("cuda")
            print('cuda available')
            t.cuda.set_device(0)
        else:
            device = t.device("cpu")
        return device

    def forward_model_shared_tissue(self, batch_size):
        from image_functions import build_voxelwise_priors
        voxelwise_dist = build_voxelwise_priors(self.segmentations, self.tissue_distributions)
        m_samples = t.reshape(voxelwise_dist[0, :, 0], (1, self.D, 1))
        m_samples = t.nn.Softplus(1e5)(m_samples)
        y_samples = t.cat([m_samples for x in range(batch_size)], dim=-1)


        return y_samples

    def forward_model(self, batch_idx):
        """
        Create the forward model given the current variable states
        """
        from image_functions import transform_images
        batch_size = batch_idx.shape[0]
        if self.model_type == SodiumModelType.SHARED_TISSUE_VALUE:
            gen_images = self.forward_model_shared_tissue(batch_size)
        elif self.model_type == SodiumModelType.VOXELWISE_WITH_TISSUE_PRIOR:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        # Apply the image transformations
        angles_batch = t.cat(self.angles, dim=0)[batch_idx]
        translations_batch = t.cat(self.translations, dim=0)[batch_idx]

        y_samples = transform_images(gen_images, angles_batch, translations_batch, 1.0, (64, 64, 64))
        y_samples = t.reshape(y_samples, (batch_size, 1, 64, 64, 64))

        if self.psf is not None:
            y_samples = t.conv3d(t.nn.functional.pad(y_samples, (2, 2, 2, 2, 2, 2)).to(self.device), self.psf, stride=1)

        y_samples = t.split(t.reshape(y_samples, (batch_size, 1, self.D)), 1, dim=0)
        y_samples = t.stack([t.squeeze(x, 0) for x in y_samples], dim=-1)
        return y_samples

    def calculate_loss(self, observed, predicted, batch_idx, mask_nll=True, print_losses=True):
        """
        Calculate the loss (nll+KL terms)
        """
        from losses import calculate_image_nll

        nll = calculate_image_nll(observed[:, batch_idx], predicted, self.noise_parameter,
                                  image_loss_type=self.im_loss_type)
        # Ignore pixels with very low signal
        mask = predicted > 0.01
        if mask_nll:
            nll = t.where(mask, nll, t.zeros_like(nll))
        # Sum over images
        nll = nll.sum()

        tissue_prior_cost = 0.0
        if self.model_type == SodiumModelType.SHARED_TISSUE_VALUE:
            diff = t.reshape(self.tissue_distributions, (2, -1))[0, :] - \
                   t.reshape(self.tissue_mean_priors, (2, -1))[0, :]
            tissue_prior_cost = t.sum(t.square(diff) / t.reshape(self.tissue_mean_priors, (2, -1))[1, :])

        transform_prior_cost = t.sum(
            t.square(t.cat(self.translations, 0)[batch_idx] - self._translations_prior[:,
                                                              :3]) / self.translation_var) + t.sum(
            t.square(t.cat(self.angles, 0)[batch_idx] - self._angles_prior[:, :4]) / self.rotation_var)

        cost = nll + tissue_prior_cost + transform_prior_cost

        cost = cost / (mask).sum()
        cost.to(self.device)

        if print_losses:
            print("Cost", cost.cpu().detach().numpy(), "NLL: ", nll.cpu().detach().numpy(),
                  "KL_TISSUE_MEAN", tissue_prior_cost.cpu().detach().numpy(),
                  "Transform prior", transform_prior_cost.cpu().detach().numpy())

        return cost
