# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: #Enter feature name here
import torch as t
from model import SodiumModel, SodiumModelType


class SodiumOptimizer:
    """
    Define the optimizer for the Sodium motion correction model
    """

    def __init__(self, model, transform_initial_lr, appearance_initial_lr, mask_nll):
        assert (isinstance(model, SodiumModel))
        self.model = model

        no_ims = len(self.model.translations)

        # Update the priors along with the appearances
        appearance_params = [self.model.tissue_distributions, self.model.noise_parameter]

        if self.model.model_type == SodiumModelType.VOXELWISE_WITH_TISSUE_PRIOR:
            raise NotImplementedError()

        # Create a separate optimiser for each of the images
        self.transform_optims = [t.optim.Adam([self.model.translations[i], self.model.angles[i]],
                                              lr=transform_initial_lr, betas=(0.0, 0.9)) for i in range(no_ims)]
        # The appearance parameters are shared, so only require a single optimiser
        self.appearance_optim = t.optim.Adam(appearance_params, lr=appearancelf_initial_lr, betas=(0.0, 0.9))

        self.transform_schedulers = [t.optim.lr_scheduler.StepLR(self.transform_optims[i], 100, gamma=0.1)
                                     for i in range(no_ims)]
        self.appearance_scheduler = t.optim.lr_scheduler.StepLR(self.appearance_optim, 100, gamma=0.1)
        self.mask_nll = mask_nll

    def update_appearance(self, observed, batch_size):
        # Choose a random batch of images to optimise the appearance
        perm = t.randperm(observed.size(-1))
        batch_idx = perm[:batch_size]

        def closure():
            self.appearance_optim.zero_grad()
            predictions = self.model.forward_model(batch_idx)
            loss = self.model.calculate_loss(observed, predictions, batch_idx, self.mask_nll, False)
            loss.backward(retain_graph=True)
            return loss

        self.appearance_optim.step(closure)
        self.appearance_scheduler.step()

    def update_transform(self, observed):
        for idx in range(observed.size(-1)):
            def closure():
                self.transform_optims[idx].zero_grad()
                batch_idx = t.tensor([idx], device=observed.device)
                predictions = self.model.forward_model(batch_idx)

                loss = self.model.calculate_loss(observed, predictions, batch_idx, self.mask_nll, False)
                loss.backward(retain_graph=True)
                return loss

            self.transform_optims[idx].step(closure)
            self.transform_schedulers[idx].step()

    def sweep_rotations(self, observed, steps=(-0.1, -0.05, 0.0, 0.05, 0.1)):
        import copy
        with t.no_grad():
            for idx in range(observed.size(-1)):
                original_rotation = copy.deepcopy(self.model.angles[idx][0, 3].cpu().numpy())
                best_rotation = copy.deepcopy(original_rotation)
                best_loss = None

                for s in steps:
                    batch_idx = t.tensor([idx], device=observed.device)
                    self.model.angles[idx][0, 3] = original_rotation + s
                    predictions = self.model.forward_model(batch_idx)
                    loss = self.model.calculate_loss(observed, predictions, batch_idx, self.mask_nll,
                                                     True).cpu().numpy()
                    if best_loss is None:
                        best_loss = copy.deepcopy(loss)
                    elif loss < best_loss:
                        best_loss = copy.deepcopy(loss)
                        best_rotation = copy.deepcopy(self.model.angles[idx][0, 3].cpu().numpy())
                print(best_rotation)
                self.model.angles[idx][0, 3] = t.Tensor(best_rotation).to(self.model.angles[0].device)

    def update_parameters(self, observed, batch_size, no_appearance_updates=4):
        perm = t.randperm(observed.size(-1))
        batch_idx = perm[:batch_size]
        for i in range(no_appearance_updates):
            self.update_appearance(observed, batch_size)

        self.update_transform(observed)
        with t.no_grad():
            predictions = self.model.forward_model(batch_idx)
            self.model.calculate_loss(observed, predictions, batch_idx, self.mask_nll, True)
