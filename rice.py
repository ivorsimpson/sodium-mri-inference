# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Efficient implementation of the Rice distribution

import numpy as np
import torch as t


class Rice(t.nn.Module):
    def __init__(self, v, sigma2, k_max=50):
        """
        We use the definition of the rice (y/sigma2)exp(-(y^2+v^2)/2sigma2)I_0(yv/sigma2)
        This is not exactly the same as the scipy implementation.
        :param v: a tensor of shape (N_SAMPLES, N) where N is the number of random variables
        :param sigma2: The noise level (scalar)
        :param k_max: The order of the expansion for calculating the I_0 Bessel function
        """
        super(Rice, self).__init__()
        assert (len(v.shape) == 2)
        # We use the same parameterisation as in scipy.stats, where we divide v by sigma2
        self.v = v / sigma2
        self.sigma2 = sigma2
        # Precompute and save some stuff
        self.k = t.arange(k_max, dtype=t.float32, device=v.device)
        self.K_MAX = k_max
        self.k = self.k.reshape(1, 1, k_max)
        self.k = self.k.detach()
        self.k_sq_fact = 2.0 * (t.lgamma(self.k + 1))
        self.k_sq_fact = self.k_sq_fact.detach()

    def i0(self, z):
        # Evaluation at K_MAX differet points
        k = t.arange(self.K_MAX, dtype=t.float32, device=z.device)
        k = k.reshape(1, 1, self.K_MAX)

        log_divisor = t.lgamma(k + 1) * 2
        log_numerator = k * t.log(0.25 * t.pow(t.unsqueeze(z, -1), 2))

        result = (t.exp(log_numerator - log_divisor)).sum(dim=-1)
        return result

    def log_i0(self, z):
        """
        Calculate the log of the modified Bessel function of the first kind
        Approximate the infinite sum https://mathworld.wolfram.com/ModifiedBesselFunctionoftheFirstKind.html
        :param z: The input values
        """
        # Calculate the log numerator
        log_numerator = self.k * (np.log(0.25) + 2.0 * t.log(t.unsqueeze(z, -1)))
        # Use the precalculated log denominator
        log_denominator = self.k_sq_fact
        # Use logsumexp to calculate the values for the series
        result = t.logsumexp(log_numerator - log_denominator, dim=-1)

        return result

    def log_prob(self, x):
        """
        Calculate the log probability of x given this distribution.
        Uses the same parameterisation used by scipy.stats where we divide x, v and the pdf by sigma2.
        :param x: Tensor of shape (N_SAMPLES, N)
        :return:
        """
        # Check that x is the right size
        assert (x.shape[1] == self.v.shape[1])
        # Divide out the scale
        x = x / self.sigma2

        result = t.log(x) - (x ** 2 + self.v ** 2) / 2.0
        result = result + self.log_i0(self.v * x)
        # Subtract the log scale from the pdf
        result = result - t.log(self.sigma2)

        return result

    @staticmethod
    def test_rice_likelihood():
        """
        Test that we get the same results with our Rice distribution as using scipy
        :return: boolean
        """
        # Compare the torch Rice distribution with the one in scipy.stats
        from scipy.stats import rice as ss_rice
        n_samples = 10
        # Generate some random rice parameters
        v = np.fabs(np.random.rand() * 0.14)
        sigma = 0.08

        v = np.array(v)
        sigma = np.array(sigma)
        # Create a rice distribution
        r = ss_rice(v / sigma, scale=sigma)
        # Generate some samples
        samples = r.rvs(size=n_samples) * sigma

        # Calculate the log probability
        lp = r.logpdf(samples)

        # Create one of our Rice distributions
        r_v = Rice(t.reshape(t.Tensor(v), (1, 1)), t.tensor(sigma), 150)
        # Calculate the log probability of the samples
        lp2 = r_v.log_prob(t.reshape(t.Tensor(samples), (n_samples, 1)))

        lp2 = lp2.flatten().cpu().detach().numpy()
        #print(v, lp, lp2, (lp - lp2))
        assert (np.any(abs(lp - lp2) > 1e-2) == False)


if __name__ == "__main__":
    Rice.test_rice_likelihood()
