import time
import warnings
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.special as ssl
from scipy.integrate import quad
import multiprocessing as mp
from functools import partial
from contextlib import contextmanager
import bilby.core.utils as bcu


class LinearParameterEvidenceCalculator:
    """
    A class to calculate the Bayesian evidence for linear parameters in a model.

    This class is designed for gravitational-wave analysis where some model
    parameters (e.g., spherical harmonic mode amplitudes) are linear. It manages
    the transformation from the internal 'A-space' (cosine/sine components)
    to the physical amplitude-phase space.

    The key functions are:
    1.  Calculating evidence under two different prior assumptions:
        - Uniform prior on the A-space components.
        - Uniform prior on physical amplitudes and phases (a more physical prior).
    2.  Generating resampled posteriors for the amplitude and phase parameters.

    The implementation is highly optimized to use NumPy for vectorized operations
    and to minimize data serialization overhead during multiprocessing, making it
    efficient for large posterior sample sets.
    """

    def __init__(self, likelihood, As_range=(0., 50.), n_samples_multiplier=100,
                 npool=20, suppress_warnings=True):
        """
        Initializes the evidence calculator.

        Parameters
        ----------
        likelihood : bilby.core.likelihood.GravitationalWaveTransient
            The bilby likelihood object, which contains the model and data.
            This is stored as an instance attribute.
        As_range : tuple, optional
            The prior range (min, max) for the amplitude parameters.
            Defaults to (0., 50.).
        n_samples_multiplier : int, optional
            A multiplier to determine the number of Monte Carlo samples used for
            evidence integration. The total samples per posterior point is
            `n_samples_multiplier * (number_of_modes)^2`. Defaults to 100.
        npool : int, optional
            The number of parallel processes to use for computation.
            Defaults to 20.
        suppress_warnings : bool, optional
            If True, suppresses warnings from bilby during likelihood evaluation.
            Defaults to True.
        """
        self.likelihood = likelihood
        self.As_range = As_range
        self.n_samples_multiplier = n_samples_multiplier
        self.npool = npool
        self.suppress_warnings = suppress_warnings

        # Cache for frequently accessed mode information to avoid re-computation.
        self._cached_lmn_all = None
        self._cached_names = None

    @contextmanager
    def _warning_suppression(self):
        """A context manager to suppress warnings if enabled."""
        if self.suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        else:
            yield

    @staticmethod
    def A_to_Amp_phase_array(data):
        """
        Converts A-space parameters to an amplitude-phase representation.

        This static method is optimized for performance by operating directly on
        NumPy arrays. It assumes an input array where even-indexed columns
        are cosine components and odd-indexed columns are sine components.

        Parameters
        ----------
        data : np.ndarray
            A 2D NumPy array of shape (n_samples, n_params) containing the
            A-space parameters.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape (n_samples, n_params) where even-indexed
            columns contain the amplitudes and odd-indexed columns contain the
            phases (in radians, from 0 to 2*pi).
        """
        # A-space components are ordered (A_cos_1, A_sin_1, A_cos_2, A_sin_2, ...)
        sin_parts = data[:, 1::2]
        cos_parts = data[:, ::2]
        
        # Form complex numbers z = cos + i*sin
        z1 = cos_parts + 1j * sin_parts
        
        # Amplitude is the absolute value, phase is the angle
        Amp = np.abs(z1)
        phase = np.angle(z1) % (2 * np.pi)

        # Interleave the amplitude and phase arrays for the final output
        n_samples, n_modes = Amp.shape
        result = np.zeros((n_samples, 2 * n_modes))
        result[:, ::2] = Amp   # Amplitudes at even indices
        result[:, 1::2] = phase # Phases at odd indices
        
        return result

    @staticmethod
    def A_to_Amp_phase_df(data, names):
        """
        Converts A-space parameters to an amplitude-phase DataFrame.

        This method is a convenient wrapper around `A_to_Amp_phase_array`
        and should be used for creating final, human-readable outputs rather
        than in performance-critical loops.

        Parameters
        ----------
        data : np.ndarray
            A 2D NumPy array of A-space parameters.
        names : list of str
            A list of column names for the output DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the amplitude and phase parameters.
        """
        amp_phase_array = LinearParameterEvidenceCalculator.A_to_Amp_phase_array(data)
        return pd.DataFrame(amp_phase_array, columns=names)

    def _get_mode_info(self):
        """
        Extracts and caches mode information from the likelihood object.

        Returns
        -------
        tuple
            A tuple containing:
            - lmn_all (list): A list of all mode names (e.g., ['220', '221']).
            - names (list): A list of parameter names for the DataFrame
              (e.g., ['amp220', 'phi220', ...]).
        """
        # Return cached values if they already exist to save computation time.
        if self._cached_lmn_all is not None and self._cached_names is not None:
            return self._cached_lmn_all, self._cached_names

        # Extract mode configuration from the waveform generator
        lmns = self.likelihood.waveform_generator.waveform_arguments['lmns']
        lmn_all = ['%s%d' % (lmn[:2], n) for lmn in lmns for n in range(int('%s' % lmn[-1]))]
        
        # Define the parameter name prefixes
        keys_pre = ['amp', 'phi']
        names = ['%s%s' % (key, lmn) for lmn in lmn_all for key in keys_pre]
        
        # Cache the results for future calls
        self._cached_lmn_all = lmn_all
        self._cached_names = names
        
        return lmn_all, names

    def calculate_evidence(self, result, use_multiprocessing=True):
        """
        Calculates the Bayesian evidence for the linear parameters.

        This is the main method of the class. It iterates over each posterior
        sample from the result object, calculates the evidence marginalized over
        the linear parameters for that sample, and then combines the results.

        Parameters
        ----------
        result : bilby.core.result.Result
            The result object from a bilby run, containing posterior samples.
        use_multiprocessing : bool, optional
            If True, uses a multiprocessing pool to parallelize the calculation
            across posterior samples. Defaults to True.

        Returns
        -------
        bilby.core.result.Result
            The updated result object, now containing the evidence calculations
            and the extended posterior with amplitude-phase samples.
        """
        st = time.time()
        
        post1 = result.posterior
        A_min, A_max = self.As_range
        l0 = len(post1)

        # Get mode information
        lmn_all, names = self._get_mode_info()
        n_modes = len(lmn_all)
        n_params = 2 * n_modes  # Each mode has an amplitude and a phase

        bcu.logger.info(f'Calculating evidence for {n_modes} modes with {l0} posterior samples')

        # Pre-allocate NumPy arrays for results. This is much faster than
        # appending to lists or concatenating DataFrames in a loop.
        total_samples = self.n_samples_multiplier * n_modes**2
        all_amp_phase_samples = np.zeros((l0 * total_samples, n_params))
        all_mean_samples = np.zeros((l0, n_params))
        ln_Evi_A0_array = np.zeros(l0)
        ln_Evi_A1_array = np.zeros(l0)

        # Prepare arguments for each worker process
        args_list = []
        for i in range(l0):
            post_dict = post1.iloc[i].to_dict()
            # Pass only essential parameters to reduce serialization overhead
            essential_params = {k: v for k, v in post_dict.items()
                                if k not in ['log_likelihood', 'log_prior']}
            args_list.append((
                i, essential_params, self.As_range, lmn_all,
                self.n_samples_multiplier, self.suppress_warnings
            ))

        if use_multiprocessing and l0 > 1:
            # The likelihood object is passed to starmap separately. This is
            # efficient as it's pickled only once and shared with all workers.
            with mp.Pool(processes=self.npool) as pool:
                results = pool.starmap(optimized_evidence_worker,
                                       [(args, self.likelihood) for args in args_list])
        else:
            # Fallback to sequential execution
            results = [optimized_evidence_worker(args, self.likelihood) for args in args_list]

        # Efficiently collect results from workers into the pre-allocated arrays
        for idx, (amp_phase_samples, mean_sample, ln_ev_a0, ln_ev_a1) in enumerate(results):
            start_idx = idx * total_samples
            end_idx = start_idx + len(amp_phase_samples)
            
            all_amp_phase_samples[start_idx:end_idx] = amp_phase_samples
            all_mean_samples[idx] = mean_sample
            ln_Evi_A0_array[idx] = ln_ev_a0
            ln_Evi_A1_array[idx] = ln_ev_a1

        # Trim any unused space from the pre-allocated array
        actual_total_samples = sum(len(r[0]) for r in results)
        all_amp_phase_samples = all_amp_phase_samples[:actual_total_samples]

        bcu.logger.info(f'Generated {actual_total_samples} total amplitude-phase samples')

        # Generate the resampled posterior using the collected samples
        self._generate_resampled_posterior_optimized(
            result, all_amp_phase_samples, all_mean_samples, lmn_all, names, l0
        )

        # Calculate the final Bayes factors
        self._calculate_bayes_factors(result, ln_Evi_A0_array, ln_Evi_A1_array, lmn_all, l0)

        et = time.time()
        bcu.logger.info(f'Evidence calculation completed in {et - st:.6f} seconds')
        
        return result

    def _generate_resampled_posterior_optimized(self, result, all_amp_phase_samples,
                                                all_mean_samples, lmn_all, names, l0):
        """
        Generates resampled posteriors using optimized NumPy operations.

        This method creates three sets of amplitude-phase posteriors:
        1. Mean: The maximum likelihood estimate for each original sample.
        2. Uniform (O_): Samples drawn assuming a uniform prior on the A-space components.
        3. Weighted (P_): Samples drawn assuming a uniform prior on amplitude and phase.

        Parameters
        ----------
        result : bilby.core.result.Result
            The result object to update.
        all_amp_phase_samples : np.ndarray
            All collected amplitude-phase samples.
        all_mean_samples : np.ndarray
            The mean (max likelihood) amplitude-phase values for each original sample.
        lmn_all : list
            List of mode names.
        names : list
            List of parameter column names.
        l0 : int
            Number of original posterior samples.
        """
        total_samples = len(all_amp_phase_samples)
        bcu.logger.info(f'Processing {total_samples} extended linear parameter samples')

        # Case 1: Uniform resampling (for uniform-in-A prior)
        # This is a simple random choice from all generated samples.
        uniform_indices = np.random.choice(total_samples, size=l0, replace=True)
        uniform_samples = all_amp_phase_samples[uniform_indices]

        # Case 2: Importance resampling (for uniform-in-amplitude/phase prior)
        # The weights are 1/amplitude for each mode. We use logarithms for stability.
        amplitude_cols = all_amp_phase_samples[:, ::2]  # Amplitudes are at even indices
        
        # Calculate log weights: log(w) = sum over modes[-log(amplitude)]
        # A small epsilon prevents log(0).
        log_weights = -np.sum(np.log(amplitude_cols + 1e-10), axis=1)
        
        # Normalize weights using logsumexp for numerical stability
        log_weights_normalized = log_weights - ssl.logsumexp(log_weights)
        weights_normalized = np.exp(log_weights_normalized)
        
        # Perform weighted sampling
        weighted_indices = np.random.choice(
            total_samples,
            size=l0,
            p=weights_normalized,
            replace=True
        )
        weighted_samples = all_amp_phase_samples[weighted_indices]

        # Convert final NumPy arrays to DataFrames only once at the end
        mean_df = pd.DataFrame(all_mean_samples, columns=names)
        uniform_df = pd.DataFrame(uniform_samples, columns=names).add_prefix("O_")
        weighted_df = pd.DataFrame(weighted_samples, columns=names).add_prefix("P_")
        
        # Concatenate the new posteriors with the original one
        result.posterior = pd.concat([result.posterior, mean_df, uniform_df, weighted_df],
                                     axis=1, ignore_index=False)

    def _calculate_bayes_factors(self, result, ln_Evi_A0, ln_Evi_A1, lmn_all, l0):
        """
        Calculates the final Bayes factors and stores them in the result metadata.

        Parameters
        ----------
        result : bilby.core.result.Result
            The result object to update.
        ln_Evi_A0 : np.ndarray
            Array of log evidence values for the uniform-in-A prior.
        ln_Evi_A1 : np.ndarray
            Array of log evidence values for the physical (uniform-in-amplitude) prior.
        lmn_all : list
            List of mode names.
        l0 : int
            Number of original posterior samples.
        """
        # The total evidence is the average of the evidences calculated at each
        # posterior sample point. Averaging is done by logsumexp(evidence) - log(N).
        ln_BF0 = result.log_bayes_factor - np.log(l0) + ssl.logsumexp(ln_Evi_A0)
        ln_BF1 = result.log_bayes_factor - np.log(l0) + ssl.logsumexp(ln_Evi_A1)
        
        # Fully Integrated Case (FIC) for comparison
        ln_FIC = result.log_bayes_factor - 2. * len(lmn_all)

        # Store the computed values in the result object's metadata
        result.meta_data['ln_BF0'] = ln_BF0
        result.meta_data['ln_BF1'] = ln_BF1
        result.meta_data['ln_FIC'] = ln_FIC

        bcu.logger.info(f'Log Bayes factor (uniform A prior): {ln_BF0:.6f}')
        bcu.logger.info(f'Log Bayes factor (physical prior): {ln_BF1:.6f}')
        bcu.logger.info(f'Fully integrated case (FIC): {ln_FIC:.6f}')


def optimized_evidence_worker(args, likelihood):
    """
    Worker function for parallel evidence calculation.

    This function is designed to be called by multiprocessing.Pool. It takes one
    posterior sample, calculates the linear parameter evidence for it, and
    generates Monte Carlo samples from the linear parameter posterior.

    It returns NumPy arrays to minimize serialization overhead between processes.

    Parameters
    ----------
    args : tuple
        A tuple containing the arguments: (i, post_sample_dict, As_range,
        lmn_all, n_samples_multiplier, suppress_warnings).
    likelihood : bilby.core.likelihood.GravitationalWaveTransient
        The bilby likelihood object. Passed separately for efficiency.

    Returns
    -------
    tuple
        A tuple containing: (amp_phase_samples_array, mean_sample_array,
        ln_Evi_A0, ln_Evi_A1). All are NumPy arrays for maximum efficiency.
    """
    i, post_sample, As_range, lmn_all, n_samples_multiplier, suppress_warnings = args
    A_min, A_max = As_range
    n_modes = len(lmn_all)

    if suppress_warnings:
        warnings.filterwarnings("ignore")

    try:
        # Set the likelihood's non-linear parameters to the current posterior sample
        likelihood.parameters.update(post_sample)

        # This call computes the marginalized likelihood over linear parameters
        # and also sets attributes like E_A (mean) and inv_M (covariance).
        _ = likelihood.log_likelihood()
        E_A, inv_M = likelihood.A_all, likelihood.inv_M

        # Evidence under uniform priors on A-space components (ln_Evi_A0)
        sign, logdet = np.linalg.slogdet(inv_M)
        ln_Evi_A0 = (-n_modes * np.log(A_max**2 - A_min**2) +
                     0.5 * logdet + n_modes * np.log(2.))

        # Evidence under uniform priors on amplitudes and phases (ln_Evi_A1)
        # This requires Monte Carlo integration.
        try:
            ll_A = ss.multivariate_normal(mean=E_A, cov=inv_M)
        except np.linalg.LinAlgError:
            ll_A = ss.multivariate_normal(mean=E_A, cov=inv_M, allow_singular=True)
        Ns = n_samples_multiplier * n_modes**2
        A_samples = ll_A.rvs(size=Ns)

        # Convert samples to amplitude-phase space
        amp_phase_samples = LinearParameterEvidenceCalculator.A_to_Amp_phase_array(A_samples)

        # Calculate importance weights for Monte Carlo integration
        amplitude_cols = amp_phase_samples[:, ::2]
        log_weights = -np.sum(np.log(amplitude_cols + 1e-10), axis=1)
        norm_factor1 = ssl.logsumexp(log_weights)

        ln_Evi_A1 = (norm_factor1 - n_modes * np.log(A_max - A_min) +
                     0.5 * logdet - np.log(Ns))
        
        # Calculate the mean (max likelihood) amplitude-phase values
        mean_sample = LinearParameterEvidenceCalculator.A_to_Amp_phase_array(
            np.array([E_A])
        )[0]  # Extract the single row from the 2D array

        return amp_phase_samples, mean_sample, ln_Evi_A0, ln_Evi_A1

    except Exception as e:
        bcu.logger.error(f'Error in worker {i}: {str(e)}')
        raise


def ln_Evi_lp(result, likelihood, As_range=(0., 50.), npool=20,
              n_samples_multiplier=100, suppress_warnings=True):
    """
    A convenience function to calculate the log evidence for linear parameters.

    This function initializes and runs the LinearParameterEvidenceCalculator class.

    Parameters
    ----------
    result : bilby.core.result.Result
        The result object from a bilby run.
    likelihood : bilby.core.likelihood.GravitationalWaveTransient
        The bilby likelihood object used in the run.
    As_range : tuple, optional
        The prior range (min, max) for amplitude parameters. Defaults to (0., 50.).
    npool : int, optional
        Number of processes for parallel computation. Defaults to 20.
    n_samples_multiplier : int, optional
        Multiplier for the number of Monte Carlo samples. Defaults to 100.
    suppress_warnings : bool, optional
        If True, suppresses bilby warnings. Defaults to True.

    Returns
    -------
    bilby.core.result.Result
        The updated result object with evidence calculations.
    """
    calculator = LinearParameterEvidenceCalculator(
        likelihood=likelihood,
        As_range=As_range,
        n_samples_multiplier=n_samples_multiplier,
        npool=npool,
        suppress_warnings=suppress_warnings
    )

    return calculator.calculate_evidence(result)


# Performance comparison utilities
class PerformanceTimer:
    """A simple utility class for timing code blocks."""
    
    def __init__(self):
        self.timings = {}

    @contextmanager
    def time(self, operation_name):
        """A context manager to time an operation."""
        start = time.time()
        yield
        end = time.time()
        self.timings[operation_name] = end - start

    def report(self):
        """Prints a report of all timed operations."""
        print("Performance Report:")
        print("-" * 40)
        for operation, duration in self.timings.items():
            print(f"{operation}: {duration:.4f} seconds")
        print("-" * 40)
        total = sum(self.timings.values())
        print(f"Total time: {total:.4f} seconds")


# Example usage with performance monitoring:
"""
# Assume `result` and `likelihood` objects are already defined from a bilby run.

# Basic usage with performance monitoring
timer = PerformanceTimer()

with timer.time("Evidence Calculation"):
    # The calculator is now initialized with the likelihood object
    calculator = LinearParameterEvidenceCalculator(
        likelihood=likelihood,
        As_range=(0., 50.),
        n_samples_multiplier=50,  # Reduced for faster testing
        npool=8,
        suppress_warnings=True
    )
    # The calculate_evidence method no longer needs the likelihood passed to it
    updated_result = calculator.calculate_evidence(result)

timer.report()

# Alternatively, use the convenience function
updated_result_from_func = ln_Evi_lp(result, likelihood, n_samples_multiplier=50)
"""


def calculate_amplitude_significance(amplitude_samples: np.ndarray) -> float:
    """
    Calculates the significance of an amplitude being greater than zero, 
    based on a series of posterior samples.

    The method strictly follows the steps described in the "IIB3. Computation of 
    amplitude significance" section of the "GW250114" supplement.
    
    Args:
        amplitude_samples (np.ndarray): A 1D NumPy array containing posterior 
                                        samples of the amplitude A.

    Returns:
        float: The significance in terms of Gaussian standard deviations (σ).
    """
    
    # Step 1: Create an estimate of the posterior probability density using 
    # Gaussian Kernel Density Estimation (KDE).
    # The paper mentions using the automatic bandwidth estimation from 
    # scipy.stats.gaussian_kde.
    k_raw = ss.gaussian_kde(amplitude_samples)

    # Step 2: Handle the boundary condition at A=0 by reflecting the density 
    # about the origin.
    # Defines k(A) = k_raw(A) + k_raw(-A) (Equation 9).
    # This ensures that the new density function k(A) integrates to 1 over 
    # the domain A >= 0.
    def k(A):
        return k_raw(A) + k_raw(-A)

    # Step 3: Evaluate the estimated posterior density k(A) at A=0.
    # k(0) returns a single-element array, so we take the first element.
    k_at_zero = k(0)[0]

    # Step 4: Compute 1-p, which is more numerically stable than computing p 
    # directly [cite: 190-191].
    # 1-p is the integral of k(A) over the region where k(A) <= k(0) 
    # (Equation 12) .
    
    # Define the function to be integrated. We only return the value of k(A) 
    # if it's <= k_at_zero.
    def integrand(A):
        vals = k(A)
        # Use np.where for an efficient conditional check
        return np.where(vals <= k_at_zero, vals, 0)

    # Perform the numerical integration from 0 to infinity using the `quad` function.
    # `quad` handles the infinite upper limit well.
    one_minus_p, _ = quad(integrand, 0, np.inf, limit=100)

    # Step 5: Convert 1-p to a significance x in units of σ.
    # The paper defines x such that the integral of a standard normal 
    # distribution from -x to x is equal to p (Equation 11).
    # A more numerically stable way to compute this is by using the tail 
    # probability (Equation 13):
    # the integral from -∞ to -x is (1-p)/2.
    # This is equivalent to finding the point where the cumulative distribution 
    # function (CDF) is (1-p)/2.
    # The ppf (percent point function) is the inverse of the cdf.
    # Therefore, -x = norm.ppf((1-p)/2).
    tail_probability = one_minus_p / 2.0
    
    # Handle edge cases where one_minus_p is very close to 0 or 1.
    if tail_probability == 0:
        return np.inf
    if tail_probability >= 0.5:
        return 0.0
        
    x_sigma = -ss.norm.ppf(tail_probability)

    return x_sigma

"""
### Example Usage

Below is an example of how to use this function. We generate two sets of mock data: one representing a low-significance signal (with significant posterior density near zero) and another representing a high-significance signal (with posterior density far from zero).

```python
# Set the random seed for reproducible results
np.random.seed(42)

# --- Example 1: Low-significance case ---
# Assume the amplitude posterior distribution is close to zero, 
# e.g., a Rayleigh distribution.
low_significance_samples = np.random.rayleigh(scale=0.8, size=20000)

# --- Example 2: High-significance case ---
# Assume the amplitude posterior distribution is far from zero.
# A normal distribution with mean 5, std 1, truncated at zero.
high_significance_samples_raw = np.random.normal(loc=5, scale=1, size=20000)
high_significance_samples = high_significance_samples_raw[high_significance_samples_raw > 0]


# Calculate the significance
low_sig = calculate_amplitude_significance(low_significance_samples)
high_sig = calculate_amplitude_significance(high_significance_samples)

print(f"Result for the low-significance case: {low_sig:.2f}σ")
print(f"Result for the high-significance case: {high_sig:.2f}σ")

# For comparison, we can calculate the z-score under a Gaussian approximation
# z = μ / σ
# The paper notes this simpler method can be inaccurate as it doesn't account for 
# non-Gaussian shapes.
z_score_low = np.mean(low_significance_samples) / np.std(low_significance_samples)
z_score_high = np.mean(high_significance_samples) / np.std(high_significance_samples)
print("\n--- Z-score (μ/σ) comparison ---")
print(f"Low significance (z-score): {z_score_low:.2f}σ")
print(f"High significance (z-score): {z_score_high:.2f}σ")
"""
