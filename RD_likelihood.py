import lal, qnm, pykerr
import numpy as np
import pycbc.types as pt
from bilby.gw.detector import InterferometerList
from bilby.core.likelihood import Likelihood
from bilby.gw.waveform_generator import WaveformGenerator
from pycbc import waveform

def toeplitz_slogdet(r):

    """
    Compute the sign and natural logarithm of the determinant of a positive-definite
    symmetric Toeplitz matrix.

    The determinant is computed efficiently and recursively using an intermediate
    solution from the Levinson recursion algorithm.

    This method is adopted from Marano et al., "Fitting Earthquake Spectra:
    Colored Noise and Incomplete Data", Bulletin of the Seismological Society of
    America, Vol. 107, No. 1, 2017, doi: 10.1785/0120160030. Original code
    is available at:
    http://mercalli.ethz.ch/~marra/publications/2017_fitting_earthquake_spectra_colored_noise_and_incomplete_data/
    All credits go to the original authors.

    Parameters:
    ----------
    r : np.ndarray
        The first row of the symmetric Toeplitz matrix.

    Returns:
    -------
    tuple[float, float]
        A tuple containing the sign of the determinant and the natural log
        of the absolute value of the determinant.
    """

    n      = len(r)
    r_0    = r[0]
    r      = np.concatenate((r, np.array([r_0])))
    r     /= r_0 # normalize the system so that the T matrix has diagonal of ones
    logdet = n*np.log(np.abs(r_0))
    sign   = np.sign(r_0)**n

    if(n == 1): return (sign, logdet)

    # From this point onwards, is a modification of Levinson algorithm.
    y       = np.zeros((n,))
    x       = np.zeros((n,))
    b       = -r[1:n+1]
    r       = r[:n]
    y[0]    = -r[1]
    x[0]    = b[0]
    beta    = 1
    alpha   = -r[1]
    d       = 1 + np.dot(-b[0], x[0])
    sign   *= np.sign(d)
    logdet += np.log(np.abs(d))

    for k in range(0, n-2):

        beta     = (1 - alpha*alpha)*beta
        mu       = (b[k+1] - np.dot(r[1:k+2], x[k::-1])) /beta
        x[0:k+1] = x[0:k+1] + mu*y[k::-1]
        x[k+1]   = mu

        d        = 1 + np.dot(-b[0:k+2], x[0:k+2])
        sign    *= np.sign(d)
        logdet  += np.log(np.abs(d))

        if(k < n-2):
            alpha    = -(r[k+2] + np.dot(r[1:k+2], y[k::-1]))/beta
            y[0:k+1] = y[0:k+1] + alpha * y[k::-1]
            y[k+1]   = alpha

    return (sign, logdet)

######################################################
def Pycbc_ringdown_lmn(**kwargs):
    """
    Generate a time-domain ringdown waveform using PyCBC.

    This function serves as a wrapper for PyCBC's ringdown waveform models,
    allowing for the generation of 'kerr' or 'ftau' models. It supports
    the combination of multiple quasi-normal modes (QNMs).

    Note: The parameter 'tau' in the 'ftau' model represents the inverse of the
    damping time (1/tau). This is because the damping time is often very small,
    so setting priors on its inverse can be more numerically stable.

    Parameters:
    ----------
    **kwargs : dict
        A dictionary of waveform parameters passed to the PyCBC generator.
        Key parameters include:
        - final_mass : float, Remnant black hole mass in solar masses.
        - final_spin : float, Remnant black hole dimensionless spin.
        - lmns : list[str], List of modes to include (e.g., ['222'] for 220 and 221).
        - model : {'kerr', 'ftau'}, The underlying waveform model to use.
        - amp... : float, Amplitudes of the corresponding modes.
        - phi... : float, Phases of the corresponding modes.
        - f... : float, Frequencies of the modes (for 'ftau' model).
        - tau... : float, Inverse damping times of the modes (for 'ftau' model).

    Returns:
    -------
    dict[str, pycbc.types.TimeSeries]
        A dictionary containing the 'plus' and 'cross' polarizations of the
        generated waveform as PyCBC TimeSeries objects.
    """
    waveform_params = dict(taper=False,final_mass=20.,final_spin=None,lmns=['222'],amp220=1.,phi220=0.,inclination=0.,delta_t=1./2048,model='kerr')
    waveform_params.update(kwargs)
    model = waveform_params['model']
    lmn_all = ['%s%d'%(lmn[:2],n) for lmn in waveform_params['lmns'] for n in range(int('%s'%lmn[-1]))]
    if len(lmn_all)>1:
        lmn_all.remove('220') ## except 220
        if model=='ftau':
            waveform_params['tau_220'] = 1./waveform_params['tau_220']
        for lmn in lmn_all:
            waveform_params['amp%s'%lmn] = waveform_params['amp%s'%lmn]/waveform_params['amp220']

    waveform_params['amp220'] = waveform_params['amp220']*1.e-20
    if model=='kerr':
        hplus, hcross = waveform.ringdown.get_td_from_final_mass_spin(**waveform_params)
    elif model=='ftau':
        for lmn in lmn_all:
            waveform_params['tau_%s'%str(lmn)] = 1./waveform_params['tau_%s'%str(lmn)]

        hplus, hcross = waveform.ringdown.get_td_from_freqtau(**waveform_params)

    return {'plus':hplus, 'cross':hcross}

######################################################

def spher_harms(harmonics='spherical', l=None, m=None, n=0,
                inclination=0., azimuthal=0.,
                spin=None):
    r"""
    Compute the spin-weighted (-2) spherical or spheroidal harmonics.

    This function returns the harmonic polarizations for the +m and -m modes,
    which are essential for constructing the ringdown waveform.

    Parameters
    ----------
    harmonics : {'spherical', 'spheroidal'}, optional
        The type of harmonic to generate. Defaults to 'spherical'.
    l : int, optional
        The polar mode number. Required for both harmonic types.
    m : int, optional
        The azimuthal mode number. Required for both harmonic types.
    n : int, optional
        The overtone number. Only used for spheroidal harmonics. Defaults to 0.
    inclination : float, optional
        The inclination angle in radians. Used for both harmonic types.
        Defaults to 0.
    azimuthal : float, optional
        The azimuthal angle in radians. Used for both harmonic types.
        Defaults to 0.
    spin : float, optional
        The dimensionless spin of the black hole. Required for spheroidal
        harmonics. Ignored otherwise.

    Returns
    -------
    xlm : complex
        The complex value of the harmonic for the +m mode.
    xlnm : complex
        The complex value of the harmonic for the -m mode.
    """
    if harmonics == 'spherical':
        xlm = lal.SpinWeightedSphericalHarmonic(inclination, azimuthal, -2,
                                                l, m)
        xlnm = lal.SpinWeightedSphericalHarmonic(inclination, azimuthal, -2,
                                                 l, -m)
    else:
        assert harmonics == 'spheroidal', 'The harmonics must be either spherical or spheroidal.'
        if spin is None:
            raise ValueError("must provide a spin for spheroidal harmonics")
        xlm = pykerr.spheroidal(inclination, spin, l, m, n, -2, phi=azimuthal)
        xlnm = pykerr.spheroidal(inclination, spin, l, -m, n, -2, phi=azimuthal)
    return xlm, xlnm

def QNMs_lmn(**kwargs):
    """
    Calculate the complex frequencies of quasi-normal modes (QNMs).

    This function computes the frequencies and damping times for specified QNM modes
    based on the remnant black hole's properties (mass and spin). It can also
    incorporate user-defined fractional deviations from the General Relativity
    predictions for these values.

    Parameters:
    ----------
    **kwargs : dict
        A dictionary of parameters. Key parameters include:
        - final_mass : float, Remnant black hole mass in solar masses.
        - final_spin : float, Remnant black hole dimensionless spin.
        - lmns : list[str], List of modes to compute (e.g., ['221', '201']).
        - harmonics : {'spherical', 'spheroidal', ...}, The harmonic type. Spheroidal
          harmonics account for mode mixing in spinning black holes.
        - model : {'qnm', 'ftau', 'pykerr'}, The underlying package or model to
          use for QNM calculations. 'pykerr' is recommended for consistency.
        - delta_f{lmn} : float, optional, Fractional deviation to apply to the
          frequency of a specific mode.
        - delta_tau{lmn} : float, optional, Fractional deviation to apply to the
          damping time of a specific mode.

    Returns:
    -------
    dict[str, dict]
        A dictionary containing another dictionary, 'Omegas', which maps each
        mode label (e.g., '220') to its complex frequency (omega = 2*pi*f - i/tau).
    """
    waveform_params = dict(final_mass=100.,final_spin=0.68,lmns=['222'],harmonics='spherical',azimuthal=0.,model='qnm')
    waveform_params.update(kwargs)
    lmn_all = ['%s%d'%(lmn[:2],n) for lmn in waveform_params['lmns'] for n in range(int('%s'%lmn[-1]))]
    ## omegas = lal.CreateCOMPLEX16Vector(len(lmn_all))
    omegas = dict()
    for lmn in lmn_all:
        if waveform_params['harmonics']=='spherical' or waveform_params['harmonics']=='arbitrary':
            if waveform_params['model']=='qnm':
                bbh = qnm.modes_cache(-2, int(lmn[0]), int(lmn[1]), int(lmn[2]))
                omega, _, _ = bbh(a=waveform_params['final_spin'])
                omega0 = omega/(lal.MTSUN_SI*float(waveform_params['final_mass']))
                f0, tau0 = omega0.real, 1./abs(omega0.imag)
            elif waveform_params['model']=='ftau':
                f0, tau0 = 2*np.pi*waveform_params['f_{}'.format(str(lmn))], 1./waveform_params['tau_{}'.format(str(lmn))]
            else:
                assert waveform_params['model']=='pykerr', 'The waveform model can only be qnm or pykerr.'
                f0 = 2*np.pi*pykerr.qnmfreq(waveform_params['final_mass'], waveform_params['final_spin'], int(lmn[0]), int(lmn[1]), int(lmn[2]))
                tau0 = pykerr.qnmtau(waveform_params['final_mass'], waveform_params['final_spin'], int(lmn[0]), int(lmn[1]), int(lmn[2]))

            if ('delta_f{}'.format(lmn) in waveform_params) and ('delta_tau{}'.format(lmn) in waveform_params):
                omegas[lmn] = f0+waveform_params['delta_f{}'.format(lmn)]*f0-1.j/(tau0+waveform_params['delta_tau{}'.format(lmn)]*tau0)
            elif 'delta_f{}'.format(lmn) in waveform_params:
                ## omegas.data[i] = f0+waveform_params['delta_f{}'.format(lmn)]-1.j/tau0
                omegas[lmn] = f0+waveform_params['delta_f{}'.format(lmn)]*f0-1.j/tau0
            elif 'delta_tau{}'.format(lmn) in waveform_params:
                ## omegas.data[i] = f0-1.j/(tau0+waveform_params['delta_tau{}'.format(lmn)])
                omegas[lmn] = f0-1.j/(tau0+waveform_params['delta_tau{}'.format(lmn)]*tau0)
            else:
                ## omegas.data[i] = f0-1.j/tau0
                omegas[lmn] = f0-1.j/tau0
        else:
            assert waveform_params['harmonics']=='spheroidal', 'The harmonics can only be spherical or spheroidal'
            if waveform_params['model']=='qnm':
                ## This is inconsistent with pykerr, I suggest to use pykerr
                bbh = qnm.modes_cache(-2, int(lmn[0]), int(lmn[1]), int(lmn[2]))
                omega, _, _ = bbh(a=waveform_params['final_spin'])
                omega0 = omega/(lal.MTSUN_SI*float(waveform_params['final_mass']))
                f0, tau0 = omega0.real, 1./abs(omega0.imag)
                bbh1 = qnm.modes_cache(-2, int(lmn[0]), -int(lmn[1]), int(lmn[2]))
                omega, _, _ = bbh1(a=waveform_params['final_spin'])
                omega1 = omega/(lal.MTSUN_SI*float(waveform_params['final_mass']))
                f1, tau1 = omega1.real, 1./abs(omega1.imag)
            else:
                assert waveform_params['model']=='pykerr', 'The waveform model can only be qnm or pykerr.'
                f0 = 2.*np.pi*pykerr.qnmfreq(waveform_params['final_mass'], waveform_params['final_spin'], int(lmn[0]), int(lmn[1]), int(lmn[2]))
                tau0 = pykerr.qnmtau(waveform_params['final_mass'], waveform_params['final_spin'], int(lmn[0]), int(lmn[1]), int(lmn[2]))
                f1 = 2.*np.pi*pykerr.qnmfreq(waveform_params['final_mass'], waveform_params['final_spin'], int(lmn[0]), -int(lmn[1]), int(lmn[2]))
                tau1 = pykerr.qnmtau(waveform_params['final_mass'], waveform_params['final_spin'], int(lmn[0]), -int(lmn[1]), int(lmn[2]))
            if ('delta_f{}'.format(lmn) in waveform_params) and ('delta_tau{}'.format(lmn) in waveform_params):
                omegas[lmn] = f0+waveform_params['delta_f{}'.format(lmn)]*f0-1.j/(tau0+waveform_params['delta_tau{}'.format(lmn)]*tau0)
                omegas['n'+lmn] = f1+waveform_params['delta_f{}'.format(lmn)]*f1-1.j/(tau1+waveform_params['delta_tau{}'.format(lmn)]*tau1)
            elif 'delta_f{}'.format(lmn) in waveform_params:
                omegas[lmn] = f0+waveform_params['delta_f{}'.format(lmn)]*f0-1.j/tau0
                omegas['n'+lmn] = f1+waveform_params['delta_f{}'.format(lmn)]*f1-1.j/tau1
            elif 'delta_tau{}'.format(lmn) in waveform_params:
                omegas[lmn] = f0-1.j/(tau0+waveform_params['delta_tau{}'.format(lmn)]*tau0)
                omegas['n'+lmn] = f1-1.j/(tau1+waveform_params['delta_tau{}'.format(lmn)]*tau1)
            else:
                omegas[lmn] = f0-1.j/tau0
                omegas['n'+lmn] = f1-1.j/tau1

    return {'Omegas':omegas}

######################################################
class TD_WaveformGenerator(WaveformGenerator):

    def time_domain_strain(self, parameters=None):
        """
        Generate time-domain waveform polarizations using the source model.

        This method calls the provided `time_domain_source_model` with the
        given parameters to produce the 'plus' and 'cross' polarizations.
        It includes a caching mechanism to avoid redundant waveform generation
        if the parameters have not changed since the last call.

        Parameters:
        ----------
        parameters : dict, optional
            A dictionary of parameters to be passed to the source model.
            If None, uses the parameters from the last call.

        Returns:
        -------
        dict or None
            A dictionary containing the waveform polarizations (e.g., 'plus',
            'cross'). Returns None if the source model call fails.
        """

        if parameters == self._cache['parameters']:
            waveform_polarizations = self._cache['waveform']
        else:
            try:
                waveform_polarizations = self.time_domain_source_model(**{**parameters,**self.waveform_arguments})
            except RuntimeError:
                return None

        self._cache['waveform'] = waveform_polarizations
        self._cache['parameters'] = parameters.copy()

        return waveform_polarizations 

######################################################
class RD_TTD_Transient(Likelihood):

    def __init__(
        self, interferometers, waveform_generator, acfs, normalisations={'H1':0., 'L1':0., 'V1':0.}, priors=None, sky_average=False
    ):
        """
        Initialize a time-domain transient likelihood for ringdown signals.

        This likelihood is designed for analyzing ringdown signals in the time domain,
        assuming stationary Gaussian noise characterized by an auto-correlation
        function (ACF). It performs a direct time-domain correlation between the
        data and a fully-generated waveform template.

        Parameters:
        ----------
        interferometers : bilby.gw.detector.InterferometerList
            A list of interferometer objects containing the strain data.
        waveform_generator : bilby.gw.waveform_generator.WaveformGenerator
            The waveform generator object to produce the signal templates.
        acfs : dict
            A dictionary mapping interferometer names to their corresponding
            whitening filters derived from the noise auto-correlation function.
        normalisations : dict, optional
            A dictionary of normalization constants for the likelihood calculation
            for each interferometer.
        priors : bilby.core.prior.PriorDict, optional
            A dictionary of priors for the waveform parameters.
        sky_average : bool, optional
            If True, use a sky-averaged antenna response instead of projecting
            the waveform onto each detector for specific sky locations. Defaults to False.
        """

        self.waveform_generator = waveform_generator
        self.acfs = acfs
        self.normalisations = normalisations
        super(RD_TTD_Transient, self).__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.priors = priors
        self.sky_average = sky_average
        self._meta_data = {}

    def noise_log_likelihood(self):
        """
        Calculate the noise log-likelihoo .

        Returns:
        -------
        float
            The natural logarithm of the noise likelihood.
        """
        log_l = 0.
        if 'nll' in self._meta_data.keys():
            return self._meta_data['nll']

        for ifm in self.interferometers:
            signal_ifo = ifm.strain_data.to_pycbc_timeseries()
            s_s = (self.acfs[ifm.name])@(signal_ifo.data)
            log_l -= 0.5*sum(s_s*s_s)+self.normalisations[ifm.name]

        self._meta_data['nll'] = log_l
        return log_l

    def get_pycbc_detector_response_td(self, ifo, waveform_polarizations, start_t):
        """
        Project a waveform onto a detector and align it in time.

        This function calculates the detector's response to a gravitational wave
        by applying the antenna patterns and accounting for the time delay from
        the geocenter. The resulting signal is then shifted to match the
        start time of the detector's data segment.

        Parameters:
        ----------
        ifo : bilby.gw.detector.Interferometer
            The interferometer object.
        waveform_polarizations : dict[str, pycbc.types.TimeSeries]
            A dictionary containing the 'plus' and 'cross' waveform polarizations.
        start_t : float
            The GPS start time of the analysis segment.

        Returns:
        -------
        pycbc.types.TimeSeries
            The time-domain strain signal as observed in the interferometer.
        """
        signal = {}
        for mode in waveform_polarizations.keys():
            det_response = ifo.antenna_response(self.parameters['ra'], self.parameters['dec'], self.parameters['geocent_time'], self.parameters['psi'], mode)
            signal[mode] = waveform_polarizations[mode] * det_response

        signal_ifo = sum(signal.values())
        shift_t = ifo.time_delay_from_geocenter(self.parameters['ra'], self.parameters['dec'], self.parameters['geocent_time'])
        dt = shift_t+self.parameters['geocent_time']-start_t.__float__()+signal_ifo.end_time.__float__()
        signal_ifo.prepend_zeros(int((dt+ifo.strain_data.duration)/signal_ifo.delta_t)) ## append zeros for roll
        signal_ifo.roll(int(round(dt/signal_ifo.delta_t, 0)))
        signal_ifo.start_time = start_t

        return signal_ifo

    def log_likelihood(self):
        """
        Calculate the log-likelihood for a given set of signal parameters.

        This method generates a waveform template based on the current parameters,
        projects it onto each detector, subtracts it from the data to form the
        residual, and then computes the likelihood of this residual assuming it is
        Gaussian noise with known statistical properties (defined by the ACFs).

        Returns:
        -------
        float
            The natural logarithm of the likelihood of the data given the signal model.
            Returns -inf for parameters that cause a generation error.
        """
        try:
            waveform_polarizations = self.waveform_generator.time_domain_strain(self.parameters)
        except RuntimeError:
            return np.nan_to_num(-np.inf)

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        log_l = 0.
        for ifm in self.interferometers:
            signal_ifo = ifm.strain_data.to_pycbc_timeseries()
            l0 = len(signal_ifo.data)
            if self.sky_average:
                waveform_det = (waveform_polarizations['plus']+waveform_polarizations['cross'])*self.sky_average
                waveform_det.append_zeros(l0)
                waveform_det.start_time = signal_ifo.start_time.__float__()
            else:
                waveform_det = self.get_pycbc_detector_response_td(ifm, waveform_polarizations, signal_ifo.start_time)
            s_h = signal_ifo.data-waveform_det.data[:l0]
            w_s_h = (self.acfs[ifm.name])@s_h
            log_l -= 0.5*sum(w_s_h*w_s_h)+self.normalisations[ifm.name]
        return log_l

    def log_likelihood_ratio(self):
        """
        Calculate the log-likelihood ratio (signal vs. noise).

        This is the difference between the log-likelihood for the signal-plus-noise
        hypothesis and the log-likelihood for the noise-only hypothesis. It represents
        the statistical evidence for the signal.

        Returns:
        -------
        float
            The natural logarithm of the likelihood ratio.
        """
        return self.log_likelihood() - self.noise_log_likelihood()

class RD_TDFs_Transient(Likelihood):
    """
    A semi-analytic, time-domain transient likelihood for ringdown signals.

    This likelihood is optimized for ringdown analysis by analytically marginalizing
    over the amplitudes and phases of the different modes. This is achieved by constructing a
    set of basis waveforms for the amplitudes&phases and then solving a linear system.
    This approach can significantly speed up sampling.

    The `acfs` parameter for this class should be the inverse of the auto-correlation
    matrix (e.g., its Cholesky decomposition), which acts as a whitening filter.

    Attributes:
    ----------
    interferometers : bilby.gw.detector.InterferometerList
        A list of interferometer objects.
    waveform_generator : bilby.gw.waveform_generator.WaveformGenerator
        The waveform generator for calculating QNM frequencies and damping times.
    acfs : dict
        A dictionary mapping interferometer names to their whitening matrices.
    """

    def __init__(
        self, interferometers, waveform_generator, acfs, priors=None, sky_average=False
    ):

        self.waveform_generator = waveform_generator
        self.acfs = acfs
        super(RD_TDFs_Transient, self).__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.priors = priors
        self.sky_average = sky_average ## This works for ET/LISA/TQ/Taiji
        self.harmonics = self.waveform_generator.waveform_arguments['harmonics']
        self.lmns = self.waveform_generator.waveform_arguments['lmns']
        self._meta_data = {}
        self.A_all = None
        self.M_all = None
        self.inv_M = None
        self.lmn_all = ['%s%d'%(lmn[:2],n) for lmn in self.lmns for n in range(int('%s'%lmn[-1]))] if self.harmonics=='spherical' or self.harmonics=='spheroidal' or self.harmonics=='arbitrary' else ['%s%d'%(lmn[:2],n) for lmn in self.lmns for n in range(int('%s'%lmn[-1]))]+['n%s%d'%(lmn[:2],n) for lmn in self.lmns for n in range(int('%s'%lmn[-1]))]

    def noise_log_likelihood(self):
        """
        Calculate the log-likelihood for the noise-only hypothesis.

        This is the likelihood of the data assuming it contains only noise.
        The result is cached after the first calculation.

        Returns:
        -------
        float
            The natural logarithm of the noise likelihood.
        """
        log_l = 0.
        if 'nll' in self._meta_data.keys():
            return self._meta_data['nll']

        for ifm in self.interferometers:
            signal_ifo = ifm.strain_data.to_pycbc_timeseries()
            s_s = (self.acfs[ifm.name])@(signal_ifo)
            log_l -= 0.5*sum(s_s*s_s)

        self._meta_data['nll'] = log_l
        return log_l

    def reconstruct_detected_waveform_td(self, ifo, Omegas, start_t):
        """
        Construct the time-domain basis waveforms for amplitude marginalization.

        For each mode, this function generates the fundamental basis functions
        (e.g., sine and cosine terms) that will be linearly combined with amplitude
        parameters to form the complete signal model.

        Parameters:
        ----------
        ifo : bilby.gw.detector.Interferometer
            The interferometer object.
        Omegas : dict
            A dictionary of complex QNM frequencies for all modes.
        start_t : float
            The lal.LIGOTime_GPS start time of the analysis segment.

        Returns:
        -------
        list[np.ndarray]
            A list of the time-domain basis waveforms. The number of waveforms
            is 2 or 4 times the number of modes, depending on the harmonic type.
        """
        det_response = {}
        for mode in ['plus', 'cross']:
            if self.sky_average:
                det_response[mode] = self.sky_average## 1.5/np.sqrt(5) for ET, which have three detectors, triangular frame and we consider the sky average case.
            else:
                det_response[mode] = ifo.antenna_response(self.parameters['ra'], self.parameters['dec'], self.parameters['geocent_time'], self.parameters['psi'], mode)

        delta_t = 1./ifo.strain_data.sampling_frequency
        t_list = np.arange(0., ifo.strain_data.duration, delta_t)
        omegas = {lmn:Omegas[lmn].real for lmn in Omegas.keys()}
        rtaus = {lmn:abs(Omegas[lmn].imag) for lmn in Omegas.keys()}
        waves = []
        for lmn in self.lmn_all:
            if self.harmonics=='arbitrary':
                A_plus = det_response['plus']
                A_cross = det_response['cross']
                ht1 = 1.e-20*A_plus*np.cos(omegas[lmn]*(t_list))*np.exp(-t_list*rtaus[lmn])
                ht2 = 1.e-20*A_plus*np.sin(omegas[lmn]*(t_list))*np.exp(-t_list*rtaus[lmn])
                ht3 = 1.e-20*A_cross*np.sin(omegas[lmn]*(t_list))*np.exp(-t_list*rtaus[lmn])
                ht4 = 1.e-20*A_cross*np.cos(omegas[lmn]*(t_list))*np.exp(-t_list*rtaus[lmn])
                waves.append(ht1)
                waves.append(ht2)
                waves.append(ht3)
                waves.append(ht4)
            elif self.harmonics=='spherical':
                fspin = self.parameters['final_spin'] if 'final_spin' in self.parameters.keys() else None
                Y_lm, Y_lnm = spher_harms(harmonics=self.harmonics, l=int(lmn[0]), m=int(lmn[1]), n=int(lmn[2]), inclination=self.parameters['inclination'], azimuthal=self.parameters['azimuthal'], spin=fspin)
                lm_p = Y_lm.real+(-1)**int(lmn[0])*Y_lnm.real
                lm_c = Y_lm.real-(-1)**int(lmn[0])*Y_lnm.real
                A_plus = det_response['plus']*lm_p
                A_cross = det_response['cross']*lm_c
                ht1 = 1.e-20*(A_plus*np.cos(omegas[lmn]*(t_list))+A_cross*np.sin(omegas[lmn]*(t_list)))*np.exp(-t_list*rtaus[lmn])
                ht2 = -1.e-20*(A_plus*np.sin(omegas[lmn]*(t_list))-A_cross*np.cos(omegas[lmn]*(t_list)))*np.exp(-t_list*rtaus[lmn])
                waves.append(ht1)
                waves.append(ht2)
            elif self.harmonics=='spheroidal':
                Y_lm, Y_lnm = spher_harms(harmonics=self.harmonics, l=int(lmn[0]), m=int(lmn[1]), n=int(lmn[2]), inclination=self.parameters['inclination'], azimuthal=self.parameters['azimuthal'], spin=self.parameters['final_spin'])
                h1 = Y_lm*np.exp((1.j*omegas[lmn]-rtaus[lmn])*t_list)
                h2 = Y_lnm*np.exp((1.j*omegas['n'+lmn]-rtaus['n'+lmn])*t_list)
                h3 = h1+(-1)**int(lmn[0])*h2
                h4 = 1.j*(h1-(-1)**int(lmn[0])*h2)
                ht1 = 1.e-20*(det_response['plus']*np.real(h3)+det_response['cross']*np.imag(h3))
                ht2 = 1.e-20*(det_response['plus']*np.real(h4)+det_response['cross']*np.imag(h4))
                waves.append(ht1)
                waves.append(ht2)
            else:
                assert self.harmonics=='spheroidal2', 'the harmonics can only be spherical, spheroidal, or spheroidal2'
                Y_lm, Y_lnm = spher_harms(harmonics='spheroidal', l=int(lmn[0]), m=int(lmn[1]), n=int(lmn[2]), inclination=self.parameters['inclination'], azimuthal=self.parameters['azimuthal'], spin=self.parameters['final_spin'])
                A_plus = det_response['plus']*(Y_lm.__abs__())
                A_cross = det_response['cross']*(Y_lm.__abs__())
                nA_plus = det_response['plus']*(Y_lnm.__abs__())
                nA_cross = det_response['cross']*(Y_lnm.__abs__())
                ht1 = 1.e-20*(A_plus*np.cos(omegas[lmn]*(t_list))+A_cross*np.sin(omegas[lmn]*(t_list)))*np.exp(-t_list*rtaus[lmn])
                ht2 = -1.e-20*(A_plus*np.sin(omegas[lmn]*(t_list))-A_cross*np.cos(omegas[lmn]*(t_list)))*np.exp(-t_list*rtaus[lmn])
                waves.append(ht1)
                waves.append(ht2)
                ht3 = 1.e-20*(nA_plus*np.cos(omegas['n'+lmn]*(t_list))+nA_cross*np.sin(omegas['n'+lmn]*(t_list)))*np.exp(-t_list*rtaus['n'+lmn])
                ht4 = -1.e-20*(nA_plus*np.sin(omegas['n'+lmn]*(t_list))-nA_cross*np.cos(omegas['n'+lmn]*(t_list)))*np.exp(-t_list*rtaus['n'+lmn])
                waves.append(ht3)
                waves.append(ht4)

        return waves

    def F_matrix(self, Omegas):
        """
        Compute the components required for analytic amplitude marginalization.

        This function calculates two key quantities:
        1. The data vector `S`, which is the inner product of the whitened data
           with each whitened basis waveform.
        2. The Fisher matrix `M`, which is the inner product of each whitened
           basis waveform with every other one.

        Parameters:
        ----------
        Omegas : dict
            A dictionary of complex QNM frequencies for all modes.

        Returns:
        -------
        S_all : np.ndarray
            The combined data vector from all interferometers.
        M_all : np.ndarray
            The combined Fisher matrix from all interferometers.
        """
        l0 = 4*len(self.lmn_all) if (self.harmonics=='arbitrary' or self.harmonics=='spheroidal2') else 2*len(self.lmn_all)
        S_all = np.zeros(l0)
        M_all = np.zeros((l0, l0))
        for ifm in self.interferometers:
            hts = self.reconstruct_detected_waveform_td(ifm, Omegas, ifm.strain_data.start_time)
            Sd = np.zeros(l0)
            Md = np.zeros((l0, l0))
            wd = (self.acfs[ifm.name])@(ifm.strain_data.time_domain_strain)
            whs = [(self.acfs[ifm.name])@(hts[i]) for i in range(l0)]

            for i in range(l0):
                Sd[i] = sum(whs[i]*wd)
                for j in range(l0):
                    Md[i][j] = sum(whs[i]*whs[j]) if j>=i else Md[j][i]

            S_all = S_all+Sd
            M_all = M_all+Md
        return S_all, M_all

    def log_likelihood_ratio(self):
        """
        Calculate the marginalized log-likelihood ratio.

        This method computes the log-likelihood ratio after analytically maximizing
        over the linear amplitude parameters. The calculation uses the data vector `S`
        and the Fisher matrix `M` computed by `F_matrix`. The maximum likelihood
        amplitudes are given by A = M^-1 * S, and the marginalized log-likelihood
        ratio is 0.5 * (S . A).

        Returns:
        -------
        float
            The natural logarithm of the marginalized likelihood ratio. Returns -inf
            if the Fisher matrix is singular.
        """
        try:
            Omegas = self.waveform_generator.time_domain_strain(self.parameters)['Omegas']
        except RuntimeError:
            return np.nan_to_num(-np.inf)

        if Omegas is None:
            return np.nan_to_num(-np.inf)

        S_all, self.M_all = self.F_matrix(Omegas)
        ## log_l = 0.5*(S_all@(np.linalg.inv(M_all))@(S_all.T))
        try:
            self.inv_M = np.linalg.inv(self.M_all)
        except np.linalg.LinAlgError:
            return np.nan_to_num(-np.inf)
        self.inv_M = np.linalg.inv(self.M_all)
        self.A_all = S_all@self.inv_M
        log_l = 0.5*sum(self.A_all*S_all)
        return log_l

    def log_likelihood(self):
        """
        Calculate the full marginalized log-likelihood.

        This returns the total log-likelihood by adding the marginalized
        log-likelihood ratio to the noise log-likelihood.

        Returns:
        -------
        float
            The full, marginalized log-likelihood.
        """
        return self.log_likelihood_ratio() + self.noise_log_likelihood()

