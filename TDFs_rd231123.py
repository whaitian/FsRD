import os, sys, lal, json
import bilby
import numpy as np
import scipy.linalg as sl
import bilby.core.utils as bcu
from gwpy.timeseries import TimeSeries as gtts
from bilby.gw.detector import InterferometerList
base_name = os.path.expanduser('~')
from RD_likelihood import RD_TDFs_Transient, TD_WaveformGenerator, QNMs_lmn
from RD_reconstruct_linear_params import ln_Evi_lp

## os.environ['PYCBC_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
# Init CONFIG
label_source = 'GW231123'
sampling_frequency = 1024.## *4

##########################################
"""
ii: modes considered in this test; "221" means 220; "221,201" means 220+200
jj: The time delay after the polarization peak.
"""
ii = '221'
jj = 10

label = '{0}_{1}M_0-4s'.format(ii, jj)
outdir = 'Fs_only2/{0}'.format(label)
##########################################
## bcu.setup_logger(outdir=outdir, label=label)
##########################################

time_of_event = 1384782888.6+0.0153731## Match to the polarization peak
det_names = ["H1", "L1"]
ifos = InterferometerList(det_names)
##########################################
from pycbc.detector import Detector as pdd
dets = {det_name:pdd(det_name) for det_name in det_names}

m0 = 298.*lal.MTSUN_SI
t0 = time_of_event+m0*jj
ra0, dec0 = 3.24, 0.25
delays0 = {detname: det.time_delay_from_earth_center(ra0, dec0, time_of_event) for detname,det in dets.items()}
##########################################
d0 = 0.4
slice_duration = 8.
f_filter = 20.
l_cov = int(d0*sampling_frequency)
time_se = {dn:{'start_time':t0+delays0[dn], 'end_time':t0+d0+0.1} for dn in det_names}
## print(time_se), exit()

f_name = './TD_data/PyCBC_psd_acfs_{0}_{1}-{2}Hz_t{3}s-v0.npy'.format(label_source, int(f_filter), int(sampling_frequency), int(slice_duration))
with open(f_name, 'rb') as g:
    mt = np.load(g, allow_pickle=True)

acfs = mt.item()['acfs1']
Notes = mt.item()['Notes']
strain_data = mt.item()['strain_data']
##########################################
cov_matrix = {name:{} for name in det_names}
for ifo in ifos:
    strain_td = strain_data[ifo.name].time_slice(time_se[ifo.name]['start_time'], time_se[ifo.name]['end_time'])[:l_cov]
    ifo_data = gtts(strain_td.numpy(), sample_rate=strain_td.get_sample_rate(), times=strain_td.sample_times.numpy(), channel='{0}:GWOSC-4KHZ_R1_STRAIN'.format(ifo.name))
    ifo.set_strain_data_from_gwpy_timeseries(ifo_data)
    L0 = sl.cholesky(sl.toeplitz(acfs[ifo.name][:l_cov]), lower=True)
    acfs[ifo.name] = sl.solve_triangular(L0, np.eye(L0.shape[0]), lower=True)


##########################################
kwargs0 = {'total mass':m0, 'start time':t0, 'ra':ra0, 'dec':dec0, 'slice_duration':d0, 'sampling_frequency':sampling_frequency}
bcu.logger.info('some important kwars are: {}'.format(kwargs0))
bcu.logger.info('The filename of the ACFs is: \n{}'.format(f_name))
bcu.logger.info('Notes for the ACFs are: \n{}'.format(Notes))
##########################################
# Set Priors
from bilby.gw.prior import PriorDict
from bilby.core.prior import Uniform, Sine, Cosine

priors = PriorDict()
priors['geocent_time'] = time_of_event
priors['ra'] = ra0
priors['dec'] = dec0
priors['psi'] = 2.23
priors['azimuthal'] = 0.
priors['inclination'] = Sine(name='inclination', boundary='reflective', latex_label='$\iota$')
##########################################
priors['final_mass'] = Uniform(name='final_mass', minimum=100, maximum=500, unit='$M_{\\odot}$', latex_label='$M_f$')
priors['final_spin'] = Uniform(name='final_spin', minimum=0., maximum=0.99, latex_label='$\chi_f$')

##########################################
# Create Waveform Generator
lmns = [item.strip() for item in ii.split(',')]
lmn_all = ['%s%d'%(lmn[:2],n) for lmn in lmns for n in range(int('%s'%lmn[-1]))]
bcu.logger.info(f'Modes considered in this running are: \n{lmn_all}')

waveform_arguments={'lmns': lmns, 'delta_t':1./sampling_frequency, 'harmonics':'spherical'}
bcu.logger.info('the waveform_arguments is: %s'%str(waveform_arguments))
waveform_generator = TD_WaveformGenerator(duration=1., sampling_frequency=sampling_frequency,\
    time_domain_source_model=QNMs_lmn, 
    waveform_arguments=waveform_arguments)
##########################################
likelihood = RD_TDFs_Transient(interferometers=ifos, waveform_generator=waveform_generator, acfs=acfs, priors=priors)

ss0 = {'nact':20, 'sample':'rwalk', 'bound':'live-multi', 'proposals':['diff', 'volumetric'], 'n_check_point':1000}
bcu.logger.info('The setting of the sampler is: {}.'.format(ss0))
result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000, nthreads=20, outdir=outdir, label=label, resume=True, **ss0)

result.plot_corner(parameters=['final_mass', 'final_spin', 'inclination'], filename='%s/%s_part_corner.png'%(outdir,label), **{'quantiles':[0.05, 0.95]})

updated_result = ln_Evi_lp(result, likelihood, npool=20)
updated_result.save_to_file(filename=label+'_p2', overwrite=True, outdir=outdir)
