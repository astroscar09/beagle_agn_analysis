from astropy.io import fits
import spectres as spectres
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d
from astropy.convolution import convolve, convolve_fft
from scipy.integrate import trapezoid
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import pickle
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

cobaltblue = '#2e37fe'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13


def read_agn_emission(hdu):

    df = Table(hdu_agn_mup100_default['AGN EMISSION'].data).to_pandas()
    return df
    

def read_hii_emission(hdu):
    df = Table(hdu_agn_mup100_default['HII EMISSION'].data).to_pandas()
    return df

def read_dispersion_curve():

    tab = Table.read('data/jwst_nirspec_prism_disp.fits')
    rwave, R = tab['WAVELENGTH'], tab['R']

    return rwave, R

rwave, R = read_dispersion_curve()

df = pd.read_csv('/Users/oac466/Downloads/MIRI_LRS_Resolution_Kendrew2015.txt', skiprows = 2, names = ['wave', 'R'])
miri_arr = df.wave.values
miri_R = df.R.values

def read_nirspec_data():

    nirspec = Table.read('data/spectra/1395_Only_Spec.fits')
    wave, flux, fluxerr = nirspec['wav']*1e10, nirspec['flux'], nirspec['fluxerr']

    return wave, flux, fluxerr


def get_posterior_df(hdu):

    posterior_df = Table(hdu['POSTERIOR PDF'].data).to_pandas()

    return posterior_df

def compute_BIC(df):
    
    num_parameters = df.shape[1] - 4
    n_sample = df.n_data.values[0]
    return BIC_Criterion(df['ln_likelihood'], num_parameters, n_sample)

def BIC_Criterion(lnL, k, n):
    return -2 * lnL + k * np.log(n)


def compute_residuals(wave, flux, fluxerr, marg_wave, marg_sed):

    fluxes = np.interp(marg_wave, wave, flux)
    fluxerrs = np.interp(marg_wave, wave, fluxerr)
    
    residuals = (marg_sed - fluxes)/fluxerrs
    
    return residuals

def compute_bic_region(df, marg_wave, marg_sed, wave, flux, fluxerr, plot = None, plot_file = None):

    #model_wave, model_flux = read_marg_sed(bgl_output)
    residual = compute_residuals(wave, flux, fluxerr, marg_wave, marg_sed)
    chi2 = residual**2
    mask = (marg_wave > 19000) & (marg_wave < 26500)
    L = np.exp(-0.5*np.sum(chi2[mask]))
    lnL = np.log(L)
    n = mask.sum()
    k = len(df.columns) - 4
    bic = BIC_Criterion(lnL, k, n)
    return bic

def grab_bic(hdu, wave, flux, fluxerr, kind = 'median'):

    marg_wave, marg_seds = grab_marginal(hdu)
    df = get_posterior_df(hdu)

    if kind == 'median':
        bic = compute_bic_region(df, marg_wave[0], np.median(marg_seds, axis = 0), wave, flux, fluxerr)
    else:

        bic = np.array([compute_bic_region(df, marg_wave[0], x, wave, flux, fluxerr) for x in marg_seds])
    
    
    return bic

def read_beagle_output_file(filename):
    
    hdu = fits.open(filename)

    return hdu

def agn_emission_table(hdu):

    agn_emit_df = Table(hdu['AGN EMISSION'].data).to_pandas()

    return agn_emit_df
    
def read_continuum_info(hdu):

    cont_wave, cont_sed = hdu['CONTINUUM SED WL'].data[0][0], hdu['CONTINUUM SED'].data

    return cont_wave, cont_sed

def read_agn(hdu):

    agn_sed = hdu['AGN FULL SED'].data
    
    return agn_sed

def read_full_sed(hdu):

    full_sed_wave = hdu['FULL SED WL'].data[0][0]
    full_sed = hdu['FULL SED'].data

    return full_sed_wave, full_sed

def grab_marginal(hdu):

    marg_wav, marg_sed = hdu['MARGINAL SED WL'].data[0], hdu['MARGINAL SED'].data

    return marg_wav[0], marg_sed

def generate_wav_obs_fine_vectorized(model_wave, r_wave, r_curve, oversample):
    min_wave = 0.95 * model_wave[0]
    max_wave = 1.05 * model_wave[-1]

    wav_grid = np.linspace(min_wave, max_wave, 1000000)
    R_interp = interp1d(r_wave, r_curve, bounds_error=False, fill_value='extrapolate')
    R_vals = R_interp(wav_grid)

    dwavs = wav_grid / np.abs(R_vals) / oversample
    lam = np.cumsum(dwavs) + min_wave
    lam = lam[lam < max_wave]
    return lam

def test_convolve_model_spec(r_wave, r_curve, model_wave, model_flux, wave_obs_spectra, oversample = 10, f_LSF = 1):

    '''
    #oversample = define some over-sampling value, I usually use 10 
    #f_LSF = fudge factor increasing the resolution by a constant factor, based on e.g. the position in the slit 
             (I usually use ~1.3 or so for point sources)
    
    '''

    mask = np.isfinite(model_flux)

    model_wave = np.array(model_wave[mask])
    model_flux = np.array(model_flux[mask])
    
    
    # Construct wavelength grid (x is an array of wavelengths with equal separations in R space)
    #print('generating grid')
    wav_obs_fine = generate_wav_obs_fine_vectorized(model_wave, r_wave, r_curve, oversample)
    # Construct your model on the grid of wavelengths you defined above 
    model = interp1d(model_wave, model_flux, 
                     bounds_error=False, fill_value='extrapolate')
    
    flux_fine = model(wav_obs_fine)
    
    # Convolve with the resolution curve
    sigma_pix = oversample/2.35/f_LSF  # sigma width of kernel in pixels
    k_size = 5*int(sigma_pix+1)
    x_kernel_pix = np.arange(-k_size, k_size+1) 
    kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2)) # construct Gaussian kernel
    kernel /= np.trapezoid(kernel)  # Explicitly normalise kernel
    #print('performing the convolution')
    flux_fine = convolve(flux_fine, kernel)
    
    # Downsample to the wavelength grid of the instrument (here I used Adam Carnall's spectres package) 
    #need to map onto actual observed spectral data

    #print('Downsampling')
    flux = spectres.spectres(wave_obs_spectra, wav_obs_fine, flux_fine, fill=0, verbose=False)

    return flux

def convolve_model_spec(r_wave, r_curve, model_wave, model_flux, wave_obs_spectra, oversample = 10, f_LSF = 1):

    '''
    #oversample = define some over-sampling value, I usually use 10 
    #f_LSF = fudge factor increasing the resolution by a constant factor, based on e.g. the position in the slit 
             (I usually use ~1.3 or so for point sources)
    
    '''

    mask = np.isfinite(model_flux)

    model_wave = np.array(model_wave[mask])
    model_flux = np.array(model_flux[mask])
    
    
    # Construct wavelength grid (x is an array of wavelengths with equal separations in R space)
    #print('generating grid')
    wav_obs_fine = [0.95*model_wave[0]]
    while wav_obs_fine[-1] < 1.05*model_wave[-1]:
        R_val = np.interp(wav_obs_fine[-1], r_wave, r_curve)
        dwav = wav_obs_fine[-1]/np.abs(R_val)/oversample
        wav_obs_fine.append(wav_obs_fine[-1] + dwav)
    #print('done_generating grid')
    wav_obs_fine = np.array(wav_obs_fine)

    #print('interpolating model onto grid')
    # Construct your model on the grid of wavelengths you defined above 
    model = interp1d(model_wave, model_flux, 
                     bounds_error=False, fill_value='extrapolate')
    
    flux_fine = model(wav_obs_fine)

    # plt.figure()
    # plt.plot(wav_obs_fine, flux_fine)
    # plt.show()
    
    # Convolve with the resolution curve
    sigma_pix = oversample/2.35/f_LSF  # sigma width of kernel in pixels
    k_size = 5*int(sigma_pix+1)
    x_kernel_pix = np.arange(-k_size, k_size+1) 
    kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2)) # construct Gaussian kernel
    kernel /= np.trapezoid(kernel)  # Explicitly normalise kernel
    #print('performing the convolution')
    flux_fine = convolve(flux_fine, kernel)
    
    # Downsample to the wavelength grid of the instrument (here I used Adam Carnall's spectres package) 
    #need to map onto actual observed spectral data

    #print('Downsampling')
    flux = spectres.spectres(wave_obs_spectra, wav_obs_fine, flux_fine, fill=0, verbose=False)

    return flux

def get_maxlnl_z(hdu):

    max_idx = np.argmax(Table(hdu['POSTERIOR PDF'].data)['ln_likelihood'])
    redshift = Table(hdu['GALAXY PROPERTIES'].data)['redshift'][max_idx]

    return redshift
############


def convolve_model_to_grid(model_wave, model_flux, wav_obs_fine, oversample, f_LSF=1):
    """
    Convolve model spectrum to the fixed high-res grid.
    
    Returns:
        flux_fine (np.ndarray): convolved model flux on wav_obs_fine
    """
    mask = np.isfinite(model_flux)
    model_interp = interp1d(model_wave[mask], model_flux[mask], bounds_error=False, fill_value='extrapolate')
    flux_fine = model_interp(wav_obs_fine)

    sigma_pix = oversample / 2.35 / f_LSF
    k_size = 5 * int(sigma_pix + 1)
    x_kernel_pix = np.arange(-k_size, k_size + 1)
    kernel = np.exp(-(x_kernel_pix ** 2) / (2 * sigma_pix ** 2))
    kernel /= np.trapezoid(kernel)

    flux_fine = convolve(flux_fine, kernel)
    return flux_fine

def downsample_to_obs_grid(wave_obs, wav_obs_fine, flux_fine):
    return spectres.spectres(wave_obs, wav_obs_fine, flux_fine, fill=0, verbose=False)


def convolve_single_model(data_wave, model_wave, model_flux, oversample = 75):
    wav_obs_fine = generate_wav_obs_fine_vectorized(model_wave, rwave*1e4, R, oversample)
    flux_fine = convolve_model_to_grid(model_wave,model_flux, wav_obs_fine, oversample)
    flux_obs = downsample_to_obs_grid(data_wave, wav_obs_fine, flux_fine)
    return flux_obs

def convolve_single_model_miri(data_wave, model_wave, model_flux, oversample = 75):
    wav_obs_fine = generate_wav_obs_fine_vectorized(model_wave, miri_arr*1e4, miri_R, oversample)
    flux_fine = convolve_model_to_grid(model_wave,model_flux, wav_obs_fine, oversample)
    flux_obs = downsample_to_obs_grid(data_wave, wav_obs_fine, flux_fine)
    return flux_obs
    
def process_model(args):
    model_wave, model_flux, wave_obs, wav_obs_fine = args
    flux_fine = convolve_model_to_grid(model_wave, model_flux, wav_obs_fine, 75)
    flux_obs = downsample_to_obs_grid(wave_obs, wav_obs_fine, flux_fine)
    return flux_obs

def test_pool_convolve(model_wave, seds, data_wave, oversample = 75):

    # Precompute this once
    wav_obs_fine = generate_wav_obs_fine_vectorized(model_wave, rwave*1e4, R, oversample)
    
    # This is fixed for all models
    shared_args = (model_wave, wav_obs_fine)
    
    # List of (model_flux, wave_obs) tuples
    model_fluxes_and_obs = [(model_wave, sed, data_wave, wav_obs_fine) for sed in seds]
    n_tasks = len(model_fluxes_and_obs)
    with Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(process_model, model_fluxes_and_obs), total=n_tasks))
    
    return np.array(results)
#######################
def convolve_all_seds_prism(wave, seds, data_wave, oversample = 75):
    print('precomputing grid')
    wav_obs_fine = generate_wav_obs_fine_vectorized(wave, rwave*1e4, R, oversample)

    
    convolved_SED = []
    for sed in tqdm(seds, total=seds.shape[0]):
        flux_fine = convolve_model_to_grid(wave, sed, wav_obs_fine, oversample)
        flux_obs = downsample_to_obs_grid(data_wave, wav_obs_fine, flux_fine)
        
        convolved_SED.append(flux_obs)

    return np.array(convolved_SED)

def convolve_all_seds_miri(wave, seds, data_wave, oversample = 75):

    print('precomputing grid')
    wav_obs_fine = generate_wav_obs_fine_vectorized(wave, miri_arr*1e4, miri_R, oversample)

    convolved_SED = []
    for sed in tqdm(seds, total=seds.shape[0]):
        flux_fine = convolve_model_to_grid(wave, sed, wav_obs_fine, oversample)
        flux_obs = downsample_to_obs_grid(data_wave, wav_obs_fine, flux_fine)
        convolved_SED.append(flux_obs)
    return np.array(convolved_SED)

def computing_agn_fraction(hdu, data_wave, zmax):

    zp1 = 1+zmax
    #cont_wave, _ = read_continuum_info(hdu)
    agn_seds = read_agn(hdu)
    full_sed_wave, full_seds = read_full_sed(hdu)

    print('Convolving all the FULL SEDs')
    full_conv_seds = test_pool_convolve(full_sed_wave*zp1, full_seds, data_wave)

    print('Convolving all the AGN SEDs')
    full_agn_conv_seds = test_pool_convolve(full_sed_wave*zp1, agn_seds, data_wave)

    agn_fraction = full_agn_conv_seds/full_conv_seds

    return agn_fraction, full_agn_conv_seds, full_conv_seds

def grab_spectral_region(data_wave, data_flux, line_wave, window):

    min_window = line_wave - window
    max_window = line_wave + window
    idx = np.where((min_window < data_wave) & (data_wave < max_window))[0]

    wave = data_wave[idx]
    flux = data_flux[idx]
    
    return wave, flux
    
def generate_component_specs_plot(full_sed_wave, full_sed, full_agn_wave, full_agn_sed, data_wave, 
                                  data_flux, data_flux_err, max_lnl_z, marg_wave):
    
    #MAX_IDX = np.argmax(bgl_output['posterior']['probability'])

    zp1 = 1+max_lnl_z
    #print(bgl_output['z'])
    mult_factor = 1e20

    # Add rest-frame axis
    def obs_to_rest(w_obs):
        return w_obs * (1e4/ zp1)
    
    def rest_to_obs(w_rest):
        return w_rest * (zp1/1e4)

    og_full_sed = full_sed/zp1 #convert to observed frame
    og_agn_sed = full_agn_sed/zp1   #convert to observed frame

    og_agn_l16, og_agn_med, og_agn_u84 = np.percentile(og_agn_sed*mult_factor, q = (16, 50, 84), axis = 0)
    og_spec_l16, og_spec_med, og_spec_u84 = np.percentile(og_full_sed*mult_factor, q = (16, 50, 84), axis = 0)
    
    og_wave = full_sed_wave * zp1
    og_agn_wave = full_agn_wave * zp1

    print('Convolving Full SED to Prism Resolution')
    spec_med = convolve_model_spec(rwave*1e4, R, og_wave, og_spec_med, marg_wave, oversample = 75)
    spec_l16 = convolve_model_spec(rwave*1e4, R, og_wave, og_spec_l16, marg_wave, oversample = 75)
    spec_u84 = convolve_model_spec(rwave*1e4, R, og_wave, og_spec_u84, marg_wave, oversample = 75)

    print('Convolving AGN SED to Prism Resolution')
    agn_med = convolve_model_spec(rwave*1e4, R, og_agn_wave, og_agn_med, marg_wave, oversample = 75)
    agn_l16 = convolve_model_spec(rwave*1e4, R, og_agn_wave, og_agn_l16, marg_wave, oversample = 75)
    agn_u84 = convolve_model_spec(rwave*1e4, R, og_agn_wave, og_agn_u84, marg_wave, oversample = 75)
    
    data_wave = data_wave/1e4
    #og_mask = (og_wave >= conv_wave[0]) & (og_wave <= conv_wave[-1])

    diff_spec_l16 = spec_l16 - agn_l16
    diff_spec_med = spec_med - agn_med
    diff_spec_u84 = spec_u84 - agn_u84
    
    # Create a figure
    fig = plt.figure(figsize=(25, 10), dpi = 200)
    
    # Create a grid with 3 rows and 4 columns using gridspec
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0])

    ax.step(marg_wave/1e4, agn_med, label = 'Model AGN SED', color = 'dodgerblue', where = 'mid', zorder = 200)
    ax.fill_between(marg_wave/1e4, agn_u84, agn_l16, step = 'mid', color = 'dodgerblue')
    
    ax.step(marg_wave/1e4, spec_med, label = 'Model AGN+SF SED', color = 'purple', where = 'mid', zorder = 202)
    ax.fill_between(marg_wave/1e4, spec_u84, spec_l16, step = 'mid', color = 'purple')
    
    ax.step(marg_wave/1e4, diff_spec_med, label = 'Stellar+HII Region SED', color = 'red', where = 'mid', zorder = 200)
    ax.fill_between(marg_wave/1e4, diff_spec_u84, diff_spec_l16, step = 'mid', color = 'red')

    ax.step(data_wave, data_flux*mult_factor, color = 'black', where = 'mid', alpha = 0.7, label = 'Data')
    ax.errorbar(data_wave, data_flux*mult_factor, yerr = data_flux_err*mult_factor, color = 'gray', fmt = 'none')

    ax.set_ylabel(r'F$_{\nu}$ [1$\times$10$^{-20}$ erg s$^{-1}$cm$^2$$\AA$$^{-1}$]', fontsize = 30)
    ax.set_xlabel(r'Observed Wavelength [$\mu$m]', fontsize = 30)
    obs_ticks = ax.get_xticks()
    ax.set_xticks(obs_ticks)
    ax.set_xticklabels(obs_ticks, fontsize = 30)
    ax.legend(fontsize = 30)

    ax2 = ax.secondary_xaxis('top', functions=(obs_to_rest, rest_to_obs))
    ax2.set_xlabel("Rest-Frame Wavelength [$\\AA$]", fontsize=30)
    ax2.tick_params(labelsize=30)
    # ax2 = ax.twiny()

    # obs_xlim = ax.get_xlim()
    # rest_xlim = [x * (1e4/ zp1) for x in obs_xlim]  # Convert to rest-frame
    # ax2.set_xlim(rest_xlim)

    # obs_ticks = ax.get_xticks()
    # rest_ticks = [tick * (1e4/ zp1) for tick in obs_ticks]

    # print(obs_ticks)
    # print(rest_ticks)
    
    # ax2.set_xticks(rest_ticks)
    # ax2.set_xticklabels([f"{tick:.0f}" for tick in rest_ticks], fontsize = 30)
    # ax2.set_xlabel(r"Rest Frame Wavelengths [$\AA$]", fontsize = 30)

    offset = (40)/1e4#*zp1
    ax.annotate(r'CIV', (1548.187*(zp1/1e4) - offset, 1.25), fontsize = 25)

    ax.annotate(r'CIII]', (1908.734	*(zp1/1e4) - offset, .8), fontsize = 25)

    ax.annotate(r'MgII', (2795.528*(zp1/1e4) - offset, .3), fontsize = 25)

    ax.annotate(r'HeII', (1640.420*(zp1/1e4) - offset, .9), fontsize = 25)

    ax.annotate(r'[NeIII]', ((3868.760*(zp1/1e4)) - 800, .75), fontsize = 25)
    
    ax.set_ylim(-0.25, 1.5)
    ax.set_xlim(1.5, 5.5)
    
    return fig, [ax, ax2]


def plot_spectra_and_zoom_ins(conv_full_sed, conv_agn_seds, 
                              full_sed_wave, full_seds, 
                              full_agn_wave, full_agn_seds, 
                              data_wave, data_flux, data_flux_err, miri_spectra,  
                              miri_conv_arr, z_max):

    LYA = 1215.67 #lya
    CIV = 1549 #CIV
    HEII = 1640 #HeII
    NV = 1240
    CIII = 1909
    MGII = 2798

    # Add rest-frame axis
    def obs_to_rest(w_obs):
        w_obs=w_obs*1e4
        return w_obs /zp1
    
    def rest_to_obs(w_rest):
        w_rest = w_rest*1e4
        return w_rest * (zp1)
    
    ymin_fraction = 0
    ymax_fraction = 0.6
    zp1 = 1+z_max
    mult_factor = 1e20
    conv_full_sed_l16, conv_full_sed_med, conv_full_sed_u84 = conv_full_sed[0]/zp1, conv_full_sed[1]/zp1, conv_full_sed[2]/zp1
    
    # Create a figure
    fig = plt.figure(figsize=(16, 8), dpi = 300)
    
    # Create a grid with 3 rows and 4 columns using gridspec
    gs = fig.add_gridspec(3, 3)

    rest_wave = data_wave/zp1
    
    #############################################################################################################
    #plotting the Spectrum and the model
    ax1 = fig.add_subplot(gs[0, :])  # Span all columns in the first row
    ax1.step(data_wave/1e4, conv_full_sed_med*mult_factor, color = 'black', label = 'SF+AGN Model Fit', zorder = 10, where = 'mid')
    ax1.fill_between(data_wave/1e4, conv_full_sed_u84*mult_factor, conv_full_sed_l16*mult_factor, 
                     color = 'purple', alpha = 0.5, zorder = 10, step = 'mid')
    
    ax1.step(data_wave/1e4, data_flux*mult_factor, color = 'gray', label = "Data", where = 'mid')
    ax1.errorbar(data_wave/1e4, data_flux*mult_factor, yerr = data_flux_err, color = 'red', fmt = 'none')
    ax1.set_ylim(-0.5, 1.4)
    ax1.legend(fontsize = 12)

    ax2 = ax1.secondary_xaxis('top', functions=(obs_to_rest, rest_to_obs))
    ax2.set_xlabel("Rest-Frame Wavelength [$\\AA$]", fontsize=15)
    ax2.tick_params(labelsize=15)
    ax1.set_xlabel(r'Observed Wavelength [$\mu$m]', fontsize=15)
    
    # ax11 = ax1.twiny()

    # obs_xlim = ax1.get_xlim()
    # rest_xlim = [x / zp1 for x in obs_xlim]  # Convert to rest-frame
    # ax11.set_xlim(rest_xlim)

    # obs_ticks = ax1.get_xticks()
    # rest_ticks = [tick / zp1 for tick in obs_ticks]
    # ax11.set_xticks(rest_ticks)
    # ax11.set_xticklabels([f"{tick:.0f}" for tick in rest_ticks])
    # ax11.set_xlabel(r"Rest Frame Wavelengths [$\AA$]", fontsize = 20)
    #############################################################################################################

    # l16_fagn_miri, med_fagn_miri, u84_fagn_miri = agn_fraction_miri[0], agn_fraction_miri[1], agn_fraction_miri[2]
    # l16_fagn_prism, med_fagn_prism, u84_fagn_prism = agn_fraction_nirspec[0], agn_fraction_nirspec[1], agn_fraction_nirspec[2]
    # start = max(data_wave.min(), miri_spectra.wave_AA.values.min())
    # end = min(data_wave.max(), miri_spectra.wave_AA.values.max())
    # mask2 = (miri_spectra.wave_AA.values >= start) & (miri_spectra.wave_AA.values <= end)
    
    # full_merged_wave = np.concatenate((data_wave, miri_spectra.wave_AA.values[~mask2]))
    # l16_fagn = np.concatenate((l16_fagn_prism, l16_fagn_miri[~mask2]))
    # med_fagn = np.concatenate((med_fagn_prism, med_fagn_miri[~mask2]))
    # u84_fagn = np.concatenate((u84_fagn_prism, u84_fagn_miri[~mask2]))
    
    # ax = fig.add_subplot(gs[1, :])
    
    # ax.step(full_merged_wave/zp1, med_fagn, where = 'mid', color = 'black')
    # ax.fill_between(full_merged_wave/zp1, u84_fagn, l16_fagn, step = 'mid', color = 'gray')
    # ax.set_xlabel(r'Rest Frame Wavelength [$\AA$]', fontsize = 15)
    # ax.set_ylabel('AGN Fraction', fontsize = 15)
    # ax.text(NV+30, 0.15, "NV", fontsize=14, color='black')
    # ax.text(CIV+50, 0.35, "CIV", fontsize=14, color='black')
    # ax.text(CIII, 0.3, "CIII", fontsize=14, color='black')
    # ax.text(4900, 0.6, "[OIII]4960,5007", fontsize=14, color='black')
    # ax.text(6564, 0.705, r"H$\alpha$", fontsize=14, color='black')
    # ax.text(MGII+45, 0.1, 'MgII]', fontsize=12, color='black')
    
    
    #agn_fraction
    #l16_fagn, med_fagn, u84_fagn = agn_fraction_nirspec[0], agn_fraction_nirspec[1], agn_fraction_nirspec[2]
    
    # Second row: 4 plots
    line1 = 1549 #CIV
    window1 = 40
    #ax2 = fig.add_subplot(gs[1, 0])

    #Grabbing AGN fraction
    #wave, lflux = grab_spectral_region(rest_wave, l16_fagn, line1, window1)
    #wave, flux = grab_spectral_region(rest_wave, med_fagn, line1, window1)
    #wave, uflux = grab_spectral_region(rest_wave, u84_fagn, line1, window1)
    
    #ax2.step(wave, flux, where = 'mid', color = 'purple', alpha = 0.5)
    #ax2.fill_between(wave, uflux, lflux, step = 'mid', color = 'dodgerblue')
    #ax2.set_ylim(ymin_fraction, ymax_fraction)

    #Grabbing Model Fit
    with open('Spectral_Fitting/CIV_Spectra.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    emcee_df = pd.read_csv('Spectral_Fitting/CIV_Fit_Results.txt', index_col = 0)
    SNR = np.median(emcee_df.Fluxes.values/np.std(emcee_df.Fluxes.values, ddof = 1))
    
    inputwave = data['input_wave']
    inputflux = data['input_spec']
    inputfluxerr = data['input_spec_err']

    full_sed_l16, full_sed_med, full_sed_u84 = np.percentile(full_seds,  q= (16, 50, 84), axis = 0)
    #full_sed_wvln = bgl_agn['full_sed_wvln']

    #obs_flux = convolve_single_model_prism(inputwave, full_sed_wave, full_sed_med)
    #wave, obs_flux = grab_spectral_region(rest_wave, conv_full_sed_med, line1, window1)
    obs_flux =  convolve_model_spec(rwave*(1e4), R, full_sed_wave*zp1, np.median(full_seds, axis = 0), inputwave*zp1, oversample=75)

    civ_fagn = 53.7#48.8
    
    ax2_twin = fig.add_subplot(gs[1, 0])
    ax2_twin.step(inputwave, inputflux/zp1 *mult_factor, where = 'mid', color = 'red', label = f'SNR: {SNR:.2f}')
    ax2_twin.errorbar(inputwave, inputflux/zp1 *mult_factor, yerr = inputfluxerr/zp1 *mult_factor, color = 'red', fmt = 'none')
    ax2_twin.step(inputwave, obs_flux *mult_factor/zp1, where = 'mid', color = 'purple', label = rf'f$_{{AGN}}$: {civ_fagn:.2f}%')
    #ax2_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    ax2_twin.legend(fontsize = 10)

    #ax2_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    ax2_twin.set_title(r'CIV$\lambda$1548+50', fontsize = 15)
    ax2_twin.minorticks_on()


    lineOIII = 1660 #OIII
    windowOIII = 50

    with open('Spectral_Fitting/HeII1640_OIII16660_1666_Spectra.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    #the plots the data
    emcee_df = pd.read_csv('Spectral_Fitting/HeII1640_OIII16660_1666_Fit_Results.txt', sep = ' ', index_col = 0)
    snr_oiii = np.median(emcee_df.Fluxes_2.values+emcee_df.Fluxes_3.values)/np.std(emcee_df.Fluxes_2.values+emcee_df.Fluxes_3.values, ddof = 1)

    inputwave = data['input_wave']
    inputflux = data['input_spec']
    inputfluxerr = data['input_spec_err']

    obs_flux =  convolve_model_spec(rwave*(1e4), R, full_sed_wave*zp1, np.median(full_seds, axis = 0), inputwave*zp1, oversample=75)
  
    #wave, obs_flux = grab_spectral_region(rest_wave, conv_full_sed_med, lineOIII, windowOIII)
    oiii_fagn = 23.04
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.step(inputwave, inputflux/zp1 *mult_factor, where = 'mid', color = 'red', label = f'SNR: {snr_oiii:.2f}')
    ax3.errorbar(inputwave, inputflux/zp1 *mult_factor, yerr = inputfluxerr/zp1 *mult_factor, color = 'red', fmt = 'none')
    ax3.step(inputwave, obs_flux*mult_factor/zp1, color = 'purple', where = 'mid', label = rf'f$_{{AGN}}$: {oiii_fagn:.2f}%')
    #ax3.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    ax3.legend()

    
    ax3.set_ylabel('')
    #ax3.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    ax3.set_title(r'OIII]$\lambda$1660+66', fontsize = 15)
    #ax3.set_ylim(, )
    ax3.minorticks_on()
    
    
    line2 = 1909 #CIII
    window2 = 40
    #ax3 = fig.add_subplot(gs[1, 1])

    #wave, lflux = grab_spectral_region(rest_wave, l16_fagn, line2, window2)
    #wave, flux = grab_spectral_region(rest_wave, med_fagn, line2, window2)
    #wave, uflux = grab_spectral_region(rest_wave, u84_fagn, line2, window2)
    
    
    #this plots the AGN fraction
    #ax3.step(wave, flux, where = 'mid', color = 'purple', alpha = 0.5, label = 'Model')
    #ax3.fill_between(wave, uflux, lflux, step = 'mid', color = 'dodgerblue')

    with open('Spectral_Fitting/CIII_Spectra.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    #the plots the data
    emcee_df = pd.read_csv('Spectral_Fitting/CIII_Fit_Results.txt', index_col = 0)
    SNR = np.median(emcee_df.Fluxes.values/np.std(emcee_df.Fluxes.values, ddof = 1))

    inputwave = data['input_wave']
    inputflux = data['input_spec']
    inputfluxerr = data['input_spec_err']

    #grabbing the model fit for the region
    #wave, flux = grab_spectral_region(rest_wave, med, line2, window2)

    #obs_flux = convolve_single_model_prism(inputwave, full_sed_wave, full_sed_med)    
    #wave, obs_flux = grab_spectral_region(rest_wave, conv_full_sed_med, line2, window2)
    obs_flux =  convolve_model_spec(rwave*(1e4), R, full_sed_wave*zp1, np.median(full_seds, axis = 0), inputwave*zp1, oversample=75)
    ciii_fagn = 25.41
    ax3_twin = fig.add_subplot(gs[1, 2])
    ax3_twin.step(inputwave, inputflux/zp1 *mult_factor, where = 'mid', color = 'red', label = f'SNR: {SNR:.2f}')
    ax3_twin.errorbar(inputwave, inputflux/zp1 *mult_factor, yerr = inputfluxerr/zp1 *mult_factor, color = 'red', fmt = 'none')
    ax3_twin.step(inputwave, obs_flux*mult_factor/zp1, color = 'purple', where = 'mid', label = rf'f$_{{AGN}}$: {ciii_fagn}%')
    ax3_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]", fontsize = 15)
    ax3_twin.legend()

    
    ax3_twin.set_ylabel('')
    ax3_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    ax3_twin.set_title('CIII]$\lambda$1908', fontsize = 15)
    ax3_twin.set_ylim(ymin_fraction, ymax_fraction)
    ax3_twin.minorticks_on()
    
    #########
    #Miri Spectral Range
    #########
    #full_l16, full_med, full_u84 = np.percentile(bgl_agn['full_sed'],  q= (16, 50, 84), axis = 0)
    #full_wave = bgl_agn['full_sed_wvln']
    miri_spectra = pd.read_csv('data/spectra/MIRI_SPECTRA_for_MCMC.txt', sep = ' ', index_col = 0)

    
    line5 =  5007#OIII
    window5 = 500
    OIII_fagn = 48.41
    #l16_fagn, med_fagn, u84_fagn =  agn_fraction_miri[0], agn_fraction_miri[1], agn_fraction_miri[2] #this should be miri
    rest_wave = miri_spectra.wave_AA_rest.values
    miri_flux = miri_spectra.flux_flam.values 
    miri_flux_err = miri_spectra.fluxerr_flam.values
    
    # Third row: 4 plots
    #ax6 = fig.add_subplot(gs[1, 2])

    #wave, lflux = grab_spectral_region(rest_wave, l16_fagn, line5, window5)
    #wave, flux = grab_spectral_region(rest_wave, med_fagn, line5, window5)
    #wave, uflux = grab_spectral_region(rest_wave, u84_fagn, line5, window5)
        
    #ax6.step(wave, flux, where = 'mid', color = 'purple', alpha = 0.5)
    #ax6.fill_between(wave, uflux, lflux, step = 'mid', color = 'dodgerblue')
    #ax6.set_ylim(ymin_fraction, ymax_fraction)

    with open('Spectral_Fitting/OIII5007_OIII4960_Spectra.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    emcee_df = pd.read_csv('Spectral_Fitting/OIII5007_OIII4960_Fit_Results.txt', index_col = 0)
    SNR = np.median(emcee_df.Combined_Fluxes.values/np.std(emcee_df.Combined_Fluxes.values, ddof = 1))

    inputwave = data['input_wave']
    inputflux = data['input_spec']
    inputfluxerr = data['input_spec_err']
    
    miri_conv_l16, miri_conv_med, miri_conv_u84 = miri_conv_arr[0], miri_conv_arr[1], miri_conv_arr[2]
    
    wave, flux = grab_spectral_region(rest_wave, miri_conv_med, line5, window5)
    
    input_wave, input_flux = grab_spectral_region(rest_wave, miri_flux, line5, window5)
    _, input_fluxerr = grab_spectral_region(rest_wave, miri_flux_err, line5, window5)
    

    ax6_twin = fig.add_subplot(gs[2, 0])
    ax6_twin.step(input_wave, input_flux*mult_factor, where = 'mid', color = 'red', label = f'SNR: {SNR:.2f}')
    ax6_twin.errorbar(input_wave, input_flux*mult_factor, yerr = input_fluxerr*mult_factor, color = 'red', fmt = 'none')
    
    ax6_twin.step(wave, flux/zp1*mult_factor, color = 'black', where = 'mid', label = rf'f$_{{AGN}}$: {OIII_fagn:.2f}%')
    ax6_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]", fontsize = 15)
    ax6_twin.legend()

        
    ax6_twin.set_title(r'[OIII]$\lambda$5007', fontsize = 15)
    ax6_twin.minorticks_on()
    #ax6.set_title('log(x + 1)')
    
    
    line6 =  6564#Halpha
    window6 = 500
    #ax7 = fig.add_subplot(gs[1, 3])
    
    #wave, lflux = grab_spectral_region(rest_wave, l16_fagn, line6, window6)
    #wave, flux = grab_spectral_region(rest_wave, med_fagn, line6, window6)
    #wave, uflux = grab_spectral_region(rest_wave, u84_fagn, line6, window6)
        
    #ax7.step(wave, flux, where = 'mid', color = 'purple', alpha = 0.5)
    #ax7.fill_between(wave, uflux, lflux, step = 'mid', color = 'dodgerblue')
    #ax7.set_ylim(ymin_fraction, ymax_fraction)

    with open('Spectral_Fitting/Halpha_Spectra.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    emcee_df = pd.read_csv('Spectral_Fitting/Halpha_Fit_Results.txt', index_col = 0)
    SNR = np.median(emcee_df.Fluxes.values/np.std(emcee_df.Fluxes.values, ddof = 1))

    inputwave = data['input_wave']
    inputflux = data['input_spec']
    inputfluxerr = data['input_spec_err']

    wave, flux = grab_spectral_region(rest_wave, miri_conv_med, line6, window6)
    input_wave, input_flux = grab_spectral_region(rest_wave, miri_flux, line6, window6)
    _, input_fluxerr = grab_spectral_region(rest_wave, miri_flux_err, line6, window6)

    ax7_twin = fig.add_subplot(gs[2, 1])
    ax7_twin.step(input_wave, input_flux*mult_factor, where = 'mid', color = 'red', label = f'SNR: {SNR:.2f}')
    ax7_twin.errorbar(input_wave, input_flux*mult_factor, yerr = input_fluxerr*mult_factor, color = 'red', fmt = 'none')

    fagn_halpha = 38.79
    
    ax7_twin.step(wave, flux/zp1 * mult_factor, color = 'black', where = 'mid', label = rf'f$_{{AGN}}$: {fagn_halpha:.2f}%')
    ax7_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    ax7_twin.legend()
    
    ax7_twin.set_ylabel('')
    ax7_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]", fontsize = 15)
    ax7_twin.set_title(r'H$\alpha$', fontsize = 15)
    ax7_twin.minorticks_on()

    #ax2.set_ylabel('AGN Fraction', fontsize = 15)
    ax2_twin.set_ylabel(r'F$_{\lambda}$ [1$\times 10^{-20}$ erg s$^{-1}$cm$^2$$\AA^{-1}$]', fontsize = 30)
    ax1.set_xlim(1.5, 5.5)
    #ax1.set_ylim()
    plt.tight_layout()
    plt.show()
    
    return fig, [ax1, ax2, ax2_twin, ax3_twin, ax6_twin, ax7_twin]


def compute_agn_fraction_percentiles_prism(hdu, wave):

    max_z = get_maxlnl_z(hdu)
    agn_seds = read_agn(hdu)
    full_sed_wave, full_seds = read_full_sed(hdu)
    zp1 = 1+max_z
    
    l16_agn, med_agn, u84_agn = np.percentile(agn_seds, q = (16, 50, 84), axis = 0)
    l16_full, med_full, u84_full = np.percentile(full_seds, q = (16, 50, 84), axis = 0)
    
    
    conv_agn_l16 = convolve_model_spec(rwave*1e4, R, full_sed_wave*zp1, l16_agn, wave, oversample = 75)
    conv_agn_med = convolve_model_spec(rwave*1e4, R, full_sed_wave*zp1, med_agn, wave, oversample = 75)
    conv_agn_u84 = convolve_model_spec(rwave*1e4, R, full_sed_wave*zp1, u84_agn, wave, oversample = 75)

    conv_model_l16 = convolve_model_spec(rwave*1e4, R, full_sed_wave*zp1, l16_full, wave, oversample = 75)
    conv_model_med = convolve_model_spec(rwave*1e4, R, full_sed_wave*zp1, med_full, wave, oversample = 75)
    conv_model_u84 = convolve_model_spec(rwave*1e4, R, full_sed_wave*zp1, u84_full, wave, oversample = 75)

    conv_agn = np.array([conv_agn_l16, conv_agn_med, conv_agn_u84])
    conv_full = np.array([conv_model_l16, conv_model_med, conv_model_u84])

    agn_fraction = conv_agn/conv_full

    return agn_fraction, conv_agn, conv_full

def compute_agn_fraction_percentiles_miri(hdu, wave):

    max_z = get_maxlnl_z(hdu)
    agn_seds = read_agn(hdu)
    full_sed_wave, full_seds = read_full_sed(hdu)
    zp1 = 1+max_z

    l16_agn, med_agn, u84_agn = np.percentile(agn_seds, q = (16, 50, 84), axis = 0)
    l16_full, med_full, u84_full = np.percentile(full_seds, q = (16, 50, 84), axis = 0)
    
    #convolve_model_spec(miri_arr*1e4, miri_R, full_sed_wave*13.34, np.median(full_seds, axis = 0), miri_spectra.wave_AA.values, 
    #                oversample = 75)
    conv_agn_l16 = convolve_model_spec(miri_arr*1e4, miri_R, full_sed_wave*zp1, l16_agn, wave, oversample = 75)
    conv_agn_med = convolve_model_spec(miri_arr*1e4, miri_R, full_sed_wave*zp1, med_agn, wave, oversample = 75)
    conv_agn_u84 = convolve_model_spec(miri_arr*1e4, miri_R, full_sed_wave*zp1, u84_agn, wave, oversample = 75)

    conv_model_l16 = convolve_model_spec(miri_arr*1e4, miri_R, full_sed_wave*zp1, l16_full, wave, oversample = 75)
    conv_model_med = convolve_model_spec(miri_arr*1e4, miri_R, full_sed_wave*zp1, med_full, wave, oversample = 75)
    conv_model_u84 = convolve_model_spec(miri_arr*1e4, miri_R, full_sed_wave*zp1, u84_full, wave, oversample = 75)

    conv_agn = np.array([conv_agn_l16, conv_agn_med, conv_agn_u84])
    conv_full = np.array([conv_model_l16, conv_model_med, conv_model_u84])

    agn_fraction = conv_agn/conv_full

    return agn_fraction, conv_agn, conv_full
    
def set_up_plot_panel(hdu, wave, flux, fluxerr, mass = 'Mup100', types = 'AGN', model = 'Fiducial'):
    
    #cont_wave, cont_flux = read_continuum_info(hdu)
    agn_sed = read_agn(hdu)
    full_sed_wave, full_sed = read_full_sed(hdu)
    zmax = get_maxlnl_z(hdu)

    zp1 = 1+zmax
        
    miri_spectra = pd.read_csv('data/spectra/MIRI_SPECTRA_for_MCMC.txt', sep = ' ', index_col = 0)
    miri_wave = miri_spectra.wave_AA.values

    #print('Convolving Full and AGN SED to prism resolution')
    agn_fraction, conv_agn, conv_full = compute_agn_fraction_percentiles_prism(hdu, wave)
    #print('Convolving Full and AGN SED to miri resolution')
    miri_agn_fraction, miri_conv_agn, miri_conv_full = compute_agn_fraction_percentiles_miri(hdu, miri_wave)
        
    fig, axes = plot_spectra_and_zoom_ins(conv_full, conv_agn, 
                                          full_sed_wave, full_sed, 
                                          full_sed_wave, agn_sed, 
                                          wave, flux, fluxerr, miri_spectra, 
                                          #agn_fraction, 
                                          #miri_agn_fraction, #should be miri 
                                          miri_conv_full, zmax)

    fig.savefig('Final_Plots/Main_Plot.png')



def main_plot_panel(hdu, wave, flux, fluxerr):
    set_up_plot_panel(hdu, wave, flux, fluxerr)



def main_plot_components(hdu, wave, flux, fluxerr, plot = None):

    print('Grabbing SEDs')
    max_z = get_maxlnl_z(hdu)
    agn_seds = read_agn(hdu)
    marg_wave, marg_seds = grab_marginal(hdu)
    full_sed_wave, full_seds = read_full_sed(hdu)

    #print(full_sed_wave)
    #print(full_seds.shape)
    
    fig, axes = generate_component_specs_plot(full_sed_wave, full_seds, full_sed_wave, agn_seds, wave, flux, fluxerr, max_z, marg_wave)

    if plot:
        plt.show()
    else:
        return fig, axes

##############################################################
def get_errors_log10(val, lower_err, upper_err):

    lower_error = np.log10(val/(val - lower_err))
    upper_error = np.log10((val + upper_err)/val)

    return lower_error, upper_error

def get_galaxy_properties(hdu):

    df = Table(hdu['GALAXY PROPERTIES'].data).to_pandas()
    return df

def get_errors_log10(val, lower_err, upper_err):

    lower_error = np.log10(val/(val - lower_err))
    upper_error = np.log10((val + upper_err)/val)

    return lower_error, upper_error

def mbh_mstar_relation():

    m_star = np.linspace(7.5, 11, 1000)
    lines = np.zeros((1000, 1000))
    for i in range(1000):
        coeffs = np.random.normal(loc = [1.12, -2.54], scale = [0.08, 0.75])
    
        equation = coeffs[0] * m_star + (coeffs[1])
        lines[i] = equation
    return m_star, lines

def BH_Mass_estimates(agn_emission_table, eddington_ratio = 1):

    logLacc = np.array(agn_emission_table['agn_lacc']).astype(float)
    L_erg_s = (10**logLacc) * u.erg/u.s
    L_Lsun = L_erg_s.to(u.Lsun)

    Mass = L_Lsun/(3.2e4 * (u.Lsun/u.M_sun))/ eddington_ratio
    return Mass

def local_relation_bhmass():

    alpha = 7.45
    alpha_err = 0.08
    beta = 1.05
    beta_err = 0.11

    rand_alphas = np.random.normal(alpha, alpha_err, size = 1000)
    rand_betas = np.random.normal(beta, beta_err, size = 1000)

    Mstar = np.logspace(7.5, 11, 1000)
    Mstar_matrix = np.log10(np.stack([Mstar for x in range(1000)])/1e11)

    lines = Mstar_matrix *rand_betas.reshape(-1, 1) + rand_alphas.reshape(-1, 1)
    #print(lines)
    l16, med, u84 = np.percentile(lines, q = (16, 50, 84), axis = 0)
    
    return l16, med, u84, Mstar, lines 


def read_agn_df():

    #agn_df = pd.read_csv('/Users/oac466/Downloads/jwst_blagn_10_28_2024_Machine_Readable.csv')
    agn_df = pd.read_csv('/Users/oac466/Downloads/jwst-blagn-table-2025-05-13.csv')
    agn_df = agn_df.drop(index = 0).reset_index()
    agn_df = agn_df.drop(columns=['index'])

    agn_df['m_bh'] = agn_df.m_bh.values.astype(float)
    agn_df['m_star'] = agn_df.m_star.values.astype(float)
    agn_df['m_bh_err_plus'] = agn_df.m_bh_err_plus.values.astype(float)
    agn_df['m_bh_err_minus'] = agn_df.m_bh_err_minus.values.astype(float)
    agn_df['m_star_err_plus'] = agn_df.m_bh_err_plus.values.astype(float)
    agn_df['m_star_err_minus'] = agn_df.m_bh_err_minus.values.astype(float)
    
    return agn_df

def BHMass_Stellar_Mass_Relation(stellar_mass, 
                                 bhmasses_eddington, bhmasses_pt5_eddington, bhmasses_pt1_eddington, 
                                 agn_df, lines_highz, lr_l16, lr_med, lr_u84, 
                                 mstar_lowz, mstar_highz):

    fig, ax = plt.subplots(figsize = (8, 5), dpi = 150)

    ##########
    #Plotting out Data
    ########
    ax.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_eddington)), 
                marker = '*', color = 'red', s = 100, zorder = 99, 
                edgecolors='white', label = r'GHZ2 $\eta$: 1')
    ax.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_eddington)), 
                marker = '*', color = 'red', s = 200, zorder = 1)
    
    ax.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_pt5_eddington)), 
                marker = '*', color = 'blue', s = 100, zorder = 99, 
                edgecolors='white', label = r'GHZ2 $\eta$: 0.5')
    ax.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_pt5_eddington)), 
                marker = '*', color = 'blue', s = 200, zorder = 1)
    
    ax.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_pt1_eddington)), 
                marker = '*', color = 'black', s = 100, zorder = 99, 
                edgecolors='white', label = r'GHZ2 $\eta$: 0.1')
    ax.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_pt1_eddington)), 
                marker = '*', color = 'black', s = 200, zorder = 1)
    
    ###########
    #Plotting data from the literature
    ############
    ax.scatter(np.log10(agn_df['m_star']), np.log10(agn_df['m_bh']), color = '#170b3b')
    
    #############
    #Plotting up the High z relation
    ax.fill_between(mstar_highz, np.percentile(lines_highz, axis = 0,  q = 84),  np.percentile(lines_highz, axis = 0,  q = 16), 
                     color = '#603f8b', alpha = 0.5, label = 'High-z BH Relation')
    ax.plot(mstar_highz, np.median(lines_highz, axis = 0), color = 'black')
    
    ###########
    #Plotting up the local relation
    ax.fill_between(np.log10(mstar_lowz), lr_u84,  lr_l16, color = 'salmon', alpha = 0.5, label = 'Local BH Relation')
    
    ax.plot(np.log10(mstar_lowz), lr_med, color = 'black')
    
    ax.set_xlabel(r'log$_{10}$(M$_{*}$ [M$_{\odot}$])', fontsize = 15)
    ax.set_ylabel(r'log$_{10}$(M$_{BH}$[M$_{\odot}$])', fontsize = 15)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    
    ax.legend(fontsize = 10, ncols = 2)
    
    plt.show()


def plot_BHMass(bhmasses):

    plt.figure(figsize = (10, 5), dpi = 150)
    plt.hist(np.log10(bhmasses.value), bins = 20, color = 'purple', edgecolor="white", zorder = 99)
    plt.xlabel(r'Black Hole Masses [log$_{10}$(M/M$_{\odot}$)]', fontsize = 15)
    plt.ylabel(r'Counts', fontsize = 20)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder = 1)

    plt.show()

def plot_BHMass_ranges(bh_mass_edd1, bh_mass_edd05, bh_mass_edd01):

    fig, ax = plt.subplots(figsize = (10, 5), dpi = 150)
    ax.hist(np.log10(bh_mass_edd1.value), bins = 20, color = 'purple', edgecolor="white", zorder = 99, label = r'$\eta$: 1')
    ax.hist(np.log10(bh_mass_edd05.value), bins = 20, color = 'red', edgecolor="white", zorder = 99, label = r'$\eta$: 0.5')
    ax.hist(np.log10(bh_mass_edd01.value), bins = 20, color = 'pink', edgecolor="white", zorder = 99, label = r'$\eta$: 0.1')
    ax.set_xlabel(r'Black Hole Masses [log$_{10}$(M/M$_{\odot}$)]', fontsize = 15)
    ax.set_ylabel(r'Counts', fontsize = 20)
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder = 1)
    ax.legend()
    plt.show()

def plot_mstellar_mbh(hdu):
    df = get_posterior_df(hdu)
    total_mass = df['mass'].values
    agn_df = read_agn_df()
    agn_df['m_star'] = agn_df['m_star'].astype(float)
    agn_df['m_bh'] = agn_df['m_bh'].astype(float)

    lr_l16, lr_med, lr_u84, mstar_lowz, _ = local_relation_bhmass()
    mstar_highz, lines_highz = mbh_mstar_relation()
    
    bhmasses_eddington     =  BH_Mass_estimates(df, eddington_ratio = 1).value
    bhmasses_pt5_eddington =  BH_Mass_estimates(df, eddington_ratio = 0.5).value
    bhmasses_pt1_eddington =  BH_Mass_estimates(df, eddington_ratio = 0.1).value

    gal_prop = Table(hdu['Galaxy Properties'].data).to_pandas()
    stellar_masses = gal_prop['M_star'].values

    BHMass_Stellar_Mass_Relation(stellar_masses, 
                                 bhmasses_eddington, bhmasses_pt5_eddington, bhmasses_pt1_eddington, 
                                 agn_df, lines_highz, lr_l16, lr_med, lr_u84, 
                                 mstar_lowz, mstar_highz)

def plot_bhmass_vs_redshift(agn_emission_df):

    bh_mass_edd1 = np.log10(np.median(BH_Mass_estimates(agn_emission_df, eddington_ratio = 1).value))
    bh_mass_edd05 = np.log10(np.median(BH_Mass_estimates(agn_emission_df, eddington_ratio = 0.5).value))
    bh_mass_edd01 = np.log10(np.median(BH_Mass_estimates(agn_emission_df, eddington_ratio = 0.1).value))

    agn_df = read_agn_df()

    #getting BH Errors Log
    MBH = agn_df.m_bh.values.astype(float)
    MBH_err_plus = agn_df.m_bh_err_plus.values.astype(float)
    MBH_err_minus = agn_df.m_bh_err_minus.values.astype(float)

    low_errors, upper_errors = get_errors_log10(MBH, MBH_err_minus, MBH_err_plus)
    MBH_log10 = np.log10(MBH)
    
    fig, ax = plt.subplots(figsize = (10, 5), dpi = 150)

    ax.errorbar(agn_df.z_spec.values, MBH_log10, 
                yerr = [low_errors, upper_errors], color = 'gray', fmt = 'none', capsize = 3)
    ax.scatter(agn_df.z_spec.values, MBH_log10, 
                marker = 'o', color = 'gray', label = 'Literature')
    ax.scatter(12.34, bh_mass_edd1, color = 'red', marker = '*', s = 250, label = r'$\eta$ = 1')
    ax.scatter(12.34, bh_mass_edd05, color = 'blue', marker = '*', s = 250, label = r'$\eta$ = 0.5')
    ax.scatter(12.34, bh_mass_edd01, color = 'black', marker = '*', s = 250, label = r'$\eta$ = 0.1')
    
    ax.scatter(12.34, bh_mass_edd1, color = 'red', marker = '*', s = 200, edgecolors='white')
    ax.scatter(12.34, bh_mass_edd05, color = 'blue', marker = '*', s = 200, edgecolors='white')
    ax.scatter(12.34, bh_mass_edd01, color = 'black', marker = '*', s = 200, edgecolors='white')
    ax.set_ylabel(r'log$_{10}$(M$_{BH}$/M$_{\odot}$)', fontsize = 15)
    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in')
    ax.set_xlabel('Redshift', fontsize = 15)
    ax.legend(fontsize = 10, ncols = 2)
    plt.show()


def plot_BH_growth(hdu):

    agn_df = read_agn_df()

    lower_error = agn_df.m_bh_err_minus.values
    upper_error = agn_df.m_bh_err_plus.values

    agn_emission_df = get_posterior_df(hdu)

    bh_mass_edd1 = np.median(BH_Mass_estimates(agn_emission_df, eddington_ratio = 1).value)
    bh_mass_edd05 = np.median(BH_Mass_estimates(agn_emission_df, eddington_ratio = 0.5).value)
    bh_mass_edd01 = np.median(BH_Mass_estimates(agn_emission_df, eddington_ratio = 0.1).value)

    bh_mass_edd2 = np.median(BH_Mass_estimates(agn_emission_df, eddington_ratio = 2).value)

    zgrid = np.linspace(35, 4, 1000)
    N = 5000
    
    # DCBH: initial mass ~1e4-1e5 Msun at z~10-20, growing at the Eddington rate
    edd_dcbh = np.zeros((N,len(zgrid)))
    log_seed_mass = np.random.uniform(4,5,size=N)
    seed_mass = np.power(10., log_seed_mass)
    z_init = np.random.uniform(10,20, size=N)
    tgrid = cosmo.age(zgrid).to(u.Gyr).value
    t_init = cosmo.age(z_init).to(u.Gyr).value
    
    lam, epsilon = 1, 0.1
    tau = (4.5e8 * epsilon/(lam*(1-epsilon)))/1e9
    for i in tqdm(range(N)):
        edd_dcbh[i,:] = seed_mass[i] * np.exp((tgrid-t_init[i])/tau)
    
    # stellar/PopIII: initial mass ~10-300 Msun at z~15-30, growing at the eddington limit
    edd_stellar = np.zeros((N,len(zgrid)))
    log_seed_mass = np.random.uniform(1,2.47,size=N)
    seed_mass = np.power(10., log_seed_mass)
    z_init = np.random.uniform(15,30, size=N)
    tgrid = cosmo.age(zgrid).to(u.Gyr).value
    t_init = cosmo.age(z_init).to(u.Gyr).value
    lam, epsilon = 1, 0.1
    tau = (4.5e8 * epsilon/(lam*(1-epsilon)))/1e9
    for i in tqdm(range(N)):
        edd_stellar[i,:] = seed_mass[i] * np.exp((tgrid-t_init[i])/tau)


    logMBH = agn_df.m_bh.values
    log_mstar = agn_df.m_star.values
    
    fig, ax = plt.subplots(figsize = (12, 6), dpi = 150)
    ax.fill_between(zgrid, np.percentile(edd_dcbh, 16, axis=0), np.percentile(edd_dcbh, 84, axis=0),
                    edgecolor='none', facecolor='blue', alpha=0.15, zorder=-2000)
    ax.plot(zgrid, np.percentile(edd_dcbh, 50, axis=0), c='blue', lw=1, ls='--', label='Overmassive DCBH Eddington Growth')
    ax.fill_between(zgrid, np.percentile(edd_stellar, 16, axis=0), np.percentile(edd_stellar, 84, axis=0), 
                    edgecolor='none', facecolor='orange', alpha=0.3, zorder=-2000)
    ax.plot(zgrid, np.percentile(edd_stellar, 50, axis=0), c='orange', lw=1, ls='--', label='Stellar DCBH Eddington Growth')

    
    ax.scatter(agn_df.z_spec.values, logMBH, marker = 'o', s= 10, color = 'black', label = 'Literature')
    ax.errorbar(agn_df.z_spec.values, logMBH, yerr = [lower_error, 
                                                       upper_error], fmt='none', marker= '.', color = 'black', capsize=2)

    
    ax.scatter(12.34, bh_mass_edd1, color = 'red', marker = '*', s = 250, label = r'$\eta$ = 1')
    ax.scatter(12.34, bh_mass_edd05, color = 'blue', marker = '*', s = 250, label = r'$\eta$ = 0.5')
    ax.scatter(12.34, bh_mass_edd01, color = 'black', marker = '*', s = 250, label = r'$\eta$ = 0.1')
    ax.scatter(12.34, bh_mass_edd2, color = 'purple', marker = '*', s = 250, label = r'$\eta$ = 2')
    #plt.scatter(12.34, 5.9, color = 'orange')
    
    ax.scatter(12.34, bh_mass_edd1, color = 'red', marker = '*', s = 200, edgecolors='white')
    ax.scatter(12.34, bh_mass_edd05, color = 'blue', marker = '*', s = 200, edgecolors='white')
    ax.scatter(12.34, bh_mass_edd01, color = 'black', marker = '*', s = 200, edgecolors='white')
    ax.scatter(12.34, bh_mass_edd2, color = 'purple', marker = '*', s = 200, edgecolors='white')
    
    ax.set_yscale('log')
    ax.set_xlim(4, 13)
    ax.invert_xaxis()
    ax.set_ylim(1e4, 1e10)
    ax.legend(ncols = 2)
    ax.set_xlabel('Redshift', fontsize = 15)
    ax.set_ylabel(r'Black Hole Mass [M$_{\odot}$]', fontsize = 20)
    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in')
    
    return fig, ax 
