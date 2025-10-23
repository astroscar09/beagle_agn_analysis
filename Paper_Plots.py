from Post_BEAGLE_Analysis import *
from Helper_Functions import *
import seaborn as sb
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
import pickle
from itertools import combinations

LYA = 1215.67 #lya
CIV = 1549 #CIV
HEII = 1640 #HeII
NV = 1240
CIII = 1909
MGII = 2798

def read_dispersion_curve():

    tab = Table.read('data/jwst_nirspec_prism_disp.fits')
    rwave, R = tab['WAVELENGTH'], tab['R']

    return rwave, R

def miri_resolution(microns):
    return 4603 - 128*microns + 10**(-7.4*microns)

miri_arr = np.linspace(5, 30, 1000)
miri_R = miri_resolution(miri_arr)
miri_spectra = pd.read_csv('data/spectra/MIRI_SPECTRA_for_MCMC.txt', sep = ' ', index_col = 0)

def read_nirspec_convolved():

    hdu = fits.open('NIRSPEC_Convolved.fits')
    
    return hdu

def get_agn_fraction(hdu, extension):
    agn_fraction = hdu[extension].data
    return agn_fraction

def read_miri_convolved():
    hdu = fits.open('MIRI_Convolved.fits')
    return hdu

nirspec_conv_tab = read_nirspec_convolved()[1].data
miri_conv_tab = read_miri_convolved()[1].data

def read_bgl_object(file):

    bgl_output = Beagle_Output(file)
    return bgl_output

def read_dispersion_curve():

    tab = Table.read('data/jwst_nirspec_prism_disp.fits')
    rwave, R = tab['WAVELENGTH'], tab['R']

    return rwave, R
    
def read_nirspec_data():

    
    tb = Table.read('data/spectra/1395_Only_Spec.fits') 
    #tb = Table.read('data/spectra/1395_NIRSpec_Masked.fits')

    wave_AA, flux, fluxerr = tb['wav']*1e10, tb['flux'], tb['fluxerr']

    return wave_AA, flux, fluxerr

def mbh_mstar_relation():

    m_star = np.linspace(7.5, 11, 1000)
    lines = np.zeros((1000, 1000))
    for i in range(1000):
        coeffs = np.random.normal(loc = [1.12, -2.54], scale = [0.08, 0.75])
    
        equation = coeffs[0] * m_star + (coeffs[1])
        lines[i] = equation
    return m_star, lines

def BH_Mass_estimates(bgl_output, eddington_ratio = 1):

    logLacc = np.array(bgl_output['agn_emission']['lacc_AGN']).astype(float)
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
    return agn_df

def BHMass_Stellar_Mass_Relation(stellar_mass, bhmasses_eddington, bhmasses_pt5_eddington, bhmasses_pt1_eddington, agn_df, lines_highz, lr_l16, lr_med, lr_u84, mstar_lowz, mstar_highz):

    plt.figure(figsize = (8, 5), dpi = 150)

    ##########
    #Plotting out Data
    ########
    plt.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_eddington.value)), 
                marker = '*', color = 'red', s = 100, zorder = 99, 
                edgecolors='white', label = r'GHZ2 $\eta$: 1')
    plt.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_eddington.value)), 
                marker = '*', color = 'red', s = 200, zorder = 1)
    
    plt.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_pt5_eddington.value)), 
                marker = '*', color = 'blue', s = 100, zorder = 99, 
                edgecolors='white', label = r'GHZ2 $\eta$: 0.5')
    plt.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_pt5_eddington.value)), 
                marker = '*', color = 'blue', s = 200, zorder = 1)
    
    plt.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_pt1_eddington.value)), 
                marker = '*', color = 'black', s = 100, zorder = 99, 
                edgecolors='white', label = r'GHZ2 $\eta$: 0.1')
    plt.scatter(np.log10(np.median(stellar_mass)), np.log10(np.median(bhmasses_pt1_eddington.value)), 
                marker = '*', color = 'black', s = 200, zorder = 1)
    
    ###########
    #Plotting data from the literature
    ############
    plt.scatter(np.log10(new_agn_df['m_star']), np.log10(new_agn_df['m_bh']), color = '#170b3b')
    
    #############
    #Plotting up the High z relation
    plt.fill_between(mstar_highz, np.percentile(lines_highz, axis = 0,  q = 84),  np.percentile(lines_highz, axis = 0,  q = 16), 
                     color = '#603f8b', alpha = 0.5, label = 'High-z BH Relation')
    plt.plot(mstar_highz, np.median(lines_highz, axis = 0), color = 'black')
    
    ###########
    #Plotting up the local relation
    plt.fill_between(np.log10(mstar_lowz), lr_u84,  lr_l16, color = 'salmon', alpha = 0.5, label = 'Local BH Relation')
    
    plt.plot(np.log10(mstar_lowz), lr_med, color = 'black')
    
    plt.xlabel(r'log$_{10}$(M$_{*}$ [M$_{\odot}$])', fontsize = 15)
    plt.ylabel(r'log$_{10}$(M$_{BH}$[M$_{\odot}$])', fontsize = 15)
    
    plt.legend(fontsize = 10, ncols = 2)
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



def grab_spectral_region(data_wave, data_flux, line_wave, window):

    min_window = line_wave - window
    max_window = line_wave + window
    idx = np.where((min_window < data_wave) & (data_wave < max_window))[0]

    wave = data_wave[idx]
    flux = data_flux[idx]
    
    return wave, flux

def get_percentiles(conv_spec):

    l16_conv, med_conv, u84_conv = np.percentile(conv_spec, axis = 0, q = (16, 50, 84))

    return l16_conv, med_conv, u84_conv

def test_convolution(data_wave, model_wave, model_flux, rwave, R):

    return bgl_agn.convolve_model_spec(rwave, R, model_wave, model_flux, data_wave, oversample = 75)

# line3 =  2798 #MgII
    # window3 = 40
    # ax4 = fig.add_subplot(gs[1, 2])
    
    # wave, lflux = grab_spectral_region(rest_wave, l16_fagn, line3, window3)
    # wave, flux = grab_spectral_region(rest_wave, med_fagn, line3, window3)
    # wave, uflux = grab_spectral_region(rest_wave, u84_fagn, line3, window3)
    
    # #wave_data, flux_data = grab_spectral_region(data_wave, data_flux, line3, window3)
   
    # ax4.step(wave, flux, where = 'mid', color = 'purple', alpha = 0.5)
    # ax4.fill_between(wave, uflux, lflux, step = 'mid', color = 'dodgerblue')
    # ax4.set_ylim(0, 0.5)

    # with open('Spectral_Fitting/MgII_Spectra.pickle', 'rb') as handle:
    #     data = pickle.load(handle)
    
    # emcee_df = pd.read_csv('Spectral_Fitting/MgII_Fit_Results.txt', index_col = 0)
    # SNR = np.median(emcee_df.Combined_Fluxes.values/np.std(emcee_df.Combined_Fluxes.values, ddof = 1))

    # inputwave = data['input_wave']
    # inputflux = data['input_spec']
    # inputfluxerr = data['input_spec_err']

    # wave, flux = grab_spectral_region(rest_wave, med, line3, window3)
    
    # ax4_twin = fig.add_subplot(gs[2, 2])
    # ax4_twin.step(inputwave, inputflux, where = 'mid', color = 'red', label = f'SNR: {SNR:.2f}')
    # ax4_twin.errorbar(inputwave, inputflux, yerr = inputfluxerr, color = 'red', fmt = 'none')
   
    # ax4_twin.step(wave, flux, color = 'purple', where = 'mid', label = 'Model')
    # ax4_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    # ax4_twin.legend()

    
    # ax4.set_ylabel('')
    # ax4.set_xlabel('')
    # ax4.set_title('MgII', fontsize = 15)
    # ax4.minorticks_on()
    # #ax4.set_title('cosh(x)')

    
    
    # line4 = 3868.760
    # window4 = 20
    # ax5 = fig.add_subplot(gs[1, 3])

    
    
    # wave, lflux = grab_spectral_region(rest_wave, l16_fagn, line4, window4)
    # wave, flux = grab_spectral_region(rest_wave, med_fagn, line4, window4)
    # wave, uflux = grab_spectral_region(rest_wave, u84_fagn, line4, window4)
    
    # ax5.step(wave, flux, where = 'mid', color = 'purple', alpha = 0.5)
    # ax5.fill_between(wave, uflux, lflux, step = 'mid', color = 'dodgerblue')
    # ax5.set_ylim(0, 0.5)

    # wave, flux = grab_spectral_region(rest_wave, data_flux, line4, window4)
    # wave, flux_err = grab_spectral_region(rest_wave, data_flux_err, line4, window4)

    
    # with open('Spectral_Fitting/NeIII_Spectra.pickle', 'rb') as handle:
    #     data = pickle.load(handle)
    
    # emcee_df = pd.read_csv('Spectral_Fitting/NeIII_Fit_Results.txt', index_col = 0)
    # SNR = np.median(emcee_df.Fluxes.values/np.std(emcee_df.Fluxes.values, ddof = 1))

    # inputwave = data['input_wave']
    # inputflux = data['input_spec']
    # inputfluxerr = data['input_spec_err']

    # wave, flux = grab_spectral_region(rest_wave, med, line4, window4)
    
    
    # ax5_twin = fig.add_subplot(gs[2, 3])
    # ax5_twin.step(inputwave, inputflux/13.34, where = 'mid', color = 'red', label = f'SNR: {SNR:.2f}')
    # ax5_twin.errorbar(inputwave, inputflux/13.34, yerr = inputfluxerr/13.34, color = 'red', fmt = 'none')
    # ax5_twin.step(wave, flux, color = 'purple', where = 'mid')
    # ax5_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    # ax5_twin.legend()
    
    
    # ax5.set_ylabel('')
    # ax5.set_xlabel('')
    # ax5.set_title(r'[NeIII]3868', fontsize = 15)
    # ax5.minorticks_on()

def plot_spectra_and_zoom_ins(bgl_agn, data_wave, data_flux, data_flux_err, agn_fraction_nirspec, agn_fraction_miri, bgl_sf):

    MAX_IDX = np.argmax(bgl_agn['posterior']['probability'])

    zp1 = 1+bgl_agn['gal_prop']['redshift'][MAX_IDX]
    
    #reading in the model from beagle
    convolved_data = bgl_agn['marg_sed']
    convolved_wave = bgl_agn['marg_wvln']

    #converting to restframe
    rest_wave = bgl_agn['marg_wvln']/zp1

    l16, med, u84 = np.percentile(convolved_data, q = (16, 50, 84), axis = 0)
    
    # Create a figure
    fig = plt.figure(figsize=(25, 12), dpi = 100)
    
    # Create a grid with 3 rows and 4 columns using gridspec
    gs = fig.add_gridspec(3, 4)
    
    # Top row: single long plot spanning all 4 columns
    ax1 = fig.add_subplot(gs[0, :])  # Span all columns in the first row
    ax1.step(convolved_wave, med, color = 'purple', label = 'SF+AGN Model Fit', zorder = 10, where = 'mid')
    ax1.fill_between(convolved_wave, u84, l16, color = 'dodgerblue', alpha = 0.5, zorder = 10, step = 'mid')
    
    ax1.step(data_wave, data_flux, color = 'gray', label = "Data", where = 'mid')
    ax1.errorbar(data_wave, data_flux, yerr = data_flux_err, color = 'red', fmt = 'none')
    ax1.set_ylim(-0.5e-20, 1.2e-20)
    ax1.legend(fontsize = 15)

    ax11 = ax1.twiny()

    obs_xlim = ax1.get_xlim()
    rest_xlim = [x / zp1 for x in obs_xlim]  # Convert to rest-frame
    ax11.set_xlim(rest_xlim)

    obs_ticks = ax1.get_xticks()
    rest_ticks = [tick / zp1 for tick in obs_ticks]
    ax11.set_xticks(rest_ticks)
    ax11.set_xticklabels([f"{tick:.0f}" for tick in rest_ticks])
    ax11.set_xlabel(r"Rest Frame Wavelengths [$\AA$]", fontsize = 20)
    
    #agn_fraction
    l16_fagn, med_fagn, u84_fagn = agn_fraction_nirspec[0], agn_fraction_nirspec[1], agn_fraction_nirspec[2]
    
    # Second row: 4 plots
    line1 = 1549 #CIV
    window1 = 40
    ax2 = fig.add_subplot(gs[1, 0])

    #Grabbing AGN fraction
    wave, lflux = grab_spectral_region(rest_wave, l16_fagn, line1, window1)
    wave, flux = grab_spectral_region(rest_wave, med_fagn, line1, window1)
    wave, uflux = grab_spectral_region(rest_wave, u84_fagn, line1, window1)
    
    ax2.step(wave, flux, where = 'mid', color = 'purple', alpha = 0.5)
    ax2.fill_between(wave, uflux, lflux, step = 'mid', color = 'dodgerblue')
    ax2.set_ylim(0, 0.5)

    #Grabbing Model Fit
    with open('Spectral_Fitting/CIV_Spectra.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    emcee_df = pd.read_csv('Spectral_Fitting/CIV_Fit_Results.txt', index_col = 0)
    SNR = np.median(emcee_df.Fluxes.values/np.std(emcee_df.Fluxes.values, ddof = 1))
    
    inputwave = data['input_wave']
    inputflux = data['input_spec']
    inputfluxerr = data['input_spec_err']

    full_sed_l16, full_sed_med, full_sed_u84 = np.percentile(bgl_agn['full_sed'],  q= (16, 50, 84), axis = 0)
    full_sed_wvln = bgl_agn['full_sed_wvln']

    #wave, flux = grab_spectral_region(rest_wave, med, line1, window1)
    flux = bgl_agn.convolve_model_spec(rwave*1e4/zp1, R, full_sed_wvln, full_sed_med/zp1, inputwave, oversample = 75)

    
    ax2_twin = fig.add_subplot(gs[2, 0])
    ax2_twin.step(inputwave, inputflux/zp1, where = 'mid', color = 'red', label = f'SNR: {SNR:.2f}')
    ax2_twin.errorbar(inputwave, inputflux/zp1, yerr = inputfluxerr/13.34, color = 'red', fmt = 'none')
    ax2_twin.step(inputwave, flux, where = 'mid', color = 'purple', label = 'Model')
    ax2_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    ax2_twin.legend(fontsize = 12)

    ax2.set_xlabel('')
    ax2.set_title(r'CIV', fontsize = 15)
    ax2.minorticks_on()
    
    
    line2 = 1909 #CIII
    window2 = 40
    ax3 = fig.add_subplot(gs[1, 1])

    wave, lflux = grab_spectral_region(rest_wave, l16_fagn, line2, window2)
    wave, flux = grab_spectral_region(rest_wave, med_fagn, line2, window2)
    wave, uflux = grab_spectral_region(rest_wave, u84_fagn, line2, window2)
    
    
    #this plots the AGN fraction
    ax3.step(wave, flux, where = 'mid', color = 'purple', alpha = 0.5, label = 'Model')
    ax3.fill_between(wave, uflux, lflux, step = 'mid', color = 'dodgerblue')

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
    flux = bgl_agn.convolve_model_spec(rwave*1e4/zp1, R, full_sed_wvln, full_sed_med/zp1, inputwave, oversample = 75)
    
    ax3_twin = fig.add_subplot(gs[2, 1])
    ax3_twin.step(inputwave, inputflux/zp1, where = 'mid', color = 'red', label = f'SNR: {SNR:.2f}')
    ax3_twin.errorbar(inputwave, inputflux/zp1, yerr = inputfluxerr/13.34, color = 'red', fmt = 'none')
    ax3_twin.step(inputwave, flux, color = 'purple', where = 'mid', label = 'Model')
    ax3_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    ax3_twin.legend()

    
    ax3.set_ylabel('')
    ax3.set_xlabel('')
    ax3.set_title('CIII]', fontsize = 15)
    ax3.set_ylim(0, 0.5)
    ax3.minorticks_on()
    
    #########
    #Miri Spectral Range
    #########
    full_l16, full_med, full_u84 = np.percentile(bgl_agn['full_sed'],  q= (16, 50, 84), axis = 0)
    full_wave = bgl_agn['full_sed_wvln']

    
    line5 =  5007#OIII
    window5 = 100
    l16_fagn, med_fagn, u84_fagn =  agn_fraction_miri[0], agn_fraction_miri[1], agn_fraction_miri[2] #this should be miri
    rest_wave = miri_spectra.wave_AA_rest.values
    miri_flux = miri_spectra.flux_flam.values 
    miri_flux_err = miri_spectra.fluxerr_flam.values
    
    # Third row: 4 plots
    ax6 = fig.add_subplot(gs[1, 2])

    wave, lflux = grab_spectral_region(rest_wave, l16_fagn, line5, window5)
    wave, flux = grab_spectral_region(rest_wave, med_fagn, line5, window5)
    wave, uflux = grab_spectral_region(rest_wave, u84_fagn, line5, window5)
        
    ax6.step(wave, flux, where = 'mid', color = 'purple', alpha = 0.5)
    ax6.fill_between(wave, uflux, lflux, step = 'mid', color = 'dodgerblue')
    ax6.set_ylim(0, 0.5)

    with open('Spectral_Fitting/OIII5007_OIII4960_Spectra.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    emcee_df = pd.read_csv('Spectral_Fitting/OIII5007_OIII4960_Fit_Results.txt', index_col = 0)
    SNR = np.median(emcee_df.Combined_Fluxes.values/np.std(emcee_df.Combined_Fluxes.values, ddof = 1))

    inputwave = data['input_wave']
    inputflux = data['input_spec']
    inputfluxerr = data['input_spec_err']
    
    _, conv_spec, _ = get_percentiles(miri_conv_tab)
    wave, flux = grab_spectral_region(rest_wave, conv_spec, line5, window5)
    
    input_wave, input_flux = grab_spectral_region(rest_wave, miri_flux, line5, window5)
    _, input_fluxerr = grab_spectral_region(rest_wave, miri_flux_err, line5, window5)
    

    ax6_twin = fig.add_subplot(gs[2, 2])
    ax6_twin.step(input_wave, input_flux, where = 'mid', color = 'red', label = f'SNR: {SNR:.2f}')
    ax6_twin.errorbar(input_wave, input_flux, yerr = input_fluxerr, color = 'red', fmt = 'none')
    
    ax6_twin.step(wave, flux/zp1, color = 'black', where = 'mid')
    ax6_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    ax6_twin.legend()

        
    ax6.set_title(r'OIII5007', fontsize = 15)
    ax6.minorticks_on()
    #ax6.set_title('log(x + 1)')
    
    
    line6 =  6564#Halpha
    window6 = 100
    ax7 = fig.add_subplot(gs[1, 3])
    
    wave, lflux = grab_spectral_region(rest_wave, l16_fagn, line6, window6)
    wave, flux = grab_spectral_region(rest_wave, med_fagn, line6, window6)
    wave, uflux = grab_spectral_region(rest_wave, u84_fagn, line6, window6)
        
    ax7.step(wave, flux, where = 'mid', color = 'purple', alpha = 0.5)
    ax7.fill_between(wave, uflux, lflux, step = 'mid', color = 'dodgerblue')
    ax7.set_ylim(0, 0.5)

    with open('Spectral_Fitting/Halpha_Spectra.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    emcee_df = pd.read_csv('Spectral_Fitting/Halpha_Fit_Results.txt', index_col = 0)
    SNR = np.median(emcee_df.Fluxes.values/np.std(emcee_df.Fluxes.values, ddof = 1))

    inputwave = data['input_wave']
    inputflux = data['input_spec']
    inputfluxerr = data['input_spec_err']

    wave, flux = grab_spectral_region(rest_wave, conv_spec, line6, window6)
    input_wave, input_flux = grab_spectral_region(rest_wave, miri_flux, line6, window6)
    _, input_fluxerr = grab_spectral_region(rest_wave, miri_flux_err, line6, window6)

    ax7_twin = fig.add_subplot(gs[2, 3])
    ax7_twin.step(input_wave, input_flux, where = 'mid', color = 'red', label = f'SNR: {SNR:.2f}')
    ax7_twin.errorbar(input_wave, input_flux, yerr = input_fluxerr, color = 'red', fmt = 'none')
    
    ax7_twin.step(wave, flux/zp1, color = 'black', where = 'mid')
    ax7_twin.set_xlabel(r"Rest-Frame Wavelengths [$\AA$]")
    ax7_twin.legend()

    
    
    ax7.set_ylabel('')
    ax7.set_title(r'H$\alpha$', fontsize = 15)
    ax7.minorticks_on()

    ax2.set_ylabel('AGN Fraction', fontsize = 15)
    ax1.set_ylabel(r'F$_{\lambda}$ [erg s$^{-1}$cm$^2$$\AA^{-1}$]', fontsize = 20)
    ax1.set_xlim(15000, 55000)
    plt.tight_layout()
    plt.show()
    
    return fig, [ax1, ax2, ax2_twin, ax3, ax3_twin, ax6, ax6_twin, ax7, ax7_twin]


def plot_bhmass_vs_redshift(bgl_agn):

    bh_mass_edd1 = np.log10(np.median(BH_Mass_estimates(bgl_agn, eddington_ratio = 1).value))
    bh_mass_edd05 = np.log10(np.median(BH_Mass_estimates(bgl_agn, eddington_ratio = 0.5).value))
    bh_mass_edd01 = np.log10(np.median(BH_Mass_estimates(bgl_agn, eddington_ratio = 0.1).value))

    agn_df = read_agn_df()

    #getting BH Errors Log
    MBH = agn_df.m_bh.values
    MBH_err_plus = agn_df.m_bh_err_plus
    MBH_err_minus = agn_df.m_bh_err_minus

    low_errors, upper_errors = get_errors_log10(MBH, MBH_err_minus, MBH_err_plus)
    MBH_log10 = np.log10(MBH)
    
    fig, ax = plt.subplots(figsize = (10, 5))

    ax.errorbar(agn_df.z_spec.values, MBH_log10, 
                yerr = [low_errors, upper_errors], color = 'black', fmt = 'none')
    ax.scatter(agn_df.z_spec.values, MBH_log10, 
                marker = 'o', color = 'black', label = 'Literature')
    ax.scatter(12.34, bh_mass_edd1, color = 'red', marker = '*', s = 250, label = r'$\eta$ = 1')
    ax.scatter(12.34, bh_mass_edd05, color = 'blue', marker = '*', s = 250, label = r'$\eta$ = 0.5')
    ax.scatter(12.34, bh_mass_edd01, color = 'black', marker = '*', s = 250, label = r'$\eta$ = 0.1')
    
    ax.scatter(12.34, bh_mass_edd1, color = 'red', marker = '*', s = 200, edgecolors='white')
    ax.scatter(12.34, bh_mass_edd05, color = 'blue', marker = '*', s = 200, edgecolors='white')
    ax.scatter(12.34, bh_mass_edd01, color = 'black', marker = '*', s = 200, edgecolors='white')
    ax.set_ylabel(r'log$_{10}$(M$_{BH}$/M$_{\odot}$)', fontsize = 15)
    ax.set_xlabel('Redshift', fontsize = 15)
    ax.legend(fontsize = 10, ncols = 2)
    plt.show()

def test_convolution(nirspec_wave, miri_wave, nirspec_conv, miri_conv, restframe = True):

    fig, ax = plt.subplots(1, 2, figsize = (12, 6), constrained_layout = True)
    
    ax1 = ax[0]
    ax2 = ax[1]

    l16_conv_nirspec, med_conv_nirspec, u84_conv_nirspec = np.percentile(nirspec_conv, axis = 0, q = (16, 50, 84))
    l16_conv_miri, med_conv_miri, u84_conv_miri = np.percentile(miri_conv, axis = 0, q = (16, 50, 84))

    if restframe:
        nirspec_wave = nirspec_wave/13.34
        miri_wave    = miri_wave/13.34
    
    ax1.step(nirspec_wave, med_conv_nirspec, color = 'black', where = 'mid')
    ax1.fill_between(nirspec_wave, u84_conv_nirspec, l16_conv_nirspec, step = 'mid', color = 'purple', alpha = 0.5)

    ax2.step(miri_wave, med_conv_miri, color = 'black', where = 'mid')
    ax2.fill_between(miri_wave, u84_conv_miri, l16_conv_miri, step = 'mid', color = 'purple', alpha = 0.5)

    #plt.show()
    return fig, [ax1, ax2]
 

def check_fits(bgl_agn, wave_AA, flux, fluxerr):

    seds = bgl_agn['full_sed']
    agn_seds = bgl_agn['agn_sed']
    waves = bgl_agn['full_sed_wvln']

    l16, med, u84 = np.percentile(seds, q = (16, 50, 84), axis = 0)
    l16_agn, med_agn, u84_agn = np.percentile(agn_seds, q = (16, 50, 84), axis = 0)

    fig, ax = plt.subplots(1, 1, figsize = (12, 6), constrained_layout = True)
    ax.step(wave_AA, flux, color = 'gray', label = "Data", where = 'mid')
    ax.errorbar(wave_AA, flux, yerr = fluxerr, color = 'red', fmt = 'none')
    
    
    #ax.step(waves, med, color = 'black', where = 'mid', label = 'Full SED')
    #ax.step(waves, med_agn, color = 'red', where = 'mid', label = 'AGN SED')
    #ax.fill_between(waves, u84, l16, step = 'mid', color = 'purple', alpha = 0.5)
    ax.set_xlabel(r'Wavelengths [$\AA$]', fontsize = 20)
    ax.set_ylabel(r'F$_{\lambda}$ [erg s$^{-1}$cm$^2$$\AA^{-1}$]', fontsize = 20)
    #ax.set_title('Convolved SED', fontsize = 20)
    #ax.set_ylim(0, 1.2e-20)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_xlim(1e4, 3e5)
    ax.legend(fontsize = 15)
    plt.show()

def plot_bh_mass_relation(bgl_agn):        
    
    stellar_mass = bgl_agn['gal_prop']['M_star']

    bh_mass_edd1 = BH_Mass_estimates(bgl_agn, eddington_ratio = 1)
    bh_mass_edd05 = BH_Mass_estimates(bgl_agn, eddington_ratio = 0.5)
    bh_mass_edd01 = BH_Mass_estimates(bgl_agn, eddington_ratio = 0.1)
    agn_df = read_agn_df() 
    mstar_highz, lines_highz = mbh_mstar_relation()

    lr_l16, lr_med, lr_u84, mstar, _ = local_relation_bhmass()

    BHMass_Stellar_Mass_Relation(stellar_mass, bh_mass_edd1, bh_mass_edd05, bh_mass_edd01, agn_df, lines_highz, lr_l16, lr_med, lr_u84, mstar, mstar_highz)


def generate_component_specs_plot(bgl_output, data_wave, data_flux, data_flux_err):
    
    MAX_IDX = np.argmax(bgl_output['posterior']['probability'])

    zp1 = 1+bgl_output['gal_prop']['redshift'][MAX_IDX]
    #print(bgl_output['z'])
    mult_factor = 1e20

    og_full_sed = bgl_output['full_sed']
    og_agn_sed = bgl_output['agn_sed']

    og_agn_l16, og_agn_med, og_agn_u84 = np.percentile(og_agn_sed* mult_factor, q = (16, 50, 84), axis = 0)
    og_spec_l16, og_spec_med, og_spec_u84 = np.percentile(og_full_sed* mult_factor, q = (16, 50, 84), axis = 0)
    og_wave = bgl_output['full_sed_wvln'] * (1+z)

    
    
    conv_spec = (bgl_output['convolved_spectra']/zp1) * mult_factor
    conv_agn = (bgl_output['convolved_agn']/zp1) * mult_factor
    conv_wave = bgl_output['convolved_wave']

    og_mask = (og_wave >= conv_wave[0]) & (og_wave <= conv_wave[-1])
    
    agn_l16, agn_med, agn_u84 = np.percentile(conv_agn, q = (16, 50, 84), axis = 0)
    spec_l16, spec_med, spec_u84 = np.percentile(conv_spec, q = (16, 50, 84), axis = 0)
    
    diff_spec = conv_spec - conv_agn

    diff_spec_l16, diff_spec_med, diff_spec_u84 = np.percentile(diff_spec, q = (16, 50, 84), axis = 0)
    
    # Create a figure
    fig = plt.figure(figsize=(20, 10), dpi = 200)
    
    # Create a grid with 3 rows and 4 columns using gridspec
    gs = fig.add_gridspec(1, 1)

    ax = fig.add_subplot(gs[0])
    
    #ax.plot(og_wave[og_mask], bgl_output['full_sed'][MAX_IDX][og_mask]*mult_factor/zp1, label = 'Full Res SED')
    #ax.plot(og_wave[og_mask], bgl_output['agn_sed'][MAX_IDX][og_mask]*mult_factor/zp1, label = 'Full Res AGN SED' )

    ax.step(conv_wave, agn_med, label = 'Model AGN SED', color = 'purple', where = 'mid', zorder = 200)
    ax.fill_between(conv_wave, agn_u84, agn_l16, step = 'mid', color = 'purple')
    ax.step(conv_wave, spec_med, label = 'Model Full SED', color = 'dodgerblue', where = 'mid', zorder = 202)
    ax.fill_between(conv_wave, spec_u84, spec_l16, step = 'mid', color = 'dodgerblue')
    ax.step(conv_wave, diff_spec_med, label = 'Stellar+HII Region SED', color = 'red', where = 'mid', zorder = 200)
    ax.fill_between(conv_wave, diff_spec_u84, diff_spec_l16, step = 'mid', color = 'red')

    ax.step(data_wave, data_flux* mult_factor, color = 'black', where = 'mid')
    ax.errorbar(data_wave, data_flux* mult_factor, yerr = data_flux_err* mult_factor, color = 'gray', fmt = 'none')

    ax.set_ylabel(r'F$_{\nu}$ [1$\times$10$^{-20}$ erg s$^{-1}$cm$^2$$\AA$$^{-1}$]', fontsize = 20)
    ax.set_xlabel(r'Observed Wavelength [$\AA$]', fontsize = 20)
    obs_ticks = ax.get_xticks()
    ax.set_xticks(obs_ticks)
    ax.set_xticklabels(obs_ticks)
    ax.legend(fontsize = 20)

    ax2 = ax.twiny()

    obs_xlim = ax.get_xlim()
    rest_xlim = [x / zp1 for x in obs_xlim]  # Convert to rest-frame
    ax2.set_xlim(rest_xlim)

    obs_ticks = ax.get_xticks()
    rest_ticks = [tick / zp1 for tick in obs_ticks]
    ax2.set_xticks(rest_ticks)
    ax2.set_xticklabels([f"{tick:.0f}" for tick in rest_ticks])
    ax2.set_xlabel(r"Rest Frame Wavelengths [$\AA$]", fontsize = 20)

    offset = 40*zp1
    ax.annotate(r'CIV', (1548.187*zp1 - offset, 1.1), fontsize = 15)

    ax.annotate(r'CIII]', (1908.734	*zp1 - offset, .9))

    ax.annotate(r'MgII', (2795.528*zp1 - offset, .3))

    ax.annotate(r'HeII', (1640.420*zp1 - offset, .7))

    ax.annotate(r'[NeIII]', ((3868.760*zp1) - 1500, .75))
    #1640.420
    print(ax.get_xticks())
    ax.set_ylim(-0.5, 1.2)
    return fig, [ax, ax2]
    

if __name__ == "__main__":
    rwave, R = read_dispersion_curve()

    FILE_AGN = 'tacc_fits/SF_AGN_Mup100_Default.fits.gz'
    FILE_SF = 'results/CO_Grids_SF_MODEL/1395_BEAGLE.fits.gz'

    nirspec_conv_hdu = read_nirspec_convolved()
    miri_conv_hdu = read_miri_convolved()


    nirspec_agn_fraction = get_agn_fraction(nirspec_conv_hdu, 'CONVOLVED_AGN_FRACTION')
    miri_agn_fraction = get_agn_fraction(nirspec_conv_hdu, 'CONVOLVED_AGN_FRACTION')

    print('Reading Data')
    bgl_agn = read_bgl_object(FILE_AGN)
    bgl_sf = read_bgl_object(FILE_SF)
    wave_AA, flux, fluxerr = read_nirspec_data()

    marg_wave = bgl_agn['marg_wvln']
    model_wave = bgl_agn['full_sed_wvln']
    agn_seds = bgl_agn['agn_sed']
    seds = bgl_agn['full_sed']

    l16, med, u84 = np.percentile(agn_seds, q = (16, 50, 84), axis = 0)
    l16_full, med_full, u84_full = np.percentile(seds, q = (16, 50, 84), axis = 0)

    testing_conv_l16 = bgl_agn.convolve_model_spec(rwave*1e4, R, model_wave*13.34, l16, marg_wave, 75)
    testing_conv_med = bgl_agn.convolve_model_spec(rwave*1e4, R, model_wave*13.34, med, marg_wave, 75)
    testing_conv_u84 = bgl_agn.convolve_model_spec(rwave*1e4, R, model_wave*13.34, u84, marg_wave, 75)

    testing_conv_full_l16 = bgl_agn.convolve_model_spec(rwave*1e4, R, model_wave*13.34, l16_full, marg_wave, 75)
    testing_conv_full_med = bgl_agn.convolve_model_spec(rwave*1e4, R, model_wave*13.34, med_full, marg_wave, 75)
    testing_conv_full_u84 = bgl_agn.convolve_model_spec(rwave*1e4, R, model_wave*13.34, u84_full, marg_wave, 75)

    testing_conv_l16_miri = bgl_agn.convolve_model_spec(miri_arr*1e4, miri_R, model_wave*13.34, l16, miri_spectra.wave_AA.values, 75)
    testing_conv_med_miri = bgl_agn.convolve_model_spec(miri_arr*1e4, miri_R, model_wave*13.34, med, miri_spectra.wave_AA.values, 75)
    testing_conv_u84_miri = bgl_agn.convolve_model_spec(miri_arr*1e4, miri_R, model_wave*13.34, u84, miri_spectra.wave_AA.values, 75)

    testing_conv_full_l16_miri = bgl_agn.convolve_model_spec(miri_arr*1e4, miri_R, model_wave*13.34, l16_full, miri_spectra.wave_AA.values, 75)
    testing_conv_full_med_miri = bgl_agn.convolve_model_spec(miri_arr*1e4, miri_R, model_wave*13.34, med_full, miri_spectra.wave_AA.values, 75)
    testing_conv_full_u84_miri = bgl_agn.convolve_model_spec(miri_arr*1e4, miri_R, model_wave*13.34, u84_full, miri_spectra.wave_AA.values, 75)

    nirspec_agn_fraction = [testing_conv_l16/testing_conv_full_l16, testing_conv_med/testing_conv_full_med, testing_conv_u84/testing_conv_full_u84]
    miri_agn_fraction = [testing_conv_l16_miri/testing_conv_full_l16_miri, testing_conv_med_miri/testing_conv_full_med_miri, testing_conv_u84_miri/testing_conv_full_u84_miri]
    #nirspec_conv_tab

    plot_spectra_and_zoom_ins(bgl_agn, wave_AA, flux, fluxerr, nirspec_agn_fraction, miri_agn_fraction, bgl_sf)
    #test_convolution(wave_AA, miri_spectra.wave_AA.values, nirspec_conv_tab, miri_conv_tab)

    bh_mass_edd1 = BH_Mass_estimates(bgl_agn, eddington_ratio = 1)
    bh_mass_edd05 = BH_Mass_estimates(bgl_agn, eddington_ratio = 0.5)
    bh_mass_edd01 = BH_Mass_estimates(bgl_agn, eddington_ratio = 0.1)
    