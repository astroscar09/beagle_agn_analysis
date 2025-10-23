from Post_BEAGLE_Analysis import *
from Helper_Functions import *
import sys
from astropy.io import fits

args = list(map(str.lower,sys.argv))

def miri_resolution(microns):
    return 4603 - 128*microns + 10**(-7.4*microns)

def read_dispersion_curve():

    tab = Table.read('data/jwst_nirspec_prism_disp.fits')
    rwave, R = tab['WAVELENGTH'], tab['R']

    return rwave, R

def read_nirspec_data():

    
    tb = Table.read('1395_Final_Masked_Nirspec.fits')

    wave_AA, flux, fluxerr = tb['wav']*1e10, tb['flux'], tb['fluxerr']

    return wave_AA, flux, fluxerr

def read_bgl_object(file):
    try:
        bgl_output = Beagle_Output(file)
        return bgl_output
    except:
        return None

def convolve_nirspec(bgl_output, rwave, R, obs_wave, oversample):
    bgl_output.generate_convolved_spectra(rwave, R, obs_wave, oversample)
    return bgl_output['convolved_spectra'], bgl_output['convolved_agn'], bgl_output['agn_ratio_convolved']

def convolve_miri(bgl_output, rwave, R, obs_wave, oversample):
    bgl_output.generate_convolved_spectra(rwave, R, obs_wave, oversample)
    return bgl_output['convolved_spectra'], bgl_output['convolved_agn'], bgl_output['agn_ratio_convolved']

miri_arr = np.linspace(5, 30, 1000)
miri_R = miri_resolution(miri_arr)
miri_spectra = pd.read_csv('data/spectra/MIRI_SPECTRA_for_MCMC.txt', sep = ' ', index_col = 0)


if __name__ == '__main__':
    
    rwave, R = read_dispersion_curve()
    nirspec_wave, nirspec_flux, nirspec_fluxerr = read_nirspec_data()

    print(len(nirspec_wave))

    if '--file' in args:
        
        indx = args.index("--file")
        
        file = sys.argv[indx + 1]
    
    else:
        sys.exit(-1) 

    if '--kind' in args:
        
        indx = args.index("--kind")
        
        kind = sys.argv[indx + 1]
    
    else:
        sys.exit(-1) 
    
    bgl_output = read_bgl_object(file)

    if kind == 'nirspec':
    
        conv_flux, con_agn, conv_fraction = convolve_nirspec(bgl_output, rwave*1e4, R, nirspec_wave, 75)

        hdu1 = fits.ImageHDU(data=conv_flux,     name='Convolved_SED')
        hdu2 = fits.ImageHDU(data=con_agn,       name='Convolved_AGN_SED')
        hdu3 = fits.ImageHDU(data=conv_fraction, name='Convolved_AGN_Fraction')
        
        # Create a primary HDU (can be empty)
        primary_hdu = fits.PrimaryHDU()
        
        # Combine all HDUs into an HDUList
        hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])

        # Write to a FITS file
        hdul.writeto('NIRSPEC_Convolved.fits', overwrite=True)
        
    elif kind == 'miri':
        conv_flux_miri, con_agn_miri, conv_fraction_miri = convolve_nirspec(bgl_output, miri_arr*1e4, miri_R, miri_spectra.wave_AA.values, 75)

        hdu1 = fits.ImageHDU(data=conv_flux_miri,     name='Convolved_SED')
        hdu2 = fits.ImageHDU(data=con_agn_miri,       name='Convolved_AGN_SED')
        hdu3 = fits.ImageHDU(data=conv_fraction_miri, name='Convolved_AGN_Fraction')
        
        # Create a primary HDU (can be empty)
        primary_hdu = fits.PrimaryHDU()
        
        # Combine all HDUs into an HDUList
        hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])

        # Write to a FITS file
        hdul.writeto('MIRI_Convolved.fits', overwrite=True)

        