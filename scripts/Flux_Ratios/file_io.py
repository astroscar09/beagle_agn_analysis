from astropy.io import fits
import pandas as pd
from astropy.table import Table

def read_hdu(file):
    hdu = fits.open(file)
    return hdu

def read_text_file(file):
    return pd.read_csv(file, sep = ' ')

def read_line_file(file):

    maps = { 'NIV_line_lum': 'NIV_flux',
            'NIV_line_lum_err': 'NIV_fluxerr',
            'CIV_line_lum': 'CIV_flux',
            'CIV_line_lum_err': 'CIV_fluxerr',
            'HeII_line_lum': 'HeII_flux',
            'HeII_line_lum_err': 'HeII_fluxerr',
            'OIII1663_line_lum': 'OIII1663_flux',
            'OIII1663_line_lum_err': 'OIII1663_fluxerr',
            'CIII_line_lum': 'CIIIflux',
            'CIII_line_lum_err': 'CIII_fluxerr',
            'MgII_line_lum': 'MgII_flux',
            'MgII_line_lum_err': 'MgII_fluxerr',
            'OII_line_lum': 'OII_flux',
            'OII_line_lum_err': 'OII_fluxerr',
            'NeIII_line_lum': 'NeIII_flux',
            'NeIII_line_lum_err': 'NeIII_fluxerr',
            'Hbeta_line_lum': 'Hbeta_flux',
            'Hbeta_line_lum_err': 'Hbeta_fluxerr',
            'OIII4960_line_lum': 'OIII4960_flux',
            'OIII4960_line_lum_err': 'OIII4960_fluxerr',
            'OIII5007_line_lum': 'OIII5007_flux',
            'OIII5007_line_lum_err': 'OIII5007_fluxerr',
            'Halpha_line_lum': 'Halpha_flux',
            'Halpha_line_lum_err': 'Halpha_fluxerr',
            'OIII52_line_lum': 'OIII52_flux',
            'OIII52_line_lum_err': 'OIII52_fluxerr',
            'OIII88_line_lum': 'OIII88_flux',
            'OIII88_line_lum_err': 'OIII88_fluxerr',
            }
    
    input_line_df = Table.read(file).to_pandas()
    input_line_df = input_line_df.rename(columns = maps)

    return input_line_df