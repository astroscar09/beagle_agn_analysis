from file_io import read_text_file, read_hdu
from plotting_utils import add_all_fluxes, plot_bgl_fluxes, violin_plot_fluxes_models, plot_corner_plot
from data_processing import *
import matplotlib.pyplot as plt
import pandas as pd

file = 'Oscars_Line_Fluxes.txt'
df = read_text_file(file)

line_col = 'lines'
flux_col =  'fluxes'
fluxerr_col = 'errors'
upperlims = 'upper_lims'

lines = df[line_col]
fluxes = df[flux_col]
fluxerr = df[fluxerr_col]
ulims = df[upperlims]

TEST_FILE = '../ReRuns/AGNMup100_Default_Baseline.fits.gz'
TEST1_FILE = '../ReRuns/AGNMup100_Masked_NO.fits.gz'
TEST2_FILE = '../ReRuns/AGNMup100_Default_Masked_Lines.fits.gz'

mockfile1 = '../ReRuns/fixedNO_03.fits'
mockfile2 = '../ReRuns/fixedNO_03_Check.fits'

# hdu = read_hdu(TEST1_FILE)
# post_df = get_posterior(hdu)
# cols = ['mass', 'specific_sfr',
#        'current_sfr_timescale', 'nebular_logu', 'nebular_z',
#        'agn_lacc', 'agn_logu', 'agn_xi', 'agn_z']



def check_mock_agn_line_df(FILE, redshift):

    hdu = read_hdu(FILE)
    agn_em_df = agn_emission_lines_hdu(hdu)
    hii_em_df = hii_emission_lines_hdu(hdu)
    hii_em_df, agn_em_df = merge_doublets(hii_em_df, agn_em_df)
    #redshift = get_redshift(hdu)
    DL = get_luminosity_distance(redshift)
    hii_em_df = convert_lum_to_fluxes(hii_em_df, DL)
    agn_em_df = convert_lum_to_fluxes(agn_em_df, DL)
    tot_df = get_total_emission(hii_em_df, agn_em_df)

    return tot_df

check1 = check_mock_agn_line_df(mockfile1, 12.34)
check2 = check_mock_agn_line_df(mockfile2, 12.34)

diff = check1 - check2.values

#print(check1)
#print(check2)

#print(diff.head())


def agn_line_df(FILE):

    hdu = read_hdu(FILE)
    agn_em_df = agn_emission_lines_hdu(hdu)
    hii_em_df = hii_emission_lines_hdu(hdu)
    hii_em_df, agn_em_df = merge_doublets(hii_em_df, agn_em_df)
    redshift = get_redshift(hdu)
    DL = get_luminosity_distance(redshift)
    hii_em_df = convert_lum_to_fluxes(hii_em_df, DL)
    agn_em_df = convert_lum_to_fluxes(agn_em_df, DL)
    tot_df = get_total_emission(hii_em_df, agn_em_df)

    return tot_df

def sf_line_df(FILE):

    hdu = read_hdu(FILE)
    hii_em_df = hii_emission_lines_hdu(hdu)
    hii_em_df, agn_em_df = merge_doublets(hii_em_df)
    redshift = get_redshift(hdu)
    DL = get_luminosity_distance(redshift)
    tot_df = convert_lum_to_fluxes(hii_em_df, DL)

    return tot_df



# agn1_tot_flux = agn_line_df(TEST_FILE)
# agn2_tot_flux = agn_line_df(TEST2_FILE)
# agn_no_tot_flux = sf_line_df(TEST1_FILE)

# residuals_baseline = (agn1_tot_flux - fluxes.values)/fluxerr.values
# residuals_masked = (agn2_tot_flux  - fluxes.values)/fluxerr.values
# residuals_NO = (agn_no_tot_flux  - fluxes.values)/fluxerr.values

# # print(residuals_masked)

# residuals_baseline['model'] = 'AGN+SF Mup100 Default'
# residuals_masked['model'] = 'AGN+SF Mup100 Line Masked'
# residuals_NO['model'] = 'AGN+SF Mup100 NO Line Masked'
# sf_tot_flux['model'] = 'AGN+SF Mup100 NO Grid'

# merge = pd.concat((residuals_baseline, residuals_masked, residuals_NO))

#long_df = make_long_df(merge)

#fig, ax = plt.subplots(1, 1, figsize = (12, 6), tight_layout = True)
#violin_plot_fluxes_models(long_df, ax)
#violin_plot_fluxes_models(long_df, ax)
#add_all_fluxes(ax, lines, fluxes, fluxerr, ulims)
#ax.set_ylim(-1e-18, 1e-17)
#ax.set_ylabel(f'(model - data)/error', fontsize = 15)
#plt.show()