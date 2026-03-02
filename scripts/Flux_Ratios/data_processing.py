import numpy as np
from astropy.cosmology import Planck18 as cosmo 
from astropy import units as u
from astropy.table import Table

def agn_emission_lines_hdu(hdu):

    extension = 'AGN EMISSION'
        
    agn_emission = Table(hdu[extension].data).to_pandas()

    return agn_emission

def hii_emission_lines_hdu(hdu):

    extension = 'HII EMISSION'
        
    hii_emission = Table(hdu[extension].data).to_pandas()

    return  hii_emission

def get_redshift(hdu):

    return Table(hdu['POSTERIOR PDF'].data).to_pandas()['redshift'].values

def get_posterior(hdu):

    return Table(hdu['POSTERIOR PDF'].data).to_pandas()


def merge_doublets(hii_em_df, agn_em_df = None):

    doublets = {'O3_1661_1666': ['O3_1661_lum_obs', 'O3_1666_lum_obs'], 
            'C4_1548_1551':['C4_1548_lum_obs', 'C4_1551_lum_obs'], 
            'C3_1907_1910': ['C3_1907_lum_obs', 'C3_1910_lum_obs'], 
            'Mg2_2796_2803': ['Mg2_2796_lum_obs', 'Mg2_2803_lum_obs'], 
            'O2_3727_3729': ['O2_3726_lum_obs', 'O2_3729_lum_obs']}

    for key, vals in doublets.items():

        hii_sum = np.sum(hii_em_df[vals].values, axis = 1)
        hii_em_df[key] = hii_sum

        if agn_em_df is not None:        
            agn_sum = np.sum(agn_em_df[vals].values, axis = 1)
            agn_em_df[key] = agn_sum

    return hii_em_df, agn_em_df

def agn_flux_fraction(hii_em_df, agn_em_df):

    lines_to_check = [  'N4_1486_lum_obs', 'HeBaA_1640_lum_obs', 'O3_1661_1666', 
                        'C4_1548_1551', 'N4_1719_lum_obs', 'N3_1750_lum_obs', 'C3_1907_1910', 'Mg2_2796_2803', 'Ne3_3869_lum_obs', 
                        'HBaB_4861_lum_obs', 'O3_4959_lum_obs', 'O3_5007_lum_obs', 'HBaA_6563_lum_obs', 
                        'O3_518000_lum_obs', 'O3_883300_lum_obs']
    
    remap = {'N4_1486_lum_obs': r'NIV$\lambda$1486', 
             'HeBaA_1640_lum_obs': r'HeII$\lambda$1640', 
             'O3_1661_1666': r'OIII$\lambda$1660+66', 
            'C4_1548_1551': r'CIV$\lambda$1548+51', 
            'N4_1719_lum_obs': r'NIV$\lambda$1729', 
            'N3_1750_lum_obs': r'NIII$\lambda$1750', 
            'C3_1907_1910': r'CIII$\lambda$1907+10', 
            'Mg2_2796_2803': r'MgII$\lambda$2796+2803', 
            'HBaB_4861_lum_obs': r'H$\beta$', 
            'O3_4959_lum_obs': r'[OIII]$\lambda$4959', 
            'O3_5007_lum_obs': r'[OIII]$\lambda$5007', 
            'HBaA_6563_lum_obs': r'H$\alpha$', 
            'O3_518000_lum_obs': r'[O III]52$\mu$m', 
            'O3_883300_lum_obs': r'[O III]88$\mu$m'}
    

    agn_lines = agn_em_df[lines_to_check]
    hii_lines = hii_em_df[lines_to_check]

    ratio = agn_lines/(agn_lines+hii_lines.values)

    return ratio.rename(columns = remap)

def get_luminosity_distance(z):
    return cosmo.luminosity_distance(z).to(u.cm)


def convert_lum_to_fluxes(line_df, DL):

    #['NIV1488', 'CIV1549', 'HeII1640', 'OIII]1663', 'NIII]1750', 'CIII]1908', 'NeIV2424', 'MgII', 'NeV3426', 'OII3727', 'NeIII3868', 
    #          r'H$\beta$', 'OIII4959', 'OIII5007', r'H$\alpha$', 
    #          r'[OIII]52$\mu$m', r'[OIII]88$\mu$m']

    lines_to_check = [  'N4_1486_lum_obs', 'C4_1548_1551', 'HeBaA_1640_lum_obs', 'O3_1661_1666', 
                        'N3_1750_lum_obs', 'C3_1907_1910', 'Ne4_2424_lum_obs', 'Mg2_2796_2803', 'Ne5_3426_lum_obs', 'O2_3727_3729', 'Ne3_3869_lum_obs',
                        'HBaB_4861_lum_obs', 'O3_4959_lum_obs', 'O3_5007_lum_obs', 'HBaA_6563_lum_obs', 'O3_518000_lum_obs', 'O3_883300_lum_obs']
    
    remap = {   'N4_1486_lum_obs': 'N4_1486_flux', 
                'C4_1548_1551': 'C4_1548_1551_flux', 
                'HeBaA_1640_lum_obs': 'HeBaA_1640_flux', 
                'O3_1661_1666': 'O3_1661_1666_flux',  
                'N3_1750_lum_obs': 'N3_1750_flux', 
                'C3_1907_1910': 'C3_1907_1910_flux',
                'Ne4_2424_lum_obs': 'Ne4_2424_flux', 
                'Mg2_2796_2803': 'Mg2_2796_2803_flux',
                'Ne5_3426_lum_obs': 'Ne5_3426_flux',
                'O2_3727_3729':'O2_3727_3729_flux',
                'Ne3_3869_lum_obs': 'Ne3_3869_flux',
                'HBaB_4861_lum_obs': 'HBaB_4861_flux', 
                'O3_4959_lum_obs': 'O3_4959_flux', 
                'O3_5007_lum_obs': 'O3_5007_flux', 
                'HBaA_6563_lum_obs': 'HBaA_6563_flux',
                'O3_518000_lum_obs': 'O3_518000_flux', 
                'O3_883300_lum_obs': 'O3_883300_flux'
            }


    L_ergs_em_df = line_df.astype(np.float64) * 3.826e33
    flux_em_df = L_ergs_em_df[lines_to_check]/(4 * np.pi * DL.value.reshape(-1, 1)**2)
    flux_em_df = flux_em_df.rename(columns =  remap)

    return flux_em_df


def get_total_emission(hii_df, agn_df):
    
    return hii_df + agn_df.values


def make_long_df(df):

    df_long = df.melt(
                        id_vars='model',
                        var_name='line',
                        value_name='flux'
                    )
    
    return df_long