from plotting_utils import plot_agn_fraction
from data_processing import merge_doublets, agn_flux_fraction


def main_ratio(hii_em_df, agn_em_df):

    merge_doublets(hii_em_df, agn_em_df)
    ratio = agn_flux_fraction(hii_em_df, agn_em_df)
    cols = [r'NIV$\lambda$1486', r'CIV$\lambda$1548+51', r'HeII$\lambda$1640', r'OIII$\lambda$1660+66',
         r'NIV$\lambda$1729', r'NIII$\lambda$1750',
       r'CIII$\lambda$1907+10', r'MgII$\lambda$2796+2803', r'H$\beta$',
       r'[OIII]$\lambda$4959', r'[OIII]$\lambda$5007', r'H$\alpha$',
       r'[O III]52$\mu$m', r'[O III]88$\mu$m']
    fig, ax = plot_agn_fraction(ratio[cols])
    #plt.tight_layout()
    return fig, ax 