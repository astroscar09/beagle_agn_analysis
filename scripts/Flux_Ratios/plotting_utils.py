import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def plot_agn_fraction(ratio, type = 'boxplot'):

    fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    
    if type == 'boxplot':
        sb.boxplot(ratio, ax = ax)

    elif type == 'violinplot':
        sb.violinplot(ratio, ax = ax)

    ax.tick_params(axis='x', labelrotation=45)

    medians = ratio.median()
    xs = np.arange(len(medians))
    labels = medians.index
    values = medians.values

    for label in ax.get_xticklabels():
        label.set_ha('right')
        label.set_rotation_mode('anchor')

    ax.set_ylabel('Fractional AGN Contibution', fontsize = 20)
    ax.grid(alpha = 0.5, ls = '--')

    return fig, ax


def add_all_fluxes(ax, lines, fluxes, errors, upper_lims):

    xpos = np.arange(len(lines))
    dx = 0.0    

    ax.scatter(
                xpos + dx, fluxes,
                color='purple', marker='s',
                zorder=100, label='Observed Flux'
                )

    ax.errorbar(
                    xpos + dx, fluxes,
                    yerr=errors,
                    uplims=upper_lims,
                    fmt='none',
                    color='purple',
                    capsize=3,
                    zorder=99
                )
    

    ax.set_xticks(xpos)
    ax.set_xticklabels(lines, rotation=45, ha='right')
    


def plot_bgl_fluxes(df, ax,):
    
    lines  = ['NIV1488', 'CIV1549', 'HeII1640', 'OIII]1663', 'NIII]1750', 'CIII]1908', 'NeIV2424', 'MgII', 'NeV3426', 'OII3727', 'NeIII3868', 
              r'H$\beta$', 'OIII4959', 'OIII5007', r'H$\alpha$', 
              r'[OIII]52$\mu$m', r'[OIII]88$\mu$m']

    xpos = np.arange(len(lines))
    dx = 0  

    remap = {x: y for x, y in zip(df.columns, lines)}

    #plt.figure(figsize = (12, 6))
    sb.violinplot(df.rename(columns = remap), ax = ax)

    ax.set_xticks(xpos)
    ax.set_xticklabels(lines, rotation=45, ha='right')


def violin_plot_fluxes_models(df, ax):

    lines  = ['NIV1488', 'CIV1549', 'HeII1640', 'OIII]1663', 'NIII]1750', 'CIII]1908', 'NeIV2424', 'MgII', 'NeV3426', 'OII3727', 'NeIII3868', 
              r'H$\beta$', 'OIII4959', 'OIII5007', r'H$\alpha$', 
              r'[OIII]52$\mu$m', r'[OIII]88$\mu$m']

    xpos = np.arange(len(lines))
    dx = 0  

    ax = sb.violinplot(df, x = 'line', y = 'flux', hue = 'model', dodge = True, ax = ax, width = 0.5, density_norm='width')
    ax.legend()
    ax.grid()
    ax.set_xticks(xpos)
    ax.set_xticklabels(lines, rotation=45, ha='right')



def plot_corner_plot(post_df):

    sb.pairplot(post_df, corner = True, 
                plot_kws={'marker':'o', 's':2, 'alpha': 0.5, 'color': 'gray'})
    
