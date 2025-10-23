import seaborn as sb
from Post_BEAGLE_Analysis import *

AGN_Default = '/scratch/07446/astroboi/BEAGLE_TACC/results/AGN_Mup100_Default_Run/1395_BEAGLE_Default_Run.fits.gz'
AGN_Mup100_CO = '/scratch/07446/astroboi/BEAGLE_TACC/results/AGN_Mup100_CO_Grids/1395_BEAGLE_Mup100_CO_Grids.fits.gz'
AGN_Mup100_logU = '/scratch/07446/astroboi/BEAGLE_TACC/results/AGN_Mup100_logU/1395_BEAGLE_AGN_Mup100_logU.fits.gz'
AGN_Mup300 = '/scratch/07446/astroboi/BEAGLE_TACC/results/AGN_Mup300_Run/1395_BEAGLE_AGN_Mup300.fits.gz'
AGN_Mup300_CO = '/scratch/07446/astroboi/BEAGLE_TACC/results/AGN_Mup300_CO_Grids/1395_BEAGLE_AGN_Mup300_CO_Grids.fits.gz'
AGN_Mup300_logU = '/scratch/07446/astroboi/BEAGLE_TACC/results/AGN_Mup300_logU/1395_BEAGLE_AGN_Mup300_logU.fits.gz'
SF_Mup100 = '/scratch/07446/astroboi/BEAGLE_TACC/results/AGN_Mup300_logU/1395_BEAGLE_SF_Mup100.fits.gz'
SF_Mup300 = '/scratch/07446/astroboi/BEAGLE_TACC/results/TEST_SF_Mup300/1395_BEAGLE.fits.gz'

output_files = [AGN_Default, AGN_Mup100_CO, AGN_Mup100_logU, AGN_Mup300, AGN_Mup300_CO, AGN_Mup300_logU, SF_Mup100, SF_Mup300]

def read_nirspec_data(file= '/scratch/07446/astroboi/BEAGLE_TACC/data/spectra/1395_Only_Spec.fits'):

    nirspec_table = Table.read(file)
    
    return nirspec_table

def get_spec_variables(tab):

    wave, flux, ferr = tab['wav'], tab['flux'], tab['err']
    
    return wave, flux, ferr
    
def read_bgl_object(file):

    bgl_output = Beagle_Output(file)
    return bgl_output

def get_posterior(bgl_output):

    return bgl_output['posterior']

def convert_table_to_pandas(tab):

    return tab.to_pandas()

def make_corner_plot_posterior(df, agn_fit = True, save = False, filename = None):
    
    columns = ['mass', 'nebular_logu', 'nebular_z',  
               'metallicity', 'specific_sfr']

    if agn_fit:
        
        agn_cols = ['agn_lacc', 'agn_logu', 'agn_z', 'agn_xi',]
        columns.extend(agn_cols)
        
    g = sb.pairplot(df[columns], corner = True)
    g.map_lower(sb.kdeplot, levels=[.16, .5, .84], color='black')

    if save:
        if filename:
            plt.savefig(filename, dpi = 300)

    plt.show()        

def BIC_Criterion(lnL, k, n):
    return -2 * lnL + k * np.log(n)

def compute_BIC(df):
    
    num_parameters = df.shape[1] - 4
    n_sample = df.n_data.values[0]
    
    return BIC_Criterion(df['ln_likelihood'], num_parameters, n_sample)

def plot_w_ax(wave, flux, ax, label):

    ax.step(wave, flux, where = 'mid', color = 'purple', label = label)
    
def main(file, agn_save = False, filename = None):

    bgl_output = read_bgl_object(file)
    post_tab = get_posterior(bgl_output)
    post_df = convert_table_to_pandas(post_tab)
    make_corner_plot_posterior