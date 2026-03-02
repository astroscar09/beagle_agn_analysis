from astropy.table import Table
import pandas as pd
from line_fitting_bayesian_functions import *
import yaml 
import matplotlib.pyplot as plt
import seaborn as sb

with open('../line_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def read_spectrum():
    
    spectrum = pd.read_csv('Full_GHZ2_Spectrum_Rest_Frame.txt', sep = ' ')
    wave, spec, spec_err = spectrum['wave'].values, spectrum['flux'].values, spectrum['fluxerr'].values    

    return wave, spec, spec_err


masks = {'NIV': [[1520, 1590], [1625, 1700]], 
         'CIV': [[1460, 1520], [1625, 1700]], 
         'HeII_OIII': [[None, None]], 
         'NIII':      [[1525, 1590], [1625, 1700], [1880, 1930]],
         'CIII':      [[None, None]], 
         'MgII':      [[2750, 2770], [2820, 2850], [2895, 2930], [2960, 3005]], 
         'OII':       [[3740, 3750], [3850, 3900]], 
         'NeIII':     [[None, None]], 
         'OIII5007':  [[None, None]], 
         'halpha':    [[None, None]]}


def read_line_info(config, line):

    line_dict = config['line']
    read_line = line_dict[line]

    fit_type = read_line['fit_type']

    line_center = read_line['center']
    min_window = read_line['window_min']
    max_window = read_line['window_max']
    intial_condition = read_line['intitial_condition']
    
    line_ratio1 = None
    line_ratio2 = None
    
    if read_line['fit_type'] == 'double_gaussian':
        line_ratio1 = read_line['line_ratio']
    elif read_line['fit_type'] == 'triple_gaussian':
        line_ratio1 = read_line['line_ratio1']
        line_ratio2 = read_line['line_ratio2']

        
    return fit_type, line_center, min_window, max_window, intial_condition, line_ratio1, line_ratio2
    
def mask_spectra(wave, flux, fluxerr, line_center, lwindow, rwindow, line):

    mask = (wave > line_center - lwindow) & (wave < line_center + rwindow)

    xin = wave[mask]
    yin = flux[mask] *1e20
    yerr_in = fluxerr[mask]*1e20

    final_mask = np.ones(len(xin), dtype = bool)
    for bounds in masks[line]:
        if bounds[0] is None:
            continue
        else:
            line_mask = (xin > bounds[0]) & (xin < bounds[1])
            final_mask = final_mask & ~line_mask

    return xin[final_mask], yin[final_mask], yerr_in[final_mask]

def fit_line(config, line):

    wave, spec, spec_err = read_spectrum()
    fit_type, line_center, min_window, max_window, initial_condition, line_ratio1, _ = read_line_info(config, line)
    xin, yin, yerr_in = mask_spectra(wave, spec, spec_err, line_center, min_window, max_window, line)

    if fit_type == 'single_gaussian':
        fit_class = fit_single_gaussian(xin, yin, yerr_in, initial_condition, line_center, line)

    elif fit_type == 'double_gaussian':
        line2 = line_center*line_ratio1 #x, y, yerr, initial_guess, line1, line2, tied
        fit_class = fit_double_gaussian(xin, yin, yerr_in, initial_condition, line_center, line2)
        #df = fit_class.fit_spectrum()

    elif fit_type == 'triple_gaussian':
        
        fit_class = fit_triple_gaussian(xin, yin, yerr_in, initial_condition, line_center)
    
    df = fit_class.fit_spectrum()

    #fit_class.plot_best_fit_model()

    return df

#print(config.keys())
if __name__ == '__main__':

    for l in config['line'].keys():
        print(f'Fitting Line: {l}')
        line_fit = fit_line(config, l)
        line_fit.to_csv(f'line_fits/{l}_emcee_fit.txt', sep = ' ', index = False)