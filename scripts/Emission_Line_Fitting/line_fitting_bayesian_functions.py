import emcee
import numpy as np
import corner 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
from astropy.table import Table
from astropy.io import fits

class fit_single_gaussian():

    def __init__(self, x, y, yerr, initial_guess, line, line_name):

        self.x = x
        self.y = y
        self.yerr = yerr
        self.initial_guess = initial_guess
        self.line = line
        self.line_name = line_name

    def gaussian(self, x, amp, mean, sigma):
        return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

    def linear_model(self, x, m, b, center):
        return m * (x - center) + b

    def single_gaussian(self, x, A_n, mu_n, sigma_n, m, b):

        narrow = self.gaussian(x, A_n, mu_n, sigma_n)
        linear = self.linear_model(x, m, b, mu_n)

        return narrow + linear


    def log_prior_single(self, theta, og_center): 
        
        A1, mu1, sigma1, m, b = theta

        if 0 < A1 < 20 and -10 < m < 10 and -10 < b < 10 and (mu1 > og_center - 10) and (mu1 < og_center + 10) & (sigma1 > 0) & (sigma1 < 75):
            return 0.0
        
        return -np.inf

    def log_likelihood_single(self, theta, x, y, yerr):
        
        A1, mu1, sigma1, m, b = theta

            
        model = self.single_gaussian(x, A1, mu1, sigma1, m, b)
    

        return -0.5 * np.sum(((y - model) / yerr) ** 2 + np.log(2 * np.pi * yerr ** 2))

    def log_probability_single(self, theta, x, y, yerr, og_center):
        
        lp = self.log_prior_single(theta, og_center)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood_single(theta, x, y, yerr)
    
    def fit_spectrum(self, nwalkers=32, nsteps=5000):
    
        ndim = len(self.initial_guess)
        pos = self.initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
        og_center = self.initial_guess[1]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_single, args=(self.x, self.y, self.yerr, og_center))

        print(f"Running burn-in for {nsteps} steps...")
        pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)

        # Reset sampler to discard burn-in samples
        sampler.reset()

        sampler.run_mcmc(pos, nsteps, progress=True)

        flat_samples = sampler.get_chain( thin=15, flat=True)

        emcee_cols = ['A', 'mu', 'sigma', 'm', 'b']

        df = pd.DataFrame(flat_samples, columns = emcee_cols)

        flux = df.A.values * df.sigma.values * np.sqrt(2*np.pi)

        df[f'{self.line_name}_flux'] = flux

        self.df = df
        
        return df
    
    def plot_best_fit_model(self):

        xarr = np.linspace(self.x.min(), self.x.max(), 500)
        med_params = self.df.quantile(q = 0.5).values[:-1]

        model = self.single_gaussian(xarr, *med_params)

        gauss = self.gaussian(xarr, *med_params[:-2])
        line = self.linear_model(xarr, *med_params[-2:], med_params[1])

        plt.step(self.x, self.y, where = 'mid', color = 'gray')
        plt.errorbar(self.x, self.y, yerr = self.yerr, fmt = 'none', color = 'gray')
        plt.plot(xarr, model)
        plt.plot(xarr, line)
        
        plt.show()




    

class fit_double_gaussian():

    def __init__(self, x, y, yerr, initial_guess, line1, line2, tied = False):

        self.x = x
        self.y = y
        self.yerr = yerr
        self.initial_guess = initial_guess
        self.line1 = line1
        self.line2 = line2
        self.tied = tied

    
    def gaussian(self, x, amp, mean, sigma):
        return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

    def linear_model(self, x, m, b, center):
        return m * (x - center) + b

    def double_gaussian(self, x, A1, mu1, sigma1, A2, mu2, sigma2, m, b):

        gauss1 = self.gaussian(x, A1, mu1, sigma1)
        gauss2 = self.gaussian(x, A2, mu2, sigma2)
        linear = self.linear_model(x, m, b, mu1)

        return gauss1 + gauss2 + linear


    def log_prior_double(self, theta, og_center): 
        
        A1, mu1, sigma1, A2, sigma2, m, b = theta

        amp1_mask = (0 < A1) & (A1 < 20)
        amp2_mask = (0 < A2) & (A2 < 20)

        sigma1_mask = (sigma1 > 0) & (sigma1 < 50)
        sigma2_mask = (sigma2 > 0) & (sigma2 < 50)

        slope_mask = (-10 < m) & (m < 10)
        intercept_mask = (-10 < b) & (b < 10)

        mu1_mask = (mu1 > og_center - 10) & (mu1 < og_center + 10)

        if amp1_mask & amp2_mask & sigma1_mask & sigma2_mask & slope_mask & intercept_mask & mu1_mask:
            return 0.0
        
        return -np.inf

    def log_likelihood_double(self, theta, x, y, yerr, line1, line2, tied):
        
        A1, mu1, sigma1, A2, sigma2, m, b = theta

        mu2 = mu1 * (line2/line1)

        if tied:

            model = self.double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma1, m, b)

        else:
            
            model = self.double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, m, b)
    

        return -0.5 * np.sum(((y - model) / yerr) ** 2 + np.log(2 * np.pi * yerr ** 2))

    def log_probability_double(self, theta, x, y, yerr, og_center, line1, line2, tied):
        
        lp = self.log_prior_double(theta, og_center)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood_double(theta, x, y, yerr, line1, line2, tied)
    
    def fit_spectrum(self, nwalkers=32, nsteps=5000):
    
        ndim = len(self.initial_guess)
        
        pos = self.initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
        og_center = self.initial_guess[1]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_double, args=(self.x, self.y, self.yerr, og_center, self.line1, self.line2, self.tied))

        print(f"Running burn-in for {nsteps} steps...")
        pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)

        # Reset sampler to discard burn-in samples
        sampler.reset()

        sampler.run_mcmc(pos, nsteps, progress=True)

        flat_samples = sampler.get_chain( thin=15, flat=True)

        emcee_cols = ['A_OIII', 'mu_OIII', 'sigma_OIII', 'A_HeII', 'sigma_HeII', 'm', 'b']


        df = pd.DataFrame(flat_samples, columns = emcee_cols)

        OIII_flux = np.sqrt(2*np.pi) * df['A_OIII'].values * df['sigma_OIII'].values
        HeII_flux = np.sqrt(2*np.pi) * df['A_HeII'].values * df['sigma_HeII'].values
        
        df['HeII_flux'] = HeII_flux
        df['OIII_flux'] = OIII_flux
        
        self.df = df

        return df
    
    def plot_best_fit_model(self):

        xarr = np.linspace(self.x.min(), self.x.max(), 500)
        med_params = self.df.quantile(q = 0.5).values[:-1]

        new_med_params = [med_params[0], med_params[1], med_params[2], med_params[3], med_params[1] * (self.line2/self.line1), med_params[4], med_params[5], med_params[6]]

        model = self.double_gaussian(xarr, *new_med_params)

        line = self.linear_model(xarr, med_params[5], med_params[6], med_params[1])
        gauss1 = self.gaussian(xarr, med_params[0], med_params[1], med_params[2])
        gauss2 = self.gaussian(xarr, med_params[3], med_params[1] * (self.line2/self.line1), med_params[4])

        plt.step(self.x, self.y, where = 'mid', color = 'gray')
        plt.errorbar(self.x, self.y, yerr = self.yerr, fmt = 'none', color = 'gray')
        plt.plot(xarr, gauss1+line)
        plt.plot(xarr, gauss2+line)
        plt.plot(xarr, model)
        plt.show()


class fit_triple_gaussian():

    def __init__(self, x, y, yerr, initial_guess, line1, tied = False):

        self.x = x
        self.y = y
        self.yerr = yerr
        self.initial_guess = initial_guess
        self.line1 = line1
        self.tied = tied
    
    def gaussian(self, x, amp, mean, sigma):
        return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

    def linear_model(self, x, m, b, center):
        return m * (x - center) + b

    def triple_gaussian(self, x, A1, mu1, sigma1, A3, sigma3, m, b):

        A2 = A1/3
        mu2 = mu1*(4960/5007)
        mu3 = mu1*(4863/5007)
        sigma2 = sigma1

        gauss1 = self.gaussian(x, A1, mu1, sigma1)
        gauss2 = self.gaussian(x, A2, mu2, sigma2)
        gauss3 = self.gaussian(x, A3, mu3, sigma3)
        linear = self.linear_model(x, m, b, mu1)

        return gauss1 + gauss2 + gauss3 + linear


    def log_likelihood_triple(self, theta, x, y, yerr):
        
        A1, mu1, sigma1, A3, sigma3, m, b = theta
        model = self.triple_gaussian(x, A1, mu1, sigma1, A3, sigma3, m, b)
        
        return -0.5 * np.sum(((y - model) / yerr) ** 2 + np.log(2 * np.pi * yerr ** 2))

    def log_prior_triple(self, theta, og_center): 
        
        A1, mu1, sigma1, A3, sigma3, m, b = theta

        amp_mask = (0 < A1) & (A1 < 20)
        slope_mask = (-10 < m) & (m < 10)
        intercept_mask = (-10 < b) & (b < 10)
        line_center_mask = (mu1 > og_center - 50) & (mu1 < og_center + 20)
        sigma1_mask = (sigma1 < 75) & (sigma1 > 0)
        sigma3_mask = (sigma3 < 75) & (sigma3 > 0)
        
        amp3_mask = (A3 > -5)& (A3 < 20)

        if amp_mask and slope_mask and intercept_mask and line_center_mask and sigma1_mask & sigma3_mask & amp3_mask:
            return 0.0
        
        return -np.inf

    def log_probability_triple(self, theta, x, y, yerr, og_center):
        
        lp = self.log_prior_triple(theta, og_center)
        
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood_triple(theta, x, y, yerr)
    

    def fit_spectrum(self, nwalkers=32, nsteps=5000):
    
        ndim = len(self.initial_guess)
        
        pos = self.initial_guess + np.random.randn(nwalkers, ndim)*1e-2
        og_center = self.initial_guess[1]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_triple, args=(self.x, self.y, self.yerr, og_center))

        print(f"Running burn-in for {nsteps} steps...")
        pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)

        # Reset sampler to discard burn-in samples
        sampler.reset()

        sampler.run_mcmc(pos, nsteps, progress=True)

        flat_samples = sampler.get_chain( thin=15, flat=True)

        emcee_cols = ['Amp_5007', 'mu_5007', 'sigma_oiii', 'Amp_hb', 'sigma_hb', 'slope', 'y-int']

        df = pd.DataFrame(flat_samples, columns = emcee_cols)

        OIII5007 = np.sqrt(2*np.pi) * df.Amp_5007.values* df.sigma_oiii.values
        OIII4960 = np.sqrt(2*np.pi) * df.Amp_5007.values/3 * df.sigma_oiii.values
        HBETA = np.sqrt(2*np.pi) * df.Amp_hb.values * df.sigma_hb.values

        df['Hbeta_Flux'] = HBETA
        df['OIII4960_Flux'] = OIII4960
        df['OIII5007_Flux'] = OIII5007

        self.df = df

        return df
    
    def plot_best_fit_model(self):

        xarr = np.linspace(self.x.min(), self.x.max(), 500)
        med_params = self.df.quantile(q = 0.5).values[:-3]

        model = self.triple_gaussian(xarr, *med_params)

        line = self.linear_model(xarr, med_params[-2], med_params[-1], med_params[1])
        gauss5007 = self.gaussian(xarr, med_params[0], med_params[1], med_params[2])
        gauss4960 = self.gaussian(xarr, med_params[0]/3, med_params[1]*(4960/5007), med_params[2])
        hbeta = self.gaussian(xarr, med_params[3], med_params[1]*(4863/5007), med_params[4])
        plt.step(self.x, self.y, where = 'mid', color = 'gray')
        plt.errorbar(self.x, self.y, yerr = self.yerr, fmt = 'none', color = 'gray')
        plt.plot(xarr, model)
        plt.plot(xarr, hbeta+line)
        plt.plot(xarr, gauss4960+line)
        plt.plot(xarr, gauss5007+line)
        plt.show()
