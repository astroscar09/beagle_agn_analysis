# BEAGLE-AGN Analysis

## Background

GHZ2 is a galaxy at a redshift of z = 12.34, making it one of the earliest known galaxies with spectroscopic evidence of metal enrichment. Its spectrum reveals high-ionization emission lines that are difficult to explain with stellar populations alone.

This raises a fundamental question:

       What powers the spectrum of GHZ2?

- Extremely massive, metal-poor stars?

- An actively accreting black hole (AGN)?

- A combination of both?

To address this, we use BEAGLE-AGN, an advanced spectral modeling framework that jointly incorporates:

- Stellar population synthesis

- Nebular emission

- Active Galactic Nucleus (AGN) contributions

We run an extensive suite of models across multiple parameter grids and physical assumptions to explore all plausible explanations for the observed spectrum.

Our goal is to determine the physical mechanisms driving the ionizing radiation field in one of the earliest galaxies currently known.

## Dataset

### Photometry
We use photometry from Castellano and this was provided by Jorge Zavala

### Spectroscopy
The Spectroscopic data that we used was the NIRSpec PRISM which covered the spectrum all the way up to rest-frame ~3800 angstroms. We supplemented this data with MIRI spectroscopy and this covers the rest-frame optical and extends the NIRSPEC PRISM coverage albeit at a different resolution. 

In the PRISM spectrum we have access to the following emission lines:

- **N IV] λ1486**
- **C IV λλ1548, 1550**
- **He II λ1640**
- **O III] λλ1661, 1666**
- **N III] λ1750**
- **C III] λλ1907, 1909**
- **Mg II λλ2796, 2803**

and in the MIRI spectrum, the optical emission lines we have access to are: 

- **Hβ λ4861**
- **[O III] λ4960**
- **[O III] λ5007**
- **Hα λ6563**


## Methods