# BEAGLE-AGN Analysis

## Background

GHZ2 is a galaxy at a redshift of z = 12.34, making it one of the earliest known galaxies with spectroscopic evidence of metal enrichment. Its spectrum exhibits high-ionization emission features that are difficult to explain using stellar populations alone.

This motivates the central scientific question of this work:

       What powers the spectrum of GHZ2?

Possible explanations include:

- Extremely massive, metal-poor stars?

- An actively accreting black hole (AGN)?

- A combination of stellar and AGN-driven ionizing radiation

To address this, we use BEAGLE-AGN, an advanced spectral modeling framework that jointly incorporates:

- Stellar population synthesis
- Nebular emission
- Active Galactic Nucleus (AGN) contributions

To investigate this, we use BEAGLE-AGN, a spectral modeling framework that simultaneously incorporates stellar population synthesis, nebular emission, and AGN contributions.

We perform an extensive suite of models across multiple parameter grids and physical assumptions to explore all physically plausible interpretations of the observed spectrum.

The primary objective is to constrain the physical mechanisms responsible for the ionizing radiation field in one of the earliest known galaxies.

## Dataset

### Photometry
The photometric data used in this analysis are based on the catalog from Castellano et al., and were provided by Jorge Zavala.

### Spectroscopy
Spectroscopic observations were obtained using the NIRSpec PRISM, which covers the spectrum up to approximately rest-frame ~3800 Å. To extend wavelength coverage into the optical regime, MIRI spectroscopy was also incorporated. Although the MIRI data provide complementary spectral coverage, they operate at a different spectral resolution than the PRISM data.

The PRISM spectrum provides access to the following ultraviolet emission lines:

- **N IV] λ1486**
- **C IV λλ1548, 1550**
- **He II λ1640**
- **O III] λλ1661, 1666**
- **N III] λ1750**
- **C III] λλ1907, 1909**
- **Mg II λλ2796, 2803**
- **[O II] λλ3726, 3729**
- **[Ne III] λ3869**

The MIRI spectrum enables analysis of key optical emission lines, including:

- **Hβ λ4861**
- **[O III] λ4960**
- **[O III] λ5007**
- **Hα λ6563**


## Methods