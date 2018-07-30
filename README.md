#############################################################################################################################################
#############################################################################################################################################

She-ra (python suite to work with anaconda3)
Stellar parameters estimation tHrough full-spEctrum fitting Restricted by spectral features, using Any set of stellar templates 

Copyright (C) 2017-2018, Sofia Meneses-Goytia (SMG)
e-mail: smg@astro-research.net or s.menesesgoytia@gmail.com

Updated versions of the software are available from github repository and my website
https://github.com/Astro-SMG | https://smg.astro.research.net

If you have found this software useful for your research, 
I would appreciate an acknowledgment to the use of:
"Stellar parameters estimation tHrough full-spEctrum fitting Restricted by spectral features, using Any set of stellar templates 
(She-ra) by Meneses-Goytia et al. in prep"

This software is provided as is without any warranty whatsoever. 
Permission to use, for non-commercial purposes is granted provided users site Meneses-Goytia+ in prep. where the full suite is detailed.
Permission to modify for personal or internal use is granted, provided this copyright and disclaimer are included unchanged at the beginning of the file.
All other rights are reserved.

#############################################################################################################################################
#############################################################################################################################################

Version 1.0_alpha
  SMG, 2017-2018
  Institute of Cosmology and Gravitation (ICG) - University of Portsmouth (UoP), Portmouth, UK

  Current version has been built to work only with the MaNGA stellar library (MaStar).
  It reads the file with the specific information of MaStar visits.
  The parameters can be calculated for sets of n_size (one or many).

  Required routines:
    calculate_magnitudes:
      calculates the synthetic magnitudes of any spectrum using any filter curves definitions (SMG, 2013-2018) - provided
    calculate_indices: 
      calculates the line-strength absorption indices of any spectrum using any definition (SMG, 2013-2018) - provided
    estimations_3d: 
      interpolates z values from a 3D surface, starting from an irregular set of x, y and z points (SMG, 2017) - provided
    dust_functions: 
      calculates the redenning correction for a given star according to its coordinates (SMG, 2017 based on FF routines) - provided
    pPXF: 
      full-spectrum fitting (FSF) of a spectrum using a set of templates (M. Capellari, 2001-2018) - not provided

  Input file must include:
     mangaid, plate, ifudesign, mjd: pipeline information
     ra, dec: coordinates
     wavelength, flux: spectral information

  Used templates: 
    ATLAS9 - Meszaros+2012, Bohlin+2017
    MARCS - Gustafsson+2008

    Treated with prepare_atlas9_marcs_templates.py - provided
      Set-up to MaStar wavelength range and bin size
      Convolved to an homogenous R of 2000 (higher than MaStar)
      Synthetic magnitudes and line-strength absorption indices calculated 
    
  The priors for the stellar parameters are calculated based on the templates' relations:
    Teff as a function of (g-i)
    log g as a function of Na I from La Barbera
    metallicity as a function of CaT and Teff

  The size of the boxes to select the templates to used based on the priors are:
    Teff >= 12000: teff_guess = 2000
    Teff <  12000: teff_guess =  500
    logg_guess                = 0.50
    metal_guess               = 0.35

  Output file for each visit includes:
                            identifiers:
                                        mangaid, mastarid, id, plate, ifudesign, mjd

                            coordinates: 
                                        objra, objdec

                            teff1, logg1, metal1: 
                                                  priors for effective temperature, surface gravity, and metallicity

                            teff2, logg2, metal2:
                                                  weighted average parameters of all the templates used in the FSF - named as set1

                            teff3, logg3, metal3:
                                                  parameters of template with the largest weight - named as set2

                            chi2,    vel,  sigma:
                                                  estimates for chi2/DOF, radial velocity (km s-1) and velocity dispersion (km s-1)

                            magnitudes: 
                                        Umag Gmag Rmag Imag Zmag

                            indices:
                                     OII3727 HdA HdF CN1 CN2 Ca4227 G4300 HgA HgF Fe4383 Ca4455 Fe4531 Fe4668 Hbeta Hbetap OIII1 OIII2 
                                     Fe5015 Mg1 Mg2 Mgb5177 Fe5270 Fe5335 Fe5406 Fe5709 Fe5782 NaD TiO1 TiO2 NaILaB NaISp NaICo 
                                     Ca1 Ca2 Ca3 MgI Ca1AZ Ca2AZ Ca3AZ CaT PaT sCaT


  Output plot of the best-fit in png format

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------

Calling sequence (in terminal)
  python she-ra_beta_ver "nstart nend"

  nstar: first visit of the set for which the parameters are calculated
  nend: last visit of the set  

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------

Modification history

  Version 1.0_beta
  SMG, Jan-Jul 2018
  ICG - UoP, Portmouth, UK

    Used templates:
      ATLAS9MARCS used with a denser parameter coverage in effective temperature, surface gravity, and metallicity.
      Grid created by merol3.py - provided - using the templates previously set-up by prepare_atlas9_marcs_templates.py

    Input file must also include:
      mastarid: unique identifier for each visit
      minlogg: log g as a function of Gaia DR2 colors using isoparfit.pro -provided - that fits the observed photometry to the photometry of isochrones

    New priors for the stellar parameters are calculated as follows:
      Teff as a function of (g-i) - same as alpha_version
      log g as prior from isoparfit.pro
      metallicity is left as an free parameter this it is not a real prior

    The size of the boxes to select the templates to used based on the priors are as follows:
      Teff >= 12000: teff_guess = 2000
      Teff <  12000: teff_guess = 1000 (wider box for cooler stars)
      logg_guess                = 0.80 (wider box for all stars)
      metal_guess               = 2.0 (metallicity is now a free parameter)

  -------------------------------------------------------------------------------------------------------------------------------------------

  Version 1.0_gamma
  SMG, Jul-onwards 2018
  ICG - UoP, Portmouth, UK

    To include:
      The isoparfit routine integrated into She-ra
      A better quantification of the chi2/DOF for the template with the largest weight and the best-fiting combination of templates
      Real noise from the MaStar spectra
      Masking of bad pixels


#############################################################################################################################################
#############################################################################################################################################
