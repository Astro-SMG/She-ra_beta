'''
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

'''


#############################################################################################################################################
#############################################################################################################################################

import matplotlib
matplotlib.use('Agg')

import sys
for arg in sys.argv:
    print(arg)

import sys

from astropy.io import fits
import numpy as np

import matplotlib.pyplot as plt

from dust_functions import *

from calculate_magnitudes import *
from calculate_indices import *

from scipy.interpolate import interp1d
from estimations_3d import estimation

from ppxf import ppxf
import ppxf_util as util

from time import clock

prelim = str.split(arg)
nstart = int(prelim[0])
nend = int(prelim[1])


#############################################################################################################################################
#############################################################################################################################################

def interface_sets(nstart,nend):

  full_time = clock()

  ####### setting up some variables for magnitudes and indices calculation ########

  c = 299792.458

  filter_list = 'sdss_manga_pipeline.res'
  zero_point = 'AB'
  redshift = 0.0

  indices_list = 'MaNGA_range.def'

  print('     > gathering info')

  ######## setting up directories and files ########

  data_dir = '/mnt/lustre/smg/stellar_libraries/MaNGA/'
  output_dir = '/mnt/lustre/smg/stellar_libraries/MaNGA/parameters/she-ra/beta_tests/combo_test/'

  file_spectra = 'MaStar_spectra_ebv_gaia_rb_isopars'

  ##### templates information #####

  templates_dir = '/mnt/lustre/smg/stellar_libraries/atlas9marcs/'
  templates_parameters = 'templates_grid_parameters_R2000.fits'

  header = fits.open(templates_dir+templates_parameters)

  templ_id, templ_teff, templ_logg, templ_metal = header[1].data['NAME'], header[1].data['TEFF'], header[1].data['LOGG'], header[1].data['METAL']
  templ_gmag, templa_rmag, templ_imag = header[1].data['Gmag'], header[1].data['Rmag'], header[1].data['Imag']
  
  tot_templ = len(templ_id)

  estimate_teff = interp1d(templ_gmag-templ_imag,templ_teff,bounds_error=False,fill_value='extrapolate')
  
  ######## reading spectra file ########

  print('     > reading information from the spectra file')

  header = fits.open(data_dir+'spectra/'+file_spectra+'.fits')

  mangaid, mastarid, starid            = header[1].data['mangaid'], header[1].data['mastarid'], header[1].data['id']  
  plate, ifudesign, mjd                = header[1].data['plate'], header[1].data['ifudesign'], header[1].data['mjd']
  ra, dec                              = header[1].data['objra'], header[1].data['objdec']
  wave, flux                           = header[1].data['wave'], header[1].data['flux']

  minlogg = header[1].data['minlogg']
  bad_info = np.where(minlogg <= -8)
  minlogg[bad_info] = 'NaN'

  nspectra = int(header[1].header['naxis2'])

  naxis1 = int(np.shape(wave)[1])

  crval1 = min(wave[0,:])
  crval2 = max(wave[0,:])
  cdelt1 = abs(crval1-crval2)/naxis1

  new_wave = np.linspace(start=crval1,stop=crval2,num=naxis1)


  ######## setting up arrays ########

  parameters = np.zeros((nspectra,9))
  selected_parameters = np.zeros((nspectra,3))
  ppxf_info = np.zeros((nspectra,3))

  indices = np.zeros((nspectra,42))
  mags = np.zeros((nspectra,5))

  #############################################################################################################################################

  ######## running things for each spectrum ########

  print('     > working on each spectrum')

  for i in range(nstart,nend+1):

    start_time = clock()

    new_flux = np.interp(new_wave, wave[i,:],flux[i,:])
    new_flux = new_flux*1e-17 # to have 10^-17 erg/s/cm2/Angstrom as units

    ######## calculating the E(B-V) from Schlegel+98 as a function of coordinates ########

    print('          > calculating the E(B-V) from Schlegel+98')
    ebv = get_dust_radec(ra[i],dec[i],'ebv')

    ######## calculating the reddening as a function of wavelength ########

    print('          > calculating the reddening')
    reddening = dust_allen_py(ebv,new_wave)

    ######## in order to correct, the flux needs to be divided by the reddening ########

    corrected_flux = new_flux/reddening

    bad_flux = np.isnan(corrected_flux) | np.isinf(corrected_flux) | (corrected_flux <= 0.0)
    corrected_flux[bad_flux] = 0.0

    ######## calculating the indices ########
    print('          > calculating indices within the MaNGA wavelength range')
    indices[i,:], err_indices = calculate_indices(new_wave,corrected_flux,indices_list,mastarid[i],ef=0,rv=0,plot=False,sim=False)

    ######## calculating the magnitudes ########
    print('          > calculating SDSS magnitudes')
    mags[i,:] = calculate_magnitudes(new_wave,corrected_flux,filter_list,zero_point,redshift)

    ######## calculating the first guess for the stellar parameters ########

    print('          > calculating the first set of teff, logg, and metallicity')

    parameters[i,0] = estimate_teff(mags[i,1]-mags[i,3]) # Teff as a function of (g-i)
    parameters[i,1] = minlogg[i] # from isoparsfit - isochrone fitting
    parameters[i,2] = -1.0 # open metallicity

    bad_data = np.isnan(parameters[i,0]) | np.isinf(parameters[i,0]) | np.isnan(parameters[i,1]) | np.isinf(parameters[i,1]) | np.isnan(parameters[i,2]) | np.isinf(parameters[i,2])
    if bad_data: 
        parameters[i,:] = -999
        ppxf_info[i,:] = -999
        ##### making a preliminary output file #####
        print('          > making a preliminary output file')
        f = open(output_dir+'prelim_output/'+mastarid[i]+'_she-ra','w')
        f.write('%r %r %r %i %i %i %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
                %(mangaid[i], mastarid[i], starid[i], \
                plate[i], ifudesign[i], mjd[i], \
                ra[i],dec[i], \
                parameters[i,0],parameters[i,1],parameters[i,2], \
                parameters[i,3],parameters[i,4],parameters[i,5], \
                parameters[i,6],parameters[i,7],parameters[i,8], \
                ppxf_info[i,0], ppxf_info[i,1], ppxf_info[i,2], \
                mags[i,0], mags[i,1], mags[i,2], mags[i,3], mags[i,4], \
                indices[i,0], indices[i,1], indices[i,2], indices[i,3], indices[i,4], \
                indices[i,5], indices[i,6], indices[i,7], indices[i,8], indices[i,9], \
                indices[i,10], indices[i,11], indices[i,12], indices[i,13], indices[i,14], \
                indices[i,15], indices[i,16], indices[i,17], indices[i,18], indices[i,19], \
                indices[i,20], indices[i,21], indices[i,22], indices[i,23], indices[i,24], \
                indices[i,25], indices[i,26], indices[i,27], indices[i,28], indices[i,29], \
                indices[i,30], indices[i,31], indices[i,32], indices[i,33], indices[i,34], \
                indices[i,35], indices[i,36], indices[i,37], indices[i,38], indices[i,39], \
                indices[i,40], indices[i,41]))
        f.close()
        print('     > spectrum '+str(i)+' done')
        continue         

    print('               > Teff  = '+str(parameters[i,0])+' K')
    print('               > logg  = '+str(parameters[i,1])+' dex')
    print('               > metal = '+str(parameters[i,2])+' dex') # not a real prior

    ######## calculating the second guess for the stellar parameters ########

    if (parameters[i,0] >= 12000): teff_guess = 2000
    if (parameters[i,0] < 12000): teff_guess = 1000 

    logg_guess = 0.80 # fixed value
    metal_guess = 2.00 # open metallicity 

    selected_parameters[i,0] = parameters[i,0]
    selected_parameters[i,1] = parameters[i,1]
    selected_parameters[i,2] = parameters[i,2]

    if (parameters[i,0] <= min(templ_teff)): selected_parameters[i,0] = min(templ_teff)+teff_guess
    if (parameters[i,0] >= max(templ_teff)): selected_parameters[i,0] = max(templ_teff)-teff_guess

    if ((parameters[i,0] >= 12000) & (parameters[i,1] <= 3.5)): selected_parameters[i,1] = 3.5
    if ((parameters[i,0] >= 8000) & (parameters[i,0] <= 12000) & (parameters[i,1] <= 2.0)): selected_parameters[i,1] = 2.5
    if ((parameters[i,0] >= 6000) & (parameters[i,0] <= 8000) & (parameters[i,1] <= 1.0)): selected_parameters[i,1] = 1.5
    if ((parameters[i,0] >= 2500) & (parameters[i,0] <= 6000) & (parameters[i,1] <= 0.0)): selected_parameters[i,1] = 0.0
    if ((parameters[i,0] >= 2500) & (parameters[i,0] <= 12000) & (parameters[i,1] >= 5.0)): selected_parameters[i,1] = 5.0

    if (parameters[i,2] <= min(templ_metal)): selected_parameters[i,2] = min(templ_metal)+metal_guess
    if (parameters[i,2] >= max(templ_metal)): selected_parameters[i,2] = max(templ_metal)-metal_guess       

    ok_set = np.where((np.abs(templ_teff-selected_parameters[i,0]) <= teff_guess) & (np.abs(templ_logg-selected_parameters[i,1]) <= logg_guess) & (np.abs(templ_metal-selected_parameters[i,2]) <= metal_guess))

    templ_set_id = templ_id[ok_set]
    templ_set_teff = templ_teff[ok_set]
    templ_set_logg = templ_logg[ok_set]
    templ_set_metal = templ_metal[ok_set]

    if not templ_set_id.size: 
        parameters[i,3] = -999
        parameters[i,4] = -999
        parameters[i,5] = -999
        parameters[i,6] = -999
        parameters[i,7] = -999
        parameters[i,8] = -999
        ##### making a preliminary output file #####
        print('          > making a preliminary output file')
        f = open(output_dir+'prelim_output/'+mastarid[i]+'_she-ra','w')
        f.write('%r %r %r %i %i %i %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
                %(mangaid[i], mastarid[i], starid[i], \
                plate[i], ifudesign[i], mjd[i], \
                ra[i],dec[i], \
                parameters[i,0],parameters[i,1],parameters[i,2], \
                parameters[i,3],parameters[i,4],parameters[i,5], \
                parameters[i,6],parameters[i,7],parameters[i,8], \
                ppxf_info[i,0], ppxf_info[i,1], ppxf_info[i,2], \
                mags[i,0], mags[i,1], mags[i,2], mags[i,3], mags[i,4], \
                indices[i,0], indices[i,1], indices[i,2], indices[i,3], indices[i,4], \
                indices[i,5], indices[i,6], indices[i,7], indices[i,8], indices[i,9], \
                indices[i,10], indices[i,11], indices[i,12], indices[i,13], indices[i,14], \
                indices[i,15], indices[i,16], indices[i,17], indices[i,18], indices[i,19], \
                indices[i,20], indices[i,21], indices[i,22], indices[i,23], indices[i,24], \
                indices[i,25], indices[i,26], indices[i,27], indices[i,28], indices[i,29], \
                indices[i,30], indices[i,31], indices[i,32], indices[i,33], indices[i,34], \
                indices[i,35], indices[i,36], indices[i,37], indices[i,38], indices[i,39], \
                indices[i,40], indices[i,41]))
        f.close()
        print('     > spectrum '+str(i)+' done')
        continue     

    ##### setting up the MaStar for pPXF #####

    velscale = c*cdelt1/max(new_wave)   # do not degrade original velocity sampling
    log_corrected_flux,log_new_wave,velscale = util.log_rebin([min(new_wave),max(new_wave)],corrected_flux,velscale=velscale)


    ##### setting up the selected templates #####

    templ_set = len(templ_set_id)
    templates = np.zeros((len(log_corrected_flux),templ_set))

    for j in range(0,templ_set):
      hdu = fits.open(templates_dir+'grid/'+templ_set_id[j]+'.fits')
      template_wave = np.linspace(start=hdu[0].header['CRVAL1'],stop=hdu[0].header['CRVAL2'],num=hdu[0].header['NAXIS1'])
      template_flux = np.interp(new_wave,template_wave,hdu[0].data)

      bad_flux = np.isnan(template_flux) | np.isinf(template_flux) | (template_flux <= 0.0)
      template_flux[bad_flux] = 0.0

      log_template_flux,log_template_wave,velscale = util.log_rebin([min(new_wave),max(new_wave)],template_flux,velscale=velscale)
      log_template_flux /= np.median(log_template_flux)

      templates[:,j] = log_template_flux

    ##### setting up the variables to run pPXF #####

    start = [0,10]

    noise = np.ones_like(log_corrected_flux)

    sol = ppxf(templates,log_corrected_flux,noise,velscale,start,lam=np.exp(log_new_wave),degree=10,moments=2,quiet=True)

    sol.plot()
    plt.title(mastarid[i])
    #plt.show()
    plt.savefig(output_dir+'fsf/'+mastarid[i]+'_bestfit.png') 
    plt.close() 

    ##### gathering the chi2, vel and sigma from the pPXF output #####

    print('          > gathering the information from the full-spectrum fitting')

    ppxf_info[i,0] = sol.chi2
    ppxf_info[i,1] = sol.sol[0]
    ppxf_info[i,2] = sol.sol[1]

    print('               > chi2/dof  = '+str(ppxf_info[i,0]))
    print('               > vel       = '+str(ppxf_info[i,1])+' km s-1')
    print('               > sigma     = '+str(ppxf_info[i,2])+' km s-1')

    prelim_teff = [ ]
    prelim_logg = [ ]
    prelim_metal = [ ]

    ppxf_weights = sol.weights[:]

    ##### calculating the weighted parameters #####

    for j in range(0,templ_set):
      prelim_teff.append(sol.weights[j]*templ_set_teff[j]/np.sum(sol.weights[:]))
      prelim_logg.append(sol.weights[j]*templ_set_logg[j]/np.sum(sol.weights[:]))
      prelim_metal.append(sol.weights[j]*templ_set_metal[j]/np.sum(sol.weights[:]))

    parameters[i,3] = np.sum(prelim_teff[:])
    parameters[i,4] = np.sum(prelim_logg[:])
    parameters[i,5] = np.sum(prelim_metal[:])+0.3

    bad_data = np.isnan(parameters[i,3]) | np.isinf(parameters[i,3]) | np.isnan(parameters[i,4]) | np.isinf(parameters[i,4]) | np.isnan(parameters[i,5]) | np.isinf(parameters[i,5])
    if bad_data: 
        parameters[i,3] = -999
        parameters[i,4] = -999
        parameters[i,5] = -999
        parameters[i,6] = -999
        parameters[i,7] = -999
        parameters[i,8] = -999
        ##### making a preliminary output file #####
        print('          > making a preliminary output file')
        f = open(output_dir+'prelim_output/'+mastarid[i]+'_she-ra','w')
        f.write('%r %r %r %i %i %i %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
                %(mangaid[i], mastarid[i], starid[i], \
                plate[i], ifudesign[i], mjd[i], \
                ra[i],dec[i], \
                parameters[i,0],parameters[i,1],parameters[i,2], \
                parameters[i,3],parameters[i,4],parameters[i,5], \
                parameters[i,6],parameters[i,7],parameters[i,8], \
                ppxf_info[i,0], ppxf_info[i,1], ppxf_info[i,2], \
                mags[i,0], mags[i,1], mags[i,2], mags[i,3], mags[i,4], \
                indices[i,0], indices[i,1], indices[i,2], indices[i,3], indices[i,4], \
                indices[i,5], indices[i,6], indices[i,7], indices[i,8], indices[i,9], \
                indices[i,10], indices[i,11], indices[i,12], indices[i,13], indices[i,14], \
                indices[i,15], indices[i,16], indices[i,17], indices[i,18], indices[i,19], \
                indices[i,20], indices[i,21], indices[i,22], indices[i,23], indices[i,24], \
                indices[i,25], indices[i,26], indices[i,27], indices[i,28], indices[i,29], \
                indices[i,30], indices[i,31], indices[i,32], indices[i,33], indices[i,34], \
                indices[i,35], indices[i,36], indices[i,37], indices[i,38], indices[i,39], \
                indices[i,40], indices[i,41]))
        f.close()
        print('     > spectrum '+str(i)+' done')
        continue     

    print('          > calculating the second set of teff, logg, and metallicity')  

    print('               > Teff  = '+str(parameters[i,3])+' K')
    print('               > logg  = '+str(parameters[i,4])+' dex')
    print('               > metal = '+str(parameters[i,5])+' dex')


    ##### gathering the parameters of the template with the largest weight #####

    print('          > calculating the third set of teff, logg, and metallicity')

    ok_heavy = np.where(ppxf_weights == max(ppxf_weights))

    parameters[i,6] = templ_set_teff[ok_heavy]
    parameters[i,7] = templ_set_logg[ok_heavy]
    parameters[i,8] = templ_set_metal[ok_heavy]+0.3

    print('               > Teff  = '+str(parameters[i,6])+' K')
    print('               > logg  = '+str(parameters[i,7])+' dex')
    print('               > metal = '+str(parameters[i,8])+' dex')


    print('          > elapsed time for one MaStar %.2f s' % (clock() - start_time))


    ##### making a preliminary output file #####

    print('          > making a preliminary output file')
    f = open(output_dir+'prelim_output/'+mastarid[i]+'_she-ra','w')
    f.write('%r %r %r %i %i %i %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
            %(mangaid[i], mastarid[i], starid[i], \
            plate[i], ifudesign[i], mjd[i], \
            ra[i],dec[i], \
            parameters[i,0],parameters[i,1],parameters[i,2], \
            parameters[i,3],parameters[i,4],parameters[i,5], \
            parameters[i,6],parameters[i,7],parameters[i,8], \
            ppxf_info[i,0], ppxf_info[i,1], ppxf_info[i,2], \
            mags[i,0], mags[i,1], mags[i,2], mags[i,3], mags[i,4], \
            indices[i,0], indices[i,1], indices[i,2], indices[i,3], indices[i,4], \
            indices[i,5], indices[i,6], indices[i,7], indices[i,8], indices[i,9], \
            indices[i,10], indices[i,11], indices[i,12], indices[i,13], indices[i,14], \
            indices[i,15], indices[i,16], indices[i,17], indices[i,18], indices[i,19], \
            indices[i,20], indices[i,21], indices[i,22], indices[i,23], indices[i,24], \
            indices[i,25], indices[i,26], indices[i,27], indices[i,28], indices[i,29], \
            indices[i,30], indices[i,31], indices[i,32], indices[i,33], indices[i,34], \
            indices[i,35], indices[i,36], indices[i,37], indices[i,38], indices[i,39], \
            indices[i,40], indices[i,41]))
    f.close()

    print('     > spectrum '+str(i)+' done')
    #sys.stdout.write("     > spectrum = %d %s \r" % (i+1,'done'))
    #sys.stdout.flush()

  print('     > all spectra done')
  print('     > elapsed time for all MaStar %.2f s' % (clock() - full_time))

#############################################################################################################################################
#############################################################################################################################################

prog = interface_sets(nstart,nend)

#############################################################################################################################################
#############################################################################################################################################

