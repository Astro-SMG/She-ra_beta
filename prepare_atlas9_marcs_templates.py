from os.path import join
from astropy.io import fits
from astropy.io import ascii

import os
import glob
import sys
import numpy as np
import pyfits
import astropy.cosmology as co
cosmo = co.Planck13
import matplotlib.pyplot as plt

from firefly_instrument import match_spectral_resolution 
#from downgrader_pPXF import downgrade

from calculate_magnitudes import *
from calculate_indices import *


##### directories section #####

print('     > setting up directories')

atlas9_dir = '/mnt/lustre/smg/stellar_libraries/bosz/'
marcs_dir = '/mnt/lustre/smg/stellar_libraries/marcs/original/'

output_dir = '/mnt/lustre/smg/stellar_libraries/atlas9marcs/'


##### setting up limits for wavelength #####

naxis1 = 15000
crval1 = 3600.0
crval2 = 10400.0
new_wave = np.linspace(start=crval1,stop=crval2,num=naxis1)
cdelt1 = (crval2-crval1)/naxis1


##### resolution to downgrade to #####

res = 2000 
sig2fwhm = np.sqrt(8.0*np.log(2.0))  
new_sres = np.full(len(new_wave),res)
new_fwhm = new_wave/new_sres
new_sigma = new_fwhm/sig2fwhm

atlas9_res = 10000
marcs_res = 20000


####### setting up some variables for magnitudes and indices calculation ########

filter_list = 'sdss_manga_pipeline.res'
zero_point = 'AB'
redshift = 0.0

indices_list = 'MaNGA_range.def'


##### getting info for atlas9 #####

print('     > looking for the ATLAS9 spectra')

atlas9_header = fits.open(atlas9_dir+'bosz_parameters_basic.fits')
#print(atlas9_header[1].header)
atlas9_template, atlas9_teff, atlas9_logg, atlas9_metal = atlas9_header[1].data['NAME'], atlas9_header[1].data['TEFF'], atlas9_header[1].data['LOGG'], atlas9_header[1].data['METAL']
n_atlas9 = int(atlas9_header[1].header['naxis2'])

print('     > number of atlas9 templates = '+str(n_atlas9))


#### getting info for marcs #####

marcs_header = fits.open(marcs_dir+'std_marcs.list.fits')
marcs_template = marcs_header[1].data['col1']
n_marcs = int(marcs_header[1].header['NAXIS2'])

marcs_params = fits.open(marcs_dir+'marcs_info.fits')
#print(marcs_params[1].header)
marcs_id, marcs_teff, marcs_logg, marcs_metal = marcs_params[1].data['col1'], marcs_params[1].data['col2'], marcs_params[1].data['col3'], marcs_params[1].data['col4']

print('     > number of marcs templates = '+str(n_marcs))

total_templ = n_atlas9 + n_marcs

print('     > total number of templates = '+str(total_templ))

parameters = np.zeros((total_templ,3))
templates_id = [ ]

indices = np.zeros((total_templ,42))
mags = np.zeros((total_templ,5))

j = 0 

##########################
##### Atlas9 section #####
##########################

##### set-up for ATLAS9 original resolution #####

sres = np.zeros(new_wave.shape,dtype=np.float64)
sres += atlas9_res
fwhm = new_wave/sres
sigma = fwhm/sig2fwhm


##### looping over the whole set of templates #####

print('     > gathering atlas9 templates')

for i in range(0,n_atlas9):
    ##### setting up the parameters #####
    parameters[j,0] = atlas9_teff[i]
    parameters[j,1] = atlas9_logg[i]
    parameters[j,2] = atlas9_metal[i]

    fits_name = 'templ'+str(j+1).zfill(6)

    ##### reading the fits file #####
    hdu = fits.open(atlas9_dir+'spectra/'+atlas9_template[i]+'.fits')

    old_wave = np.linspace(start=hdu[0].header['CRVAL1'],stop=hdu[0].header['CRVAL2'],num=hdu[0].header['NAXIS1'])
    old_flux = hdu[0].data
    new_flux = np.interp(new_wave,old_wave,old_flux)

    # DAP-downgrader
    convol_flux, matched_sres, sigma_offset, new_mask = match_spectral_resolution(new_wave,new_flux,sres,new_wave,new_sres,min_sig_pix=0.0)

    ##### calculating indices within the MaNGA wavelength range #####
    indices[j,:], err_indices = calculate_indices(new_wave,convol_flux*1e-17,indices_list,fits_name,ef=0,rv=0,plot=False,sim=False)

    ######## calculating the magnitudes ########
    mags[j,:] = calculate_magnitudes(new_wave,convol_flux*1e-17,filter_list,zero_point,redshift)

    ##### setting up the new fits file #####
    hdu = fits.PrimaryHDU()
    hdu.data = convol_flux/np.median(convol_flux)
    hdu.header['CRVAL1'] = crval1
    hdu.header['CRVAL2'] = crval2
    hdu.header['CDELT1'] = cdelt1
    hdu.header['TEFF'] = parameters[j,0]
    hdu.header['LOGG'] = parameters[j,1]
    hdu.header['METAL'] = parameters[j,2]
    hdu.writeto(output_dir+'spectra/'+fits_name+'.fits',clobber=True)

    templates_id.append(fits_name)
    j = j + 1
 
    #print('     > spectrum '+str(i+1)+' done')
    sys.stdout.write("     > spectrum = %d %s \r" % (i+1,'done'))
    sys.stdout.flush()


#########################
##### MARCS section #####
#########################

wavelength_file = fits.open(marcs_dir+'wavelengths.vac.fits')
#print(wavelength_file[1].header)
original_pixels = int(wavelength_file[1].header['NAXIS2'])
old_wave = np.array(wavelength_file[1].data['col1'])

##### set-up for MARCS original resolution #####

sres = np.zeros(new_wave.shape,dtype=np.float64)
sres += marcs_res
fwhm = new_wave/sres
sigma = fwhm/sig2fwhm

##### looping over the MARCS templates #####

print('     > gathering marcs templates')

for i in range(0,n_marcs):
    ##### setting up the parameters #####
    parameters[j,0] = marcs_teff[i]
    parameters[j,1] = marcs_logg[i]
    parameters[j,2] = marcs_metal[i]

    fits_name = 'templ'+str(j+1).zfill(6)                
        
    ##### reading the ascii file #####
    old_flux = np.loadtxt(marcs_dir+marcs_template[i],unpack=True)
    new_flux = np.interp(new_wave,old_wave,old_flux)

    # DAP-downgrader
    convol_flux, matched_sres, sigma_offset, new_mask = match_spectral_resolution(new_wave,new_flux,sres,new_wave,new_sres,min_sig_pix=0.0) 

    ##### calculating indices within the MaNGA wavelength range #####
    indices[j,:], err_indices = calculate_indices(new_wave,convol_flux*1e-17,indices_list,fits_name,ef=0,rv=0,plot=False,sim=False)

    ######## calculating the magnitudes ########
    mags[j,:] = calculate_magnitudes(new_wave,convol_flux*1e-17,filter_list,zero_point,redshift)

    ##### setting up the new fits file #####
 
    hdu = fits.PrimaryHDU()
    hdu.data = convol_flux/np.median(convol_flux)
    hdu.header['CRVAL1'] = crval1
    hdu.header['CRVAL2'] = crval2
    hdu.header['CDELT1'] = cdelt1
    hdu.header['TEFF'] = parameters[j,0]
    hdu.header['LOGG'] = parameters[j,1]
    hdu.header['METAL'] = parameters[j,2]
    hdu.writeto(output_dir+'spectra/'+fits_name+'.fits',clobber=True)

    templates_id.append(fits_name)
    j = j + 1
 
    #print('     > spectrum '+str(i+1)+' done')
    sys.stdout.write("     > spectrum = %d %s \r" % (i+1,'done'))
    sys.stdout.flush()


##### saving file name and corresponding parameters #####

print('     > creating a file with the file and parameters relation for all templates')

print('     > stellar parameters are temperature (teff), gravity (logg), metallicity (mh)')
print('                              sdss magitudes (ugriz)')
print('                              indices in the optical and NIR')

tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='NAME', array=np.array(templates_id[:]), format='20A'),
                                       fits.Column(name='TEFF', array=np.array(parameters[:,0]), format='D'),
                                       fits.Column(name='LOGG', array=np.array(parameters[:,1]), format='D'),
                                       fits.Column(name='METAL', array=np.array(parameters[:,2]), format='D'),
                                       fits.Column(name='Umag', array=np.array(mags[:,0]), format='D'),
                                       fits.Column(name='Gmag', array=np.array(mags[:,1]), format='D'),
                                       fits.Column(name='Rmag', array=np.array(mags[:,2]), format='D'),
                                       fits.Column(name='Imag', array=np.array(mags[:,3]), format='D'),
                                       fits.Column(name='Zmag', array=np.array(mags[:,4]), format='D'),
                                       fits.Column(name='OII3727', array=np.array(indices[:,0]), format='D'),
                                       fits.Column(name='HdA', array=np.array(indices[:,1]), format='D'),
                                       fits.Column(name='HdF', array=np.array(indices[:,2]), format='D'),
                                       fits.Column(name='CN1', array=np.array(indices[:,3]), format='D'),
                                       fits.Column(name='CN2', array=np.array(indices[:,4]), format='D'),
                                       fits.Column(name='Ca4227', array=np.array(indices[:,5]), format='D'),
                                       fits.Column(name='G4300', array=np.array(indices[:,6]), format='D'),
                                       fits.Column(name='HgA', array=np.array(indices[:,7]), format='D'),
                                       fits.Column(name='HgF', array=np.array(indices[:,8]), format='D'),
                                       fits.Column(name='Fe4383', array=np.array(indices[:,9]), format='D'),
                                       fits.Column(name='Ca4455', array=np.array(indices[:,10]), format='D'),
                                       fits.Column(name='Fe4531', array=np.array(indices[:,11]), format='D'),
                                       fits.Column(name='Fe4668', array=np.array(indices[:,12]), format='D'),
                                       fits.Column(name='Hbeta', array=np.array(indices[:,13]), format='D'),
                                       fits.Column(name='Hbetap', array=np.array(indices[:,14]), format='D'),
                                       fits.Column(name='OIII1', array=np.array(indices[:,15]), format='D'),
                                       fits.Column(name='OIII2', array=np.array(indices[:,16]), format='D'),
                                       fits.Column(name='Fe5015', array=np.array(indices[:,17]), format='D'),
                                       fits.Column(name='Mg1', array=np.array(indices[:,18]), format='D'),
                                       fits.Column(name='Mg2', array=np.array(indices[:,19]), format='D'),
                                       fits.Column(name='Mgb5177', array=np.array(indices[:,20]), format='D'),
                                       fits.Column(name='Fe5270', array=np.array(indices[:,21]), format='D'),
                                       fits.Column(name='Fe5335', array=np.array(indices[:,22]), format='D'),
                                       fits.Column(name='Fe5406', array=np.array(indices[:,23]), format='D'),
                                       fits.Column(name='Fe5709', array=np.array(indices[:,24]), format='D'),
                                       fits.Column(name='Fe5782', array=np.array(indices[:,25]), format='D'),
                                       fits.Column(name='NaD', array=np.array(indices[:,26]), format='D'),
                                       fits.Column(name='TiO1', array=np.array(indices[:,27]), format='D'),
                                       fits.Column(name='TiO2', array=np.array(indices[:,28]), format='D'),
                                       fits.Column(name='NaILaB', array=np.array(indices[:,29]), format='D'),
                                       fits.Column(name='NaISp', array=np.array(indices[:,30]), format='D'),
                                       fits.Column(name='NaICo', array=np.array(indices[:,31]), format='D'),
                                       fits.Column(name='Ca1', array=np.array(indices[:,32]), format='D'),
                                       fits.Column(name='Ca2', array=np.array(indices[:,33]), format='D'),
                                       fits.Column(name='Ca3', array=np.array(indices[:,34]), format='D'),
                                       fits.Column(name='MgI', array=np.array(indices[:,35]), format='D'),
                                       fits.Column(name='Ca1AZ', array=np.array(indices[:,36]), format='D'),
                                       fits.Column(name='Ca2AZ', array=np.array(indices[:,37]), format='D'),
                                       fits.Column(name='Ca3AZ', array=np.array(indices[:,38]), format='D'),
                                       fits.Column(name='CaT', array=np.array(indices[:,39]), format='D'),
                                       fits.Column(name='PaT', array=np.array(indices[:,40]), format='D'),
                                       fits.Column(name='sCaT', array=np.array(indices[:,41]), format='D')
                                     ])

tbhdu.writeto(output_dir+'templates_parameters_R2000.fits')

print('     > all spectra done')







