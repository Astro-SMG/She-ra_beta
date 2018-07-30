import numpy as np
import warnings
import math
import os
import scipy.interpolate as interpolate
from astropy.io import fits

#from firefly_fitter import *
#from firefly_library import *
#from firefly_instrument import *

# This should be used to 
def dust_allen_py(ebv,lam):

	# Calculates the attenuation for the Milky Way (MW) as found in Allen (1976).
	from scipy.interpolate import interp1d


	wave = [1000,1110,1250,1430,1670,2000,2220,2500,2860,3330,3650,4000,4400,5000,5530,6700,9000,10000,20000,100000]
	allen_k = [4.20,3.70,3.30,3.00,2.70,2.80,2.90,2.30,1.97,1.69,1.58,1.45,1.32,1.13,1.00,0.74,0.46,0.38,0.11,0.00]
	allen_k = np.array(allen_k)*3.1

	total = interp1d(wave, allen_k, kind='cubic')
	wavelength_vector = np.arange(1000,10000,100)
	fitted_function = total(wavelength_vector)


	def find_nearest(array,value,output):
		idx = (np.abs(np.array(array)-np.array(value))).argmin()
		return output[idx]

	output = []
	for l in range(len(lam)):
		k = find_nearest(wavelength_vector,lam[l],fitted_function)
		output.append(10**(-0.4*ebv*k))
	return output

def get_SFD_dust(long,lat,dustmap='ebv',interpolate=True):
    """
    Gets map values from Schlegel, Finkbeiner, and Davis 1998 extinction maps.
    
    `dustmap` can either be a filename (if '%s' appears in the string, it will be
    replaced with 'ngp' or 'sgp'), or one of:
    
    * 'i100' 
        100-micron map in MJy/Sr
    * 'x'
        X-map, temperature-correction factor
    * 't'
        Temperature map in degrees Kelvin for n=2 emissivity
    * 'ebv'
        E(B-V) in magnitudes
    * 'mask'
        Mask values 
        
    For these forms, the files are assumed to lie in the current directory.
    
    Input coordinates are in degrees of galactic latiude and logitude - they can
    be scalars or arrays.
    
    if `interpolate` is an integer, it can be used to specify the order of the
    interpolating polynomial
    
    .. todo::
        Check mask for SMC/LMC/M31, E(B-V)=0.075 mag for the LMC, 0.037 mag for
        the SMC, and 0.062 for M31. Also auto-download dust maps. Also add
        tests. Also allow for other bands.
    
    """
    from numpy import sin,cos,round,isscalar,array,ndarray,ones_like
    #from pyfits import open
    
    if type(dustmap) is not str:
        raise ValueError('dustmap is not a string')
    dml=dustmap.lower()
    if dml == 'ebv' or dml == 'eb-v' or dml == 'e(b-v)' :
        dustmapfn=os.environ['STELLARPOPMODELS_DIR']+'/data/SFD_dust_4096_%s.fits'
    elif dml == 'i100':
        dustmapfn=os.environ['STELLARPOPMODELS_DIR']+'/data/SFD_i100_4096_%s.fits'
    elif dml == 'x':
        dustmapfn=os.environ['STELLARPOPMODELS_DIR']+'/data/SFD_xmap_%s.fits'
    elif dml == 't':
        dustmapfn=os.environ['STELLARPOPMODELS_DIR']+'/data/SFD_temp_%s.fits'
    elif dml == 'mask':
        dustmapfn=os.environ['STELLARPOPMODELS_DIR']+'/data/SFD_mask_4096_%s.fits'
    else:
        dustmapfn=dustmap
    
    if isscalar(long):
        l=array([long])*math.pi/180
    else:
        l=array(long)*math.pi/180
    if isscalar(lat):
        b=array([lat])*math.pi/180
    else:
        b=array(lat)*math.pi/180
        
    if not len(l)==len(b):
        raise ValueError('input coordinate arrays are of different length')
    
    
    
    if '%s' not in dustmapfn:
        f=fits.open(dustmapfn)
        try:
            mapds=[f[0].data]
        finally:
            f.close()
        assert mapds[-1].shape[0] == mapds[-1].shape[1],'map dimensions not equal - incorrect map file?'
        
        polename=dustmapfn.split('.')[0].split('_')[-1].lower()
        if polename=='ngp':
            n=[1]
            if sum(b > 0) > 0:
                b=b
        elif polename=='sgp':
            n=[-1]
            if sum(b < 0) > 0:
                b=b
        else:
            raise ValueError("couldn't determine South/North from filename - should have 'sgp' or 'ngp in it somewhere")
        masks = [ones_like(b).astype(bool)]
    else: #need to do things seperately for north and south files
        nmask = b >= 0
        smask = ~nmask
        
        masks = [nmask,smask]
        ns = [1,-1]
        
        mapds=[]
        f=fits.open(dustmapfn%'ngp')
        try:
            mapds.append(f[0].data)
        finally:
            f.close()
        assert mapds[-1].shape[0] == mapds[-1].shape[1],'map dimensions not equal - incorrect map file?'
        f=fits.open(dustmapfn%'sgp')
        try:
            mapds.append(f[0].data)
        finally:
            f.close()
        assert mapds[-1].shape[0] == mapds[-1].shape[1],'map dimensions not equal - incorrect map file?'
    
    retvals=[]
    for n,mapd,m in zip(ns,mapds,masks):
        #project from galactic longitude/latitude to lambert pixels (see SFD98)
        npix=mapd.shape[0]
        
        x=npix/2*cos(l[m])*(1-n*sin(b[m]))**0.5+npix/2-0.5
        y=-npix/2*n*sin(l[m])*(1-n*sin(b[m]))**0.5+npix/2-0.5
        #now remap indecies - numpy arrays have y and x convention switched from SFD98 appendix
        x,y=y,x
        
        if interpolate:
            from scipy.ndimage import map_coordinates
            if type(interpolate) is int:
                retvals.append(map_coordinates(mapd,[x,y],order=interpolate))
            else:
                retvals.append(map_coordinates(mapd,[x,y]))
        else:
            x=round(x).astype(int)
            y=round(y).astype(int)
            retvals.append(mapd[x,y])
            
            
    
        
    if isscalar(long) or isscalar(lat):
        for r in retvals:
            if len(r)>0:
                return r[0]
        assert False,'None of the return value arrays were populated - incorrect inputs?'
    else:
        #now recombine the possibly two arrays from above into one that looks like  the original
        retval=ndarray(l.shape)
        for m,val in zip(masks,retvals):
            retval[m] = val
        return retval
        
    
def eq2gal(ra,dec):
	"""
	Convert Equatorial coordinates to Galactic Coordinates in the epch J2000.

	Keywords arguments:
	ra  -- Right Ascension (in radians)
	dec -- Declination (in radians)

	Return a tuple (l, b):
	l -- Galactic longitude (in radians)
	b -- Galactic latitude (in radians)
	"""
	# RA(radians),Dec(radians),distance(kpc) of Galactic center in J2000
	Galactic_Center_Equatorial=(math.radians(266.40510), math.radians(-28.936175), 8.33)

	# RA(radians),Dec(radians) of Galactic Northpole in J2000
	Galactic_Northpole_Equatorial=(math.radians(192.859508), math.radians(27.128336))

	alpha = Galactic_Northpole_Equatorial[0]
	delta = Galactic_Northpole_Equatorial[1]
	la = math.radians(33.0)

	b = math.asin(math.sin(dec) * math.sin(delta) +
					math.cos(dec) * math.cos(delta) * math.cos(ra - alpha))

	l = math.atan2(math.sin(dec) * math.cos(delta) - 
					math.cos(dec) * math.sin(delta) * math.cos(ra - alpha), 
					math.cos(dec) * math.sin(ra - alpha)
					) + la

	l = l if l >= 0 else (l + math.pi * 2.0)

	l = l % (2.0 * math.pi)


	return l*180.0/math.pi, b*180.0/math.pi

def get_dust_radec(ra,dec,dustmap,interpolate=True):
	"""
	Gets the value of dust from MW at ra and dec.
	"""
	#from .coords import equatorial_to_galactic
	l,b = eq2gal(math.radians(ra),math.radians(dec))
	return get_SFD_dust(l,b,dustmap,interpolate)

def unred(wave, ebv, R_V=3.1, LMC2=False, AVGLMC=False):
	"""
	 Deredden a flux vector using the Fitzpatrick (1999) parameterization
 
	 Parameters
	 ----------
	 wave :   array
			  Wavelength in Angstrom
	 flux :   array
			  Calibrated flux vector, same number of elements as wave.
	 ebv  :   float, optional
			  Color excess E(B-V). If a negative ebv is supplied,
			  then fluxes will be reddened rather than dereddened.
			  The default is 3.1.
	 AVGLMC : boolean
			  If True, then the default fit parameters c1,c2,c3,c4,gamma,x0 
			  are set to the average values determined for reddening in the 
			  general Large Magellanic Cloud (LMC) field by
			  Misselt et al. (1999, ApJ, 515, 128). The default is
			  False.
	 LMC2 :   boolean
			  If True, the fit parameters are set to the values determined
			  for the LMC2 field (including 30 Dor) by Misselt et al.
			  Note that neither `AVGLMC` nor `LMC2` will alter the default value 
			  of R_V, which is poorly known for the LMC.
   
	 Returns
	 -------             
	 new_flux : array 
				Dereddened flux vector, same units and number of elements
				as input flux.
 
	 Notes
	 -----

	 .. note:: This function was ported from the IDL Astronomy User's Library.

	 :IDL - Documentation:
 
	  PURPOSE:
	   Deredden a flux vector using the Fitzpatrick (1999) parameterization
	  EXPLANATION:
	   The R-dependent Galactic extinction curve is that of Fitzpatrick & Massa 
	   (Fitzpatrick, 1999, PASP, 111, 63; astro-ph/9809387 ).    
	   Parameterization is valid from the IR to the far-UV (3.5 microns to 0.1 
	   microns).    UV extinction curve is extrapolated down to 912 Angstroms.

	  CALLING SEQUENCE:
		FM_UNRED, wave, flux, ebv, [ funred, R_V = , /LMC2, /AVGLMC, ExtCurve= 
						  gamma =, x0=, c1=, c2=, c3=, c4= ]
	  INPUT:
		 WAVE - wavelength vector (Angstroms)
		 FLUX - calibrated flux vector, same number of elements as WAVE
				  If only 3 parameters are supplied, then this vector will
				  updated on output to contain the dereddened flux.
		 EBV  - color excess E(B-V), scalar.  If a negative EBV is supplied,
				  then fluxes will be reddened rather than dereddened.

	  OUTPUT:
		 FUNRED - unreddened flux vector, same units and number of elements
				  as FLUX

	  OPTIONAL INPUT KEYWORDS
		  R_V - scalar specifying the ratio of total to selective extinction
				   R(V) = A(V) / E(B - V).    If not specified, then R = 3.1
				   Extreme values of R(V) range from 2.3 to 5.3

	   /AVGLMC - if set, then the default fit parameters c1,c2,c3,c4,gamma,x0 
				 are set to the average values determined for reddening in the 
				 general Large Magellanic Cloud (LMC) field by Misselt et al. 
				 (1999, ApJ, 515, 128)
		/LMC2 - if set, then the fit parameters are set to the values determined
				 for the LMC2 field (including 30 Dor) by Misselt et al.
				 Note that neither /AVGLMC or /LMC2 will alter the default value 
				 of R_V which is poorly known for the LMC. 
			
		 The following five input keyword parameters allow the user to customize
		 the adopted extinction curve.    For example, see Clayton et al. (2003,
		 ApJ, 588, 871) for examples of these parameters in different interstellar
		 environments.

		 x0 - Centroid of 2200 A bump in microns (default = 4.596)
		 gamma - Width of 2200 A bump in microns (default  =0.99)
		 c3 - Strength of the 2200 A bump (default = 3.23)
		 c4 - FUV curvature (default = 0.41)
		 c2 - Slope of the linear UV extinction component 
			  (default = -0.824 + 4.717/R)
		 c1 - Intercept of the linear UV extinction component 
			  (default = 2.030 - 3.007*c2
	"""

	x = 10000./ wave # Convert to inverse microns 
	curve = x*0.

	# Set some standard values:
	x0 = 4.596
	gamma =  0.99
	c3 =  3.23      
	c4 =  0.41    
	c2 = -0.824 + 4.717/R_V
	c1 =  2.030 - 3.007*c2

	if LMC2:
		x0    =  4.626
		gamma =  1.05   
		c4   =  0.42   
		c3    =  1.92      
		c2    = 1.31
		c1    =  -2.16
	elif AVGLMC:   
		x0 = 4.596  
		gamma = 0.91
		c4   =  0.64  
		c3    =  2.73      
		c2    = 1.11
		c1    =  -1.28

	# Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and 
	# R-dependent coefficients
	xcutuv = np.array([10000.0/2700.0])
	xspluv = 10000.0/np.array([2700.0,2600.0])

	iuv = np.where(x >= xcutuv)[0]
	N_UV = len(iuv)
	iopir = np.where(x < xcutuv)[0]
	Nopir = len(iopir)
	if (N_UV > 0): xuv = np.concatenate((xspluv,x[iuv]))
	else:  xuv = xspluv

	yuv = c1  + c2*xuv
	yuv = yuv + c3*xuv**2/((xuv**2-x0**2)**2 +(xuv*gamma)**2)
	yuv = yuv + c4*(0.5392*(np.maximum(xuv,5.9)-5.9)**2+0.05644*(np.maximum(xuv,5.9)-5.9)**3)
	yuv = yuv + R_V
	yspluv  = yuv[0:2]  # save spline points
 
	if (N_UV > 0): curve[iuv] = yuv[2::] # remove spline points

	# Compute optical portion of A(lambda)/E(B-V) curve
	# using cubic spline anchored in UV, optical, and IR
	xsplopir = np.concatenate(([0],10000.0/np.array([26500.0,12200.0,6000.0,5470.0,4670.0,4110.0])))
	ysplir   = np.array([0.0,0.26469,0.82925])*R_V/3.1 
	ysplop   = np.array((np.polyval([-4.22809e-01, 1.00270, 2.13572e-04][::-1],R_V ), 
			np.polyval([-5.13540e-02, 1.00216, -7.35778e-05][::-1],R_V ), 
			np.polyval([ 7.00127e-01, 1.00184, -3.32598e-05][::-1],R_V ), 
			np.polyval([ 1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05][::-1],R_V ) ))
	ysplopir = np.concatenate((ysplir,ysplop))

	if (Nopir > 0): 
	  tck = interpolate.splrep(np.concatenate((xsplopir,xspluv)),np.concatenate((ysplopir,yspluv)),s=0)
	  curve[iopir] = interpolate.splev(x[iopir], tck)

	#Now apply extinction correction to input flux vector
	curve *= ebv

	return 10.**(0.4*curve)

