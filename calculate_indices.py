'''
#############################################################################################################################################
#############################################################################################################################################

This program calculates indices for spectra at any given wavelenght provided with a table with indices definitions.

Version 1.0 
by Y.P. Chen 
IDL program for calculating indices, specifically CaT

Version 1.5
by S. Meneses-Goytia
Kapteyn Instituut, Groningen, NL
Adapted to calculate diferent indices in the NIR for one spectrum within a major routine for several spectra 

Version 2.0
by S. Meneses-Goytia
Institute of Cosmology and Gravitation - University of Portsmouth, Portsmouth, UK
Adapted to calculate NIR indices for any spectra in fits format 

Version 3.0
by S. Meneses-Goytia
Institute of Cosmology and Gravitation - University of Portsmouth, Portsmouth, UK
Adapted to calculate any indices of any spectra in fits format 

Version 4.0
by S.Meneses-Goytia
Institute of Cosmology and Gravitation - University of Portsmouth, Portsmouth, UK
Adapted from IDL to python


input:  w         - wavelength in angstrom
        f         - flux or counts
        def       - definition of the indices, ew
        ef        - error of the flux
        n_index   - indices calculated, n = m + 3, m: number of index in the input list
        /slow     - show slowly the plot in the screen
        obj       - obj, give the detail description on the plot

output: index_out - calculated index
        err_index - error for the correspondant index

The following routines and functions are used within the program:
          continuum (calculates the continuum)
          def_continue (defines the continuum line)

#############################################################################################################################################
#############################################################################################################################################

'''




#############################################################################################################################################
#############################################################################################################################################

# importing packages and libraries

import pandas as pd
import numpy as np

from scipy.integrate import trapz

#############################################################################################################################################
#############################################################################################################################################

##### function continuum,sb,sr,lamr,lamb,x ##### 
##### subroutine for calculating the continuum in a spectra, based on Cardiel et al., 1998,AASS,127,597 ##### 

def continuum(sb,sr,lamr,lamb,x):
	y = sb*(lamr-x)/(lamr-lamb)+sr*(x-lamb)/(lamr-lamb)
	return y

#############################################################################################################################################
#############################################################################################################################################



#############################################################################################################################################
#############################################################################################################################################

##### subroutine for defining the continuum of the spectra using the function continuum ##### 
##### the corresponding error (ef) for the input spectra (f) is calculated here with ef=err=sqrt(variance) ##### 

def def_continuum(w,f,b1,b2,r1,r2,c1,c2,ef):

	#print(b1,b2,c1,c2,r1,r2)

	#sb_ok = np.where((w >= b1) & (w<= b2))                       
	#sb = trapz(f[sb_ok],x=w[sb_ok]) / (b2-b1)
           
	#sr_ok = np.where((w >= r1) & (w<= r2))
	#sr = trapz(f[sr_ok],x=w[sr_ok]) / (r2-r1)
        
	wavb = np.linspace(start=b1,stop=b2,num=100)
	fb = np.interp(wavb, w,f)        
	sb = trapz(fb,wavb) / (b2-b1)

	wavr = np.linspace(start=r1,stop=r2,num=100)
	fr = np.interp(wavr, w,f)        
	sr = trapz(fr,wavr) / (r2-r1)

	lamb = (b2+b1)/2.0
	lamr = (r2+r1)/2.0
        
	ok = np.where((w >= b1) & (w<= r2))
	#cont = sb*(lamr-w)/(lamr-lamb)+sr*(w-lamb)/(lamr-lamb)
	cont = continuum(sb,sr,lamr,lamb,w[ok])
	intf = 1-f[ok]/cont

	ew_ok = np.where((w[ok] >= c1) & (w[ok] <= c2))
	#ew = trapz(intf[ew_ok],w[ew_ok])
        
	wavc = np.linspace(start=c1,stop=c2,num=100)
	fintc = np.interp(wavc, w[ok],intf)        
	ew = trapz(fintc,wavc)


	if len(ef) != 0:

		fc = np.interp(wavc, w,f)        
		sc = trapz(fc,wavc) / (r2-r1)

		#sc_ok = np.where((w >= c1) & (w <= c2))
	 	#sc = trapz(f[sc_ok],w[sc_ok])
		lamc = (c1+c2)/2
		cc = continuum(sb,sr,lamr,lamb,lamc)
                
		fsigb = np.interp(wavb, w,(f**2)/(ef**2))        
		fsigr = np.interp(wavr, w,(f**2)/(ef**2))        
		fsigc = np.interp(wavc, w,(f**2)/(ef**2))        

		sigsc = 1/(trapz(fsigc,wavc))
		sigsb = (sb**2)/(trapz(fsigb,wavb))
		sigsr = (sr**2)/(trapz(fsigr,wavr))

	 	#sigsc = 1/(trapz((f[sc_ok]**2)/(ef[sc_ok]**2),w[sc_ok]))
	 	#sigsb = (sb**2)/(trapz((f[sb_ok]**2)/(ef[sb_ok]**2),w[sb_ok]))
	 	#sigsr = (sr**2)/(trapz((f[sr_ok]**2)/(ef[sr_ok]**2),w[sr_ok]))
		isig = sc/cc*np.sqrt(sigsc+sigsb*((lamr-lamc)/(lamr-lamb))**2/cc**2+sigsr*((lamb-lamc)/(lamr-lamb))**2/cc**2)

	return ew,sb,sr,isig,cont

#############################################################################################################################################
#############################################################################################################################################



#############################################################################################################################################
#############################################################################################################################################

##### this routine calculates the CaT index

def CaT_index(w,f):

    ##### definitions for the bandpasses #####
    mc11,mc12=8474.000,8484.000
    mc21,mc22=8563.000,8577.000
    mc31,mc32=8619.000,8642.000
    mc41,mc42=8700.000,8725.000
    mc51,mc52=8776.000,8792.000
                        
    tc11,tc12=8484.000,8513.000
    tc21,tc22=8522.000,8562.000
    tc31,tc32=8642.000,8682.000
                                    
    ##### to know where these definitions are in the spectra since they may not be the exactly same wavelength #####
    s1=np.where((w >= mc11) & (w <= mc12))
    s2=np.where((w >= mc21) & (w <= mc22))
    s3=np.where((w >= mc31) & (w <= mc32))
    s4=np.where((w >= mc41) & (w <= mc42))
    s5=np.where((w >= mc51) & (w <= mc52))
    
    mw = np.concatenate((w[s1],w[s2],w[s3],w[s4],w[s5]))
    mf = np.concatenate((f[s1],f[s2],f[s3],f[s4],f[s5]))

    resu= np.polyfit(mw,mf,1)
    pol = np.poly1d(resu)
    fmf=pol(mw)
    cont=pol(w)
    intf = 1.0-(f/cont)
    
    wavt1 = np.linspace(start=tc11,stop=tc12,num=100)
    ft1 = np.interp(wavt1, w,intf)        
    wavt2 = np.linspace(start=tc21,stop=tc22,num=100)
    ft2 = np.interp(wavt2, w,intf)        
    wavt3 = np.linspace(start=tc31,stop=tc32,num=100)
    ft3 = np.interp(wavt3, w,intf)        

    cat = trapz(ft1,wavt1)+trapz(ft2,wavt2)+trapz(ft3,wavt3)
 
    return cat
    
#############################################################################################################################################
#############################################################################################################################################




#############################################################################################################################################
#############################################################################################################################################

##### this routine calculates the PaT index

def PaT_index(w,f):
    
    ##### definitions for the bandpasses #####
    
    mc11,mc12=8474.000,8484.000
    mc21,mc22=8563.000,8577.000
    mc31,mc32=8619.000,8642.000
    mc41,mc42=8700.000,8725.000
    mc51,mc52=8776.000,8792.000

    tp11,tp12=8461.000,8474.000
    tp21,tp22=8577.000,8619.000
    tp31,tp32=8730.000,8772.000
    
    ##### to know where these definitions are in the spectra since they may not be the exactly same wavelength #####
    s1=np.where((w >= mc11) & (w <= mc12))
    s2=np.where((w >= mc21) & (w <= mc22))
    s3=np.where((w >= mc31) & (w <= mc32))
    s4=np.where((w >= mc41) & (w <= mc42))
    s5=np.where((w >= mc51) & (w <= mc52))
    
    mw = np.concatenate((w[s1],w[s2],w[s3],w[s4],w[s5]))
    mf = np.concatenate((f[s1],f[s2],f[s3],f[s4],f[s5]))

    resu= np.polyfit(mw,mf,1)
    pol = np.poly1d(resu)
    fmf=pol(mw)
    cont=pol(w)
    intf = 1.0-(f/cont)
    
    wavt1 = np.linspace(start=tp11,stop=tp12,num=100)
    ft1 = np.interp(wavt1, w,intf)        
    wavt2 = np.linspace(start=tp21,stop=tp22,num=100)
    ft2 = np.interp(wavt2, w,intf)        
    wavt3 = np.linspace(start=tp31,stop=tp32,num=100)
    ft3 = np.interp(wavt3, w,intf)        

    pat = trapz(ft1,wavt1)+trapz(ft2,wavt2)+trapz(ft3,wavt3)
    
    return pat

#############################################################################################################################################
#############################################################################################################################################




#############################################################################################################################################
#############################################################################################################################################

##### this routine calculates the sCaT index from the CaT and PaT

def sCaT_index(w,f,cat,pat):
    
    scat = cat - (0.93*pat)
    
    return scat

#############################################################################################################################################
#############################################################################################################################################




#############################################################################################################################################
#############################################################################################################################################

##### this routine calculates the D_CO as defined by Marmol-Queralto et al #####

def CO_index(w, f):

    ##### definitions for the continuum/blue bandpasses #####
    mc11,mc12=22460.00,22550.00
    mc21,mc22=22710.00,22770.00

    ##### definition for the absorption/central/green bandpass #####
    tc11,tc12=22880.00,23010.00

    ##### to know where these definitions are in the spectra since they may not be the exactly same wavelength #####
    s1,=np.where((w >= mc11) & (w <= mc12))
    s2,=np.where((w >= mc21) & (w <= mc22))
    s3,=np.where((w >= tc11) & (w <= tc12))
    
    mw = [w[s1],w[s2]]
    mf = [f[s1],f[s2]]
    
    #resu= np.polyfit(mw,mf,1)
    #pol = np.poly1d(resu)
    #fmf=pol(mw)
    #cont=pol(w)
    #intf = 1.0-(f/cont)
    
    wavt1 = np.linspace(start=mc11,stop=mc12,num=100)
    ft1 = np.interp(wavt1, w,f)        
    wavt2 = np.linspace(start=mc21,stop=mc22,num=100)
    ft2 = np.interp(wavt2, w,f)        
    wavt3 = np.linspace(start=tc11,stop=tc12,num=100)
    ft3 = np.interp(wavt3, w,f)        
    
    co_c = (trapz(ft1,wavt1)+trapz(ft2,wavt2))/((mc12-mc11)+(mc22-mc21))
    co_a = trapz(ft3,wavt3)/(tc12-tc11)

    co = co_c/co_a

    return co

#############################################################################################################################################
#############################################################################################################################################




#############################################################################################################################################
#############################################################################################################################################

# main routine for the indices calculation, index_cal.pro

def calculate_indices(w,f,ilist,obj,ef=0,rv=0,plot=False,sim=False):

	c = 299792.458                                                                                   # [km/s] speed of light

	datadir = '/mnt/lustre/smg/programs/'
	definitions_file = datadir+'definitions/'+ilist

	'''
	definitions_table = pd.read_table(definitions_file, usecols=[0,1,2,3,4,5,6,7], \
						names=['label', 'blue_1', 'blue_2', 'green_1', 'green_2', 'red_1', 'red_2', 'index_class'], \
						delim_whitespace=True, header=0)

	name = np.array(definitions_table['label'].values)
	index_type = np.array(definitions_table['index_class'].values)
	blue = np.array((definitions_table['blue_1'].values,(definitions_table['blue_2'].values)))
	green = np.array((definitions_table['green_1'].values,(definitions_table['green_2'].values)))
	red = np.array((definitions_table['red_1'].values,(definitions_table['red_2'].values)))

	n_index = np.shape(definitions_table)[0]
	'''

	blue_1, blue_2, green_1, green_2, red_1, red_2 = np.loadtxt(definitions_file,usecols=[1,2,3,4,5,6], unpack=True, dtype=None)
	name, index_type = np.genfromtxt(definitions_file,usecols=[0,7], unpack=True, dtype=(str,str))

	blue = np.array((blue_1,blue_2))
	green = np.array((green_1,green_2))
	red = np.array((red_1,red_2))

	n_index = len(name)

	##### radial velocity in km/s #####
  	#if len(rv) <= 0: rv =0

	frac = np.sqrt((1+rv/c)/(1-rv/c))
	w0=w/frac

	index_out = np.zeros(n_index)
	err_index = np.zeros(n_index)

	##### given a arbitrary definition of the error, modifiable #####
	#if (len(ef) <= 0): ef = f/50.0
	if (ef == 0): ef = f/50.0

	##### checking if it has elements for a Monte Carlo simulation #####
	if (sim == True):
		nsim = 100 # or more
		tmpew = np.zeros(nsim)

	for i in range(0,n_index):
        #print("index: ",i)
		index = 0.0
		ein = 0.0
		if ((blue[0,i] >= min(w0)) & (red[1,i] <= max(w0))):
			ew, sb, sr, isig, cont = def_continuum(w0,f,blue[0,i],blue[1,i],red[0,i],red[1,i],green[0,i],green[1,i],ef)
			#print('ew = ',ew)
			if (plot == True):
				import matplotlib.pyplot as plt
				ok_plot = np.where((w0 >= blue[0,i]) & (w0 <= red[1,i]))

				if len(obj) != 0:
					plt.title(obj)
				else: 
					plt.title(name[i])

				plt.xlabel(r"Wavelength [$\AA$]")
				plt.ylabel("Relative Flux")
				plt.xlim([blue[0,i],red[1,i]])
				plt.autoscale(enable=True, axis='both',tight=True)
				plt.plot(w0[ok_plot],f[ok_plot],color='black',linewidth=1)

				plt.axvspan(blue[0,i],blue[1,i], alpha=0.4, color='blue')
				plt.axvspan(green[0,i],green[1,i], alpha=0.4, color='green')
				plt.axvspan(red[0,i],red[1,i], alpha=0.4, color='red')

				plt.plot(w0[ok_plot],cont,color='gray',linewidth=1,label=name[i]+' = '+str(ew))
				plt.legend(loc='upper center')

				plt.show() 

			if (sim == True):
				for j in range(0,nsim):
					ran = np.random.normal(0,1,1)
					rf = f+ran*ef
					ew, sb, sr, isig, cont = def_continuum(w0,rf,blue[0,i],blue[1,i],red[0,i],red[1,i],green[0,i],green[1,i],ef)
					tmpew[j]=ew

		        ##### for normal atomic lines, calculate the standard deviation and mean #####
				ew = np.mean(tmpew)
				isig = np.std(tmpew)

			if (index_type[i] == 'atomic'):
				index = ew
				ein = isig
				#print('index = ', index)
			if (index_type[i] == 'molecular'):
				index = -2.5*np.log10(1.0-ew/(green[1,i]-green[0,i,]))
				ein = 2.5*10**(0.4*index)*isig/(2.3026*(green[1,i]-green[0,i]))
				#print('index = ', index)
			if (index_type[i] == 'ratio'): 
				index = -2.5*np.log10(ew/sb)
				ein = 2.5*10**(0.4*index)*isig/(2.3026*(green[1,i]-green[0,i]))
				#print('index = ', index)
			if (index_type[i] == 'special'): 
				if (name[i] == 'CaT'):
					CaT = CaT_index(w0,f)
					index = CaT
					ein = 0.0
				if (name[i] == 'PaT'):
					PaT = PaT_index(w0,f)
					index = PaT
					ein = 0.0
				if (name[i] == 'sCaT'):
					sCaT = sCaT_index(w0,f,CaT,PaT)
					index = sCaT
					ein = 0.0
				if (name[i] == 'DCO'):
					CO = CO_index(w0,f)
					index = CO
					ein = 0.0
				#print('index = ', index)

		index_out[i] = index 
		err_index[i] = ein 

	return index_out, err_index

#############################################################################################################################################
#############################################################################################################################################

