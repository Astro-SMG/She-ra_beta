import sys

import merol3 as m3

import time
import datetime

start_time = time.time()

templates_info = '/mnt/lustre/smg/stellar_libraries/atlas9marcs/templates_parameters_R2000_sdss_gaia.fits'
templates_dir = '/mnt/lustre/smg/stellar_libraries/atlas9marcs/spec/'

output_dir = '/mnt/lustre/smg/stellar_libraries/atlas9marcs/'

test = m3.merol3(templates_info,templates_dir,output_dir)
test.prepare_templates()

test.make_full_hrd_grid()

end_time = time.time()

print('     > total time = '+str((end_time-start_time)/60)+' min')
