from glob import glob
from os.path import join
from concurrent.futures import ProcessPoolExecutor
from prprocessing import get_features, fourier_transform,generate_binmask
from pprint import pprint
from osgeo import gdal 
from upaths_wx import Tpath
gdal.UseExceptions()

from demnoise import dem_instrument_noise_processing # move to preprocessing 
# clean the tandemx []
# genate binary mask by clean tdemx 
# dem derivatives
# fourier and wavelt transforms 

# add if file already exist, do not re-create unless overwrite set to T

tdem_files = sorted(glob(join(Tpath, '**', '*tdem_DEM.tif'), recursive=True))
com_files = sorted(glob(join(Tpath, '**', '*tdem_COM.tif'), recursive=True))
wbm_files = sorted(glob(join(Tpath, '**', '*cdem_WBM.tif'), recursive=True))
hem_files = sorted(glob(join(Tpath, '**', '*tdem_HEM.tif'), recursive=True))

assert len(tdem_files) == len(com_files) == len(wbm_files)
for tdem,com,wbm,hem in zip(tdem_files,com_files,wbm_files,hem_files):
  pprint([tdem,hem,wbm,com])
  print('')


def generate_ml_features(tdem,hem,wbm,com):

  dem_maskede = dem_instrument_noise_processing(dem_path=tdem,err_path=hem,wam_path=wbm,com_path=com) 
  generate_binmask(dem_maskede) # thresholds already set 
  get_features(dem_maskede) # use othe dem like edem for this 
#   #fourier_transform(efile)

if __name__ == '__main__':
  with ProcessPoolExecutor() as PPE:
      for tdem,com,wbm,hem in zip(tdem_files,com_files,wbm_files,hem_files):
        PPE.submit(generate_ml_features,tdem,hem,wbm,com)
        #generate_ml_features(tdem,com,wbm,hem)