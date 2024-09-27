import os 
import time 
from glob import glob
from concurrent.futures import ProcessPoolExecutor 
from prprocessing import get_raster_params,gdalwarp_clip
from utils import get_output_fpath,read_yaml
from pprint import pprint

#from preprocessing.upaths import tile_est_perfect
#from preprocessing.upaths import tiling_dpath, vrt_files, htdemx_files,htdemx_path, copdem_vgrid_tiles_dpath
# htdemx_files = glob(f'{htdemx_path}'); print(len(htdemx_files))
# tilenames_roi = os.listdir(copdem_vgrid_tiles_dpath); print(len(tilenames_roi))

# htdemx_filesb = [os.path.basename(i).split('_')[3] for i in htdemx_files]
# tilenames_roib = [os.path.basename(i).split('.')[0] for i in tilenames_roi]
# #rtiles = list(set(tilenames_roib) - set(htdemx_filesb))
# rtiles = tile_est_perfect
# #basefiles = [f for f in htdemx_files if os.path.basename(f).split('_')[3] not in rtiles]
# basefiles = [f for f in htdemx_files if os.path.basename(f).split('_')[3] in rtiles]
# print(f"Number of HTDEMx files after removing redundant tiles: {len(basefiles)}")

from upaths_wx import tilesnames_by_roi_dict, basefiles12_dpath,DATASET_KNAMES,step1_yaml_fpath, tiles12m_dpath



#print(tilesnames_by_roi_dict)

def filter_dict(data,VARNAMES):
    filtered_data = {key: data[key] for key in VARNAMES if key in data}
    return filtered_data


tilenames_roi = tilesnames_by_roi_dict['rio_negro']
print(tilenames_roi)

dem_tif_files = glob(os.path.join(basefiles12_dpath, '**', '*DEM.tif'), recursive=True)
print(len(dem_tif_files))

filtered_files = [
  file for file in dem_tif_files
  if any(substring in os.path.basename(file) for substring in tilenames_roi)
]

print(len(filtered_files))

basefiles = filtered_files
step1_data = read_yaml(step1_yaml_fpath)
step1_data_f =  filter_dict(step1_data, DATASET_KNAMES)
vrt_files = step1_data_fvrts = list(step1_data_f.values())




cpus = int(os.cpu_count() - 3)

if __name__ == '__main__':
    ti = time.perf_counter()
    #print(rtiles)
    #pprint(tile_est_perfect)
    #pprint(basefiles)
    with ProcessPoolExecutor(cpus) as PPE:
        for vrt in vrt_files:
            src_dataset = vrt
            print(vrt)
            for bfile in basefiles:
                dst_dataset = fo_fpath = get_output_fpath(tiles12m_dpath,bfile,vrt,"tdem")
                print(fo_fpath)
                xmin, ymin, xmax, ymax, xres, yres,epsg_code = get_raster_params(bfile)
                print(xmin, ymin, xmax, ymax, xres, yres,epsg_code)
                PPE.submit(
                    gdalwarp_clip,src_dataset, dst_dataset, xmin, ymin, xmax, ymax, xres, yres, epsg_code
                )
 
                #gdalwarp_clip(src_dataset, dst_dataset, xmin, ymin, xmax, ymax, xres, yres, epsg_code)
                #gdalwarp_clip(vrt, dst_dataset, xmin, ymin, xmax, ymax, xres, yres, epsg_code)
    tf = time.perf_counter() - ti
    print(f'run.time : {tf/60}  min(s)')