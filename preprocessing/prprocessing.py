import os 
import subprocess
from glob import glob 
import scipy.ndimage as ndimage
from osgeo import gdal
import numpy as np
import rasterio 
import pywt
from concurrent.futures import ProcessPoolExecutor

######################################################## preprocessing:::a:::gen_vrts.py
def list2txt(l,txt):
    with open(txt, 'w') as F:
        for li in l:
            F.write(li+'\n')
    print('list2txt')

def txt2vrt(txt):
    vrt = txt.replace('.txt','.vrt')
    cmd = f'gdalbuildvrt -input_file_list {txt} {vrt}'
    os.system(cmd)
    print('txt2vrt')

def list2vrt(l,txt):
    list2txt(l,txt)
    txt2vrt(txt)

def filter_x_isin_list(mylist, x):
    return [i for i in mylist if x in i]

def filter_x_notin_list(mylist, x):
     return [i for i in mylist if x not in i]

def get_x_from_list(mylist, x):
    return [i for i in mylist if x in i][0]

def gen_vrt_params(dir_VRTs, varnames, dname, allfiles):
    var_dict = dict()
    for i in range(len(varnames)):
        txt_path = os.path.join(dir_VRTs, f'{dname}_{varnames[i]}.txt')
        vrt_path = os.path.join(dir_VRTs, f'{dname}_{varnames[i]}.vrt')
        count_lists = sum(isinstance(item, list) for item in allfiles)
        if count_lists > 1:
            files = allfiles[i]
            var_dict[varnames[i]] = {
                'txt': txt_path, 'vrt': vrt_path, 'files': files}
        else:
            if allfiles:   
                var_files = filter_x_isin_list(
                    allfiles, f'{varnames[i]}.tif')
                var_dict[varnames[i]] = {
                    'txt': txt_path, 'vrt': vrt_path, 'files': var_files}
            else:
                files = 'NoPathProvided'
                var_dict[varnames[i]] = {
                    'txt': txt_path, 'vrt': vrt_path, 'files': files}
                print('Warning:', 'all files is empty')
    print('gen_vrt_params')
    return var_dict

def process_lfiles(ldar_files, ldar_wrapped_dpath, epsg, xres,cpu=50):
    # wrap_lidar_files

    ldar_wrapped_files = [os.path.join(ldar_wrapped_dpath,
                     os.path.basename(fi)) for fi in ldar_files]

    epsg_l = np.repeat(epsg, len(ldar_wrapped_files))
    res_l = np.repeat(xres, len(ldar_wrapped_files))

    with ProcessPoolExecutor(cpu) as ppe:
        ppe.map(warp_raster, ldar_files, ldar_wrapped_files, 
                res_l, epsg_l)
    return ldar_wrapped_files

def warp_raster(fi, fo, res, tepsg):
    if not os.path.exists(fo):
        cmd = f'gdalwarp -wo num_threads=all -co compress=deflate \
            -t_srs {tepsg} -tr {res} {res} {fi} {fo}'
        os.system(cmd)
    else:
        print('Already exisits')

def get_all_VRT_TXT_FILE_paths(yaml_data):
    VRT_paths, TXT_paths, FILE_paths = [], [], []
    for group_dict in list(yaml_data.keys()):
        for gdict_var in list(yaml_data[group_dict].keys()):
            vrt, txt, files = get_txt_vrt_files_by_var(
                yaml_data, group_dict, gdict_var)
            VRT_paths.append(vrt)
            TXT_paths.append(txt)
            FILE_paths.append(files)
    print('get_all_VRT_TXT_FILE_paths')
    return VRT_paths, TXT_paths, FILE_paths

def get_txt_vrt_files_by_var(yaml_data, gdict, gdict_var):
    vrt = yaml_data[gdict][gdict_var]['vrt']
    txt = yaml_data[gdict][gdict_var]['txt']
    files = yaml_data[gdict][gdict_var]['files']
    return vrt, txt, files

def buildVRT_from_list(txt_i, vrt_i, files):
    write_list_to_txt(txt_i, files) 
    buildVRT(txt_i, vrt_i)

def write_list_to_txt(txt, llist):
    with open(txt, 'w') as txt_write:
        for i in llist:
            txt_write.write(i+'\n')
    print('write_list_to_txt')


def buildVRT(txt, vrt):
    cmd = ['gdalbuildvrt', '-overwrite', '-input_file_list', txt, vrt]
    try:
        subprocess.run(cmd, check=True)
        print("VRT file created successfully at:", vrt)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
    except FileNotFoundError:
        print(f"Error: One or more files listed in '{txt}' do not exist.")
    except Exception as e:
        print("An unexpected error occurred:", e)


######################################################## preprocessing:::b:::gen_tiles.py
def gdalwarp_clip(src_dataset, dst_dataset, xmin, ymin, xmax, ymax, xres, yres, epsg_code, 
                  output_format='GTiff', creation_options=['COMPRESS=DEFLATE']):
    """
    Clips and reprojects a raster dataset using gdalwarp.

    Parameters:
    - src_dataset: Path to the source dataset.
    - dst_dataset: Path to the destination dataset.
    - xmin, ymin, xmax, ymax: Clipping extent coordinates.
    - xres, yres: Desired resolution.
    - epsg_code: EPSG code for the target coordinate system.
    - output_format: Format of the output file (default is 'GTiff').
    - creation_options: List of creation options (e.g., ['COMPRESS=DEFLATE']).

    Returns:
    - None
    """
    # Open the source dataset
    src_ds = gdal.Open(src_dataset)
    if not src_ds:
        raise RuntimeError(f"Failed to open source dataset: {src_dataset}")

    # Get the data type and number of bands
    dtype = gdal.GetDataTypeName(src_ds.GetRasterBand(1).DataType)
    num_bands = src_ds.RasterCount
    src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()

    print(f"Source dataset: {src_dataset}")
    print(f"Data type: {dtype}")
    print(f"Number of bands: {num_bands}")
    print(f"Source nodata value: {src_nodata}")

    # Determine the interpolation method
    if dtype == 'Byte':
        resampling_method = 'near'
    else:
        resampling_method = 'bilinear'

    # Build the gdalwarp command
    cmd = [
        'gdalwarp',
        '-te', str(xmin), str(ymin), str(xmax), str(ymax),
        '-tr', str(xres), str(yres),
        '-t_srs', f'EPSG:{epsg_code}',
        '-of', output_format,
        '-tap',
        '--config', 'GDAL_NUM_THREADS', 'ALL_CPUS',
        '-r', resampling_method,
    ]

    # Handle nodata values
    if dtype == 'Byte':
        if src_nodata is not None:
            cmd += ['-dstnodata', str(src_nodata)]
    else:
        if src_nodata is not None:
            cmd += ['-srcnodata', str(src_nodata), '-dstnodata', '-9999']

    # Add creation options if provided
    if creation_options:
        for option in creation_options:
            cmd += ['-co', option]

    # Add source and destination datasets to the command
    cmd += [src_dataset, dst_dataset]

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully processed {src_dataset} to {dst_dataset}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while processing {src_dataset}: {e}")

def get_raster_params(raster_path):
    """
    Get the bounding box and resolution of a raster file.
    
    Parameters:
    raster_path (str): Path to the raster file.
    
    Returns:
    tuple: Bounding box coordinates (xmin, ymin, xmax, ymax) and resolution (xres, yres).
    """
    with rasterio.open(raster_path) as ds:
        K = 0
        # Bounding box
        bbox = ds.bounds
        xmin, ymin, xmax, ymax = round(bbox.left, K), round(bbox.bottom, K), round(bbox.right, K), round(bbox.top, K)
        # Resolution
        xres, yres = ds.res
        epsg_code = ds.crs.to_epsg()
        
    return xmin, ymin, xmax, ymax, xres, yres, epsg_code

def get_tilename_from_edem_fpath(bfile):
    return os.path.basename(bfile).split('_')[3]

def get_output_fpath(tdpath,bfile,vrt):

    fo_dpath = os.path.join(tdpath,get_tilename_from_edem_fpath(bfile))
    os.makedirs(fo_dpath, exist_ok=True)
    tif = os.path.basename(vrt)
    tif = tif.replace('.vrt','.tif')
    fo_fpath = os.path.join(fo_dpath, f'{get_tilename_from_edem_fpath(bfile)}_{tif}')
    return fo_fpath

######################################################## preprocessing:::c:::gen_features.py
def load_data_gdal(fpath):
    dataset = gdal.Open(fpath)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    return array, dataset

def run_gdaldem(mode, input_dem, output_file, options):
    """
    Run gdaldem with the given mode, input DEM, output file, and options.
    
    Parameters:
    mode (str): The gdaldem mode to run (e.g., 'slope', 'aspect', etc.).
    input_dem (str): Path to the input DEM file.
    output_file (str): Path to the output file.
    options (list): List of additional options for gdaldem.
    """
    command = ['gdaldem', mode, input_dem, output_file] + options
    subprocess.run(command, check=True)
    print(f"{mode.capitalize()} saved to: {output_file}")


def dem_derivatives(dem_path):
    """
    Calculate various DEM derivatives including slope, aspect, hillshade, curvature, TPI, and roughness.
    
    Parameters:
    dem_path (str): Path to the DEM file.
    
    Returns:
    dict: Paths to the generated derivative files.
    """
    # Generate output paths based on input path
    slope_path = dem_path.replace('.tif', '_slp.tif')
    aspect_path = dem_path.replace('.tif', '_asp.tif')
    hillshade_path = dem_path.replace('.tif', '_hsd.tif')
    curvature_path = dem_path.replace('.tif', '_cur.tif')
    tpi_path = dem_path.replace('.tif', '_tpi.tif')
    roughness_path = dem_path.replace('.tif', '_rgx.tif')
    
    # Calculate and save Slope
    run_gdaldem('slope', dem_path, slope_path, ['-of', 'GTiff'])
    
    # Calculate and save Aspect
    run_gdaldem('aspect', dem_path, aspect_path, ['-of', 'GTiff'])
    
    # Calculate and save Hillshade
    run_gdaldem('hillshade', dem_path, hillshade_path, ['-of', 'GTiff'])
    
    # Calculate and save Curvature (TRI)
    run_gdaldem('TRI', dem_path, curvature_path, ['-of', 'GTiff'])
    
    # Calculate and save TPI
    run_gdaldem('TPI', dem_path, tpi_path, ['-of', 'GTiff'])
    
    # Calculate and save Roughness
    run_gdaldem('roughness', dem_path, roughness_path, ['-of', 'GTiff'])
    
    return {
        'slp': slope_path,
        'asp': aspect_path,
        'hsd': hillshade_path,
        'cur': curvature_path,
        'tpi': tpi_path,
        'rgx': roughness_path
    }

def fourier_transform(fpath):
    img, src = load_data_gdal(fpath)
    # Perform Fourier transformation
    dark_image_grey_fourier = np.log(abs(np.fft.fftshift(np.fft.fft2(img))))
    cF_path = fpath.replace('.tif', '_fft.tif')
    save_array_to_raster(dark_image_grey_fourier, src, cF_path)
    print('fourier_transform')

def wavelet_transform(fpath):
    """
    Perform 2D wavelet decomposition and Fourier transformation on a raster image,
    and save the resulting coefficients to disk.
    
    Parameters:
    fpath (str): Path to the input raster file.
    """
    img, src = load_data_gdal(fpath)

    # Perform 2D wavelet decomposition
    coeffs = pywt.dwt2(img, 'db1', mode='symmetric')

    # Extract detail and approximation coefficients
    cA, (cH, cV, cD) = coeffs

    # Define output paths
    cA_path = fpath.replace('.tif', '_aw.tif') # _Awavelet
    cH_path = fpath.replace('.tif', '_hw.tif')
    cV_path = fpath.replace('.tif', '_vw.tif')
    cD_path = fpath.replace('.tif', '_dw.tif')
    

    # Save the coefficients and Fourier transform results to disk
    save_array_to_raster(cA, src, cA_path)
    save_array_to_raster(cH, src, cH_path)
    save_array_to_raster(cV, src, cV_path)
    save_array_to_raster(cD, src, cD_path)
    

    print(f"Wavelet transform results saved to: {cA_path}, {cH_path}, {cV_path}, {cD_path}")


def save_array_to_raster(array, src, output_path):
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(
        output_path, src.RasterXSize, src.RasterYSize, 1, src.GetRasterBand(1).DataType)
    out_raster.SetGeoTransform(src.GetGeoTransform())
    out_raster.SetProjection(src.GetProjection())
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.FlushCache()

def get_features(fpath):
    wavelet_transform(fpath)
    fourier_transform(fpath)
    dem_derivatives(fpath)



##### from another script 

#####################################################################
################### tdem bad pixel removal
####################################################################


import sys
import os
import numpy as np
from osgeo import gdal
import scipy.ndimage as ndimage

def open_raster(filename):
    """Open a raster file and return the dataset and the first band as a masked array."""
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    nodata_value = band.GetNoDataValue()
    mask = np.equal(array, nodata_value)
    return np.ma.masked_array(array, mask=mask), ds

def write_raster(array, filename, reference_ds):
    """Write a masked array to a GeoTIFF file using a reference dataset for georeferencing."""
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(filename, reference_ds.RasterXSize, reference_ds.RasterYSize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(reference_ds.GetGeoTransform())
    out_ds.SetProjection(reference_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array.filled(array.fill_value))
    out_band.SetNoDataValue(array.fill_value)
    out_ds = None

def dem_instrument_noise_processing(dem_path,err_path,wam_path,com_path):
    ### need to change this values to reflect my layers - read the documentation and copwbm 
    print(dem_path)
    dem, dem_ds = open_raster(dem_path)
    print('load dataset')

    # Get original mask, True where masked
    mask = np.ma.getmaskarray(dem)
    print('load mask')

    # Theoretical height error
    err, _ = open_raster(err_path)
    max_err_multi = 1.5
    mask = np.logical_or(mask, (err.data > max_err_multi))
    print('mask hem')

    # Water mask
    wam, _ = open_raster(wam_path)
    wam_clim = (33, 127) # change this to cops 
    #mask = np.logical_or(mask, (wam >= wam_clim[0]) & (wam <= wam_clim[1]))
    # bz cop_wbm has 0,1,2,3 where 0 is land and we remove all else 
    #0:	No water #1:Ocean #2:Lake #3:River
    mask = np.logical_or(mask, wam != 0)
    print('mask Water')

    # Consistency mask
    com, _ = open_raster(com_path)
    com_invalid = (0, 1, 2)
    mask = np.logical_or(mask, np.isin(com.data, com_invalid))
    print('mask Consistency')

    # Apply mask
    dem_masked = np.ma.array(dem, mask=mask)
    #print(dem_masked.count())
    # out_masked_fn = os.path.splitext(dem_path)[0] + '_masked.tif'
    # write_raster(dem_masked, out_masked_fn, dem_ds)
    print('mask all pixels')

    # Dilate mask by n_iter px to remove isolated pixels and values around nodata
    n_iter = 1
    mask = ndimage.binary_dilation(mask, iterations=n_iter)
    # To keep valid edges, do subsequent erosion
    mask = ndimage.binary_erosion(mask, iterations=n_iter)
    print('mask all pixels and erode')

    # Apply mask
    dem_masked = np.ma.array(dem, mask=mask)
    print(dem_masked.count())
    out_masked_erode_fn = os.path.splitext(dem_path)[0] + '_M.tif'
    write_raster(dem_masked, out_masked_erode_fn, dem_ds)

    return out_masked_erode_fn


def generate_binmask(input_raster_path, lthresh=-30, hthresh=700):
    # Open the input raster file
    output_raster_path = input_raster_path.replace('.tif', '_BIN.tif')
    
    with rasterio.open(input_raster_path) as src:
        # Read the raster data as a NumPy array
        array = src.read(1)
        
        # Extract the NoData value
        nodata = src.nodata
        
        # Convert NoData values to np.nan
        array = np.where(array == nodata, np.nan, array)
        
        # Apply the thresholds
        array = np.where(array < lthresh, np.nan, array)
        array = np.where(array > hthresh, np.nan, array)
        
        # Create the binary mask
        mask = np.where(np.isnan(array), 0, 1)
        
        # Define the metadata for the output raster
        out_meta = src.meta.copy()
        out_meta.update({"dtype": 'uint8', "nodata": 0})
        
        # Write the binary mask to the output raster file
        with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
            dst.write(mask, 1)
        print(output_raster_path)