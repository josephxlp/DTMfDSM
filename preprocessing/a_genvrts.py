import os 
from os import makedirs
from glob import glob
from concurrent.futures import ProcessPoolExecutor
#from upaths import txt_files, tif_files
#from prprocessing import list2vrt
from prprocessing import (filter_x_isin_list,filter_x_notin_list,gen_vrt_params,
                          process_lfiles,get_all_VRT_TXT_FILE_paths,buildVRT_from_list)

from utils import read_yaml, write_yaml,makedict


import upaths_wx as uP

# fix the code for uP.VRTs_dpath
if __name__ == '__main__':

    #with ProcessPoolExecutor() as PPE: PPE.map(list2vrt,tif_files,txt_files)
    
    makedirs(uP.VRTs_dpath, exist_ok=True)
    print('pband processing... ')
    pband_files = glob(f'{uP.pband_dpath}/*/*.tif')
    pband_files = filter_x_isin_list(pband_files, uP.pband_k)
    pband_fpath = pband_files[0]
    single_path_files = [uP.egm08_fpath, uP.egm96_fpath,pband_fpath]

    print('edem_files processing... ')
    edem_files = glob(f'{uP.edem_dpath}/*/*/*.tif', recursive=True);# print(len(edem_files))
    edem_files = filter_x_notin_list(edem_files, uP.preview_k); #print(len(edem_files))
    edem_dict = gen_vrt_params(uP.VRTs_dpath, uP.edem_vars,'edem',edem_files)

    print('cdem_files processing... ')
    cdem_files = glob(f'{uP.cdem_dpath}/*/*/*.tif', recursive=True);# print(len(cdem_files))
    cdem_files = filter_x_notin_list(cdem_files, uP.preview_k);# lenlist(cdem_files)
    cdem_dict = gen_vrt_params(uP.VRTs_dpath, uP.cdem_vars,'cdem',cdem_files)

    print('tdem_files processing... ')
    tdemx_files = glob(f'{uP.tdemx_dpath}//**/*.tif', recursive=True); #print(len(tdemx_files))
    tdemx_files = filter_x_notin_list(tdemx_files, uP.preview_k);# print(len(tdemx_files))
    tdem_dict = gen_vrt_params(uP.VRTs_dpath, uP.tdem_vars,'tdem',tdemx_files)

    print('esa_files processing... ')
    esa_files = glob(f'{uP.esa_dpath}//**/*.tif', recursive=True)

    print('dtm_files processing... ')
    dtm_files = glob(f'{uP.dtm_dpath}/*/*.tif', recursive=True)
    dtm_dils = [i for i in dtm_files if 'ESTONIA' not in i]
    #print(dtm_dpath)
    #print(f'{dtm_dpath}/*/*.tif')
    #print('dtm_files::::::')
    #uops.lenlist(dtm_files)
    #uops.makedir(dtm_wrapped_dpath)
    ldar_wrapped_files = process_lfiles(dtm_dils, uP.dtm_wrapped_dpath, uP.epsg, uP.xres)

    #dtm_files = os.listdir(dsm_wrapped_dpath)
    print('Reprojected Lidar:', len(ldar_wrapped_files))
    print(ldar_wrapped_files)

    #print('dtm_files processing... ')
    #dsm_files = glob.glob(f'{dsm_dpath}//**/*.tif', recursive=True)
    
    # print('dsm_files:::::')
    # #uops.lenlist(dsm_files)
    # #uops.makedir(dsm_wrapped_dpath)
    # #dsm_wrapped_files = rops.process_dsms(dsm_files, dsm_wrapped_dpath, epsg_dem, xres)
    mfiles = [esa_files, ldar_wrapped_files]
    # #mfiles = [ethchm_files, esawc_files , wsf2019_files, gfc2020_files,
    # #         dtm_wrapped_files]#,dsm_wrapped_files]

    mdict = gen_vrt_params(uP.VRTs_dpath, uP.mdataset_names,'multi',mfiles)
    #for i in multi_vars: lenlist(mdict[i]['files'])
    yaml_data = {'mdict': mdict,'cdem_dict': cdem_dict,
                'edem_dict': edem_dict,'tdem_dict': tdem_dict}
    
    write_yaml(yaml_data, uP.step0_yaml_fpath)
    yaml_data = read_yaml(uP.step0_yaml_fpath)

    VRT_paths, TXT_paths, FILE_paths = get_all_VRT_TXT_FILE_paths(yaml_data)
    print(len(VRT_paths),len(TXT_paths), len(FILE_paths))

    print('building vrts processing... ')
    with ProcessPoolExecutor(len(VRT_paths)) as ppe:
        #for t,p in zip(TXT_paths, FILE_paths):
            #galBuildVRT_from_list(t,p)
        # ppe.map(galBuildVRT_from_list, TXT_paths,FILE_paths)
        #ppe.map(build_vrt_from_list, TXT_paths,VRT_paths, FILE_paths)
        ppe.map(buildVRT_from_list, TXT_paths,VRT_paths, FILE_paths)

    VRT_names = [os.path.basename(VRT_paths[i][:-4]) for i in range(len(VRT_paths))]
    params_files = single_path_files + VRT_paths 
    params_names = uP.sdataset_names + VRT_names
    params_dict = makedict(params_files,params_names)
    write_yaml(params_dict, uP.step1_yaml_fpath)







