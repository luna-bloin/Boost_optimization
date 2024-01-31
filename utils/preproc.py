import xarray as xr
import numpy as np
import dask.config
import glob
import csv
import datetime as dt
from tqdm import tqdm

import utils as ut
from utils import read_area, to_000

def preprocess(ds,area="global"):
    """returns a ds with 5xd running means over variables TREFHTMX and Z500, either cropped and as a linear mean over an area, or as global map """
    ds_out = xr.Dataset()
    if area == "global":
        for var in ["TREFHTMX","Z500"]:
            ds_out[f"{var}_x5d"] = ds[var].rolling(time=5, center=True).mean()
        return ds_out
    else:
        [min_lat,max_lat,min_lon,max_lon] = read_area(area)
        for var in ["TREFHTMX","Z500"]:
            if var in ds.data_vars:
                da = ds[var].sel(lat=slice(min_lat,max_lat),lon=slice(min_lon,max_lon))
                wgt = np.cos(np.deg2rad(ds.lat))
                ds_out[f"{var}_x5d"] = (da.weighted(wgt).mean(("lat","lon")).rolling(time=5, center=True).mean())
            else:
                print(f"Error, {var} not in data vars")
        return ds_out

def preproc_clim(in_path,output_path,area):
    """preprocesses all macro ensemble members to form a climatology between 1981 and 2010 for a specified area. TREFHT and Z500. Saves the output in location specified by output_path"""
    #reading in files
    f_clim = []
    for mb in range(4,15+1): #macro ensemble members, for climatology
        mem = to_000(f"{mb}00",length=4)
        for y in range(1981,2010+1):
            f_clim.append(in_path+f"b.e212.BHISTcmip6.f09_g17.{mem}/archive/atm/hist/b.e212.BHISTcmip6.f09_g17.{mem}.cam.h1.{y}-01-01-00000.nc")
    #preprocess and save
    with dask.config.set(**{'array.slicing.split_large_chunks': True}): # To avoid creating large chunks   
        #define preproc for that area
        def prep(ds):
            return preprocess(ds,area=area)
        clim = xr.open_mfdataset(f_clim, preprocess = prep, concat_dim="member", combine="nested")
        print("processed")
        mn = clim.groupby("time.dayofyear").mean(("member","time"))
        mn.to_netcdf(output_path+f"TREFHTMX_Z500_x5d_{area}_clim.nc")
        clim.to_netcdf(output_path+f"TREFHTMX_Z500_x5d_{area}_clim_full.nc")
        print("saved")
    return None

def preproc_unpert(in_path, output_path,area):
    """preprocesses all micro ensemble members (2005-2035) for a specified area. TREFHT and Z500. Saves the output in location specified by output_path"""
    dss = []
    #define preproc for that area
    def prep(ds):
        return preprocess(ds,area=area)
    for mem in tqdm(range(1,31)):
        file = sorted(glob.glob(in_path+f"b.e212.B*cmip6.f09_g17.001.2005.ens{to_000(str(mem))}/archive/atm/hist/b.e212.B*cmip6.f09_g17.001.2005.ens{to_000(str(mem))}.cam.h1.*-01-01-00000.nc"))
        with xr.open_mfdataset(file,preprocess = prep) as ds:
            dss.append(ds.sel(time=slice("2005","2035")))

    ds = xr.concat(dss,dim = "member")
    ds["member"] = range(1,31)
    ds = ds.set_coords('member')
    print("processed")
    ds.to_netcdf(output_path+f"TREFHTMX_Z500_x5d_{area}_2005-2035.nc")
    print("saved")
    return None

# boosted files to read TODO: create separate file for this
areas = { 
    "PNW": {
        "10-2017": [dt.date(2017, 7, 10), dt.date(2017, 7, 28)],
        "10-2007": [dt.date(2007, 7, 13), dt.date(2007, 7, 27)],
        "29-2033": [dt.date(2033, 6, 26), dt.date(2033, 7, 4)],
        "06-2013": [dt.date(2013, 6, 8), dt.date(2013, 6, 30)],
        "12-2028": [dt.date(2028, 7, 16), dt.date(2028, 7, 27)],
        "18-2034": [dt.date(2034, 6, 26), dt.date(2034, 7, 4)],
        "12-2031": [dt.date(2031, 7, 1), dt.date(2031, 8, 24)],
        "23-2033": [dt.date(2033, 5, 25), dt.date(2033, 6, 11)]
    },
    "MID": {
        "27-2029": [dt.date(2029, 7, 19), dt.date(2029, 7, 30)],
        "15-2030": [dt.date(2030, 7, 16), dt.date(2030, 7, 27)],
        "06-2026": [dt.date(2026, 7, 12), dt.date(2026, 7, 23)]
    },
    "PAR": {
        "13-2030": [dt.date(2030, 8, 8), dt.date(2030, 8, 19)],
        "19-2028": [dt.date(2028, 7, 19), dt.date(2028, 7, 30)],
        "19-2016": [dt.date(2016, 7, 1), dt.date(2016, 7, 12)]
    }
}

def preproc_boost(boost, output_path,area):
    """preprocesses all boosted cases for a specified area. TREFHT and Z500. Saves the output in location specified by output_path"""
    #define preproc for that area
    def prep(ds):
        return ut.preprocess(ds,area=area)
    for case in areas[area]:
        print(case)
        mem = case[0:-5]
        date = areas[area][case][0]
        end_date = areas[area][case][1]
        delta = dt.timedelta(days=1)
        
        #file naming conventions
        if int(case[3:]) < 2015:
            fi_len = [77,80]
        else:
            fi_len = [79,82]

        while date <= end_date:
            files = sorted(glob.glob(boost+f"B*cmip6.000*{mem}.{date}.ens*/atm/hist/B*cmip6.*.*.ens*.cam.h1.*-00000.nc"))
            f = [fi for fi in files if "old" not in fi]
            print(date,len(f))
            if f != []:
                with xr.open_mfdataset(f, preprocess=prep,concat_dim="member", combine="nested",parallel=True) as ds:
                    print("opened")
                    print([(fi[fi_len[0]:fi_len[1]]) for fi in f])
                    ds["member"] = [int(fi[fi_len[0]:fi_len[1]]) for fi in f]
                    ds = ds.set_coords('member')
                    print("processed")
                    ds.to_netcdf(output_path+f"TREFHTMX_Z500_x5d_{area}_boosted_{case}_{date}.nc")
                    print("saved")
            date += delta
    
        # gathering all lead times into one file
        files = glob.glob(output_path+f"TREFHTMX_Z500_x5d_{area}_boosted_{case}_*.nc")
        ds=xr.open_mfdataset(files,concat_dim="start_date", combine="nested",parallel=True)
        ds["start_date"] = [fi[77:-3] for fi in files]
        ds = ds.set_coords('start_date')
        print("processed")
        ds.to_netcdf(output_path+f"TREFHTMX_Z500_x5d_{area}_boosted_{case}.nc")
        print("saved")
    return None