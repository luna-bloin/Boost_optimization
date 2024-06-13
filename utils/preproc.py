import xarray as xr
import numpy as np
import dask.config
import glob
import csv
import datetime as dt
from tqdm import tqdm
import regionmask

from utils import read_area, to_000, read_regionmask, to_dt,read_boost

def preprocess(ds,read_type,area="global",var="TREFHTMX"):
    """returns a dataset ds with a variable var, either as a linear mean over an area, or as global map """
    ds_out = xr.Dataset()
    if area == "global":
        ds_out[var] = ds[var]
        return ds_out
    else:
        if read_type == "box":
            [min_lat,max_lat,min_lon,max_lon] = read_area(area)
            ds = ds.sel(lat=slice(min_lat,max_lat),lon=slice(min_lon,max_lon))
        elif read_type == "regionmask":
            country = read_regionmask(area)
            mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds.lon,ds.lat)
            ds = ds.where(mask == country,drop=True)
        if var in ds.data_vars:
            da = ds[var]
            wgt = np.cos(np.deg2rad(ds.lat))
            ds_out[var] = da.weighted(wgt).mean(("lat","lon"))
        else:
            print(f"Error, {var} not in data vars")
        return ds_out

def preproc_unpert(in_path, output_path,area,read_type):
    """preprocesses all micro ensemble members (2005-2035) for a specified area for TREFHTMX. Saves the output in location specified by output_path"""
    dss = []
    #define preproc for that area
    def prep(ds):
        return preprocess(ds,read_type,area=area)
    for mem in tqdm(range(1,31)): #TODO: what about extra members 31-35?
        file = sorted(glob.glob(in_path+f"b.e212.B*cmip6.f09_g17.001.2005.ens{to_000(str(mem))}/archive/atm/hist/b.e212.B*cmip6.f09_g17.001.2005.ens{to_000(str(mem))}.cam.h1.*-01-01-00000.nc"))
        with xr.open_mfdataset(file,preprocess = prep) as ds:
            dss.append(ds.sel(time=slice("2005","2035")))

    ds = xr.concat(dss,dim = "member")
    ds["member"] = range(1,31)
    ds = ds.set_coords('member')
    print("processed")
    ds.to_netcdf(output_path+f"TREFHTMX_{area}_2005-2035.nc")
    print("saved")
    return None

def preproc_boost(boost, output_path,area,read_type):
    """preprocesses all boosted cases (specified for each area in csv files in folder inputs), for a specified area for TREFHTMX. Saves the output in location specified by output_path"""
    #define preproc for that area
    def prep(ds):
        return preprocess(ds,read_type,area=area)
    area_boost = read_boost(area)
    for case in area_boost:
        print(case)
        mem = case[0:-5]
        date = area_boost[case][0]
        end_date = area_boost[case][1]
        delta = dt.timedelta(days=1)
        
        #file naming conventions
        if int(case[3:]) < 2015:
            fi_len = [77,80] 
        else:
            fi_len = [79,82] 
        # open and preprocess for all lead times
        dss = []
        dates = []
        while date <= end_date:
            files = sorted(glob.glob(boost+f"B*cmip6.000*{mem}.{date}.ens*/atm/hist/B*cmip6.*.*.ens*.cam.h1.*-00000.nc"))
            f = [fi for fi in files if "old" not in fi]
            print(date,len(f))
            if f != []:
                dates.append(str(date))
                with xr.open_mfdataset(f, preprocess=prep,concat_dim="member", combine="nested",parallel=True) as ds:
                    print("opened")
                    print([(fi[fi_len[0]:fi_len[1]]) for fi in f])
                    ds["member"] = [int(fi[fi_len[0]:fi_len[1]]) for fi in f]
                    ds = ds.set_coords('member')
                    print("processed")
                    dss.append(ds)
            date += delta
        # gathering all lead times into one file
        ds=xr.concat(dss,dim="start_date")
        ds["start_date"] = dates
        ds = ds.set_coords('start_date')
        print("processed")
        ds.to_netcdf(output_path+f"TREFHTMX_{area}_boosted_{case}.nc")
        print("saved")
    return None