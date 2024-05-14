import xarray as xr
import numpy as np
import dask.config
import glob
import csv
import datetime as dt
from tqdm import tqdm
import regionmask

from utils import read_area, to_000, read_regionmask, to_dt

def preprocess(ds,read_type,area="global",var="TREFHTMX"):
    """returns a ds over a variable var, either cropped and as a linear mean over an area, or as global map """
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
    """preprocesses all micro ensemble members (2005-2035) for a specified area for TREFHT. Saves the output in location specified by output_path"""
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

# boosted files to read TODO: create separate file for this
areas = { 
    "PNW": {
        "10:2017-07-31": [dt.date(2017, 7, 10), dt.date(2017, 7, 28)],
        "10:2007-08-04": [dt.date(2007, 7, 13), dt.date(2007, 7, 27)],
        "29:2033-07-11": [dt.date(2033, 6, 26), dt.date(2033, 7, 4)],
        "06:2013-07-07": [dt.date(2013, 6, 8), dt.date(2013, 6, 30)],
        "12:2028-07-30": [dt.date(2028, 7, 16), dt.date(2028, 7, 27)],
        "18:2034-07-12": [dt.date(2034, 6, 26), dt.date(2034, 7, 4)],
        "12:2031-08-28": [dt.date(2031, 7, 1), dt.date(2031, 8, 24)],
        "23:2033-06-15": [dt.date(2033, 5, 25), dt.date(2033, 6, 11)]
    },
    "MID": {
        "27:2029-08-03": [dt.date(2029, 7, 19), dt.date(2029, 7, 30)],
        "15:2030-07-31": [dt.date(2030, 7, 16), dt.date(2030, 7, 27)],
        "06:2026-07-27": [dt.date(2026, 7, 12), dt.date(2026, 7, 23)]
    },
    "PAR": {
        "13:2030-08-23": [dt.date(2030, 8, 8), dt.date(2030, 8, 19)],
        "19:2028-08-03": [dt.date(2028, 7, 19), dt.date(2028, 7, 30)],
        "19:2016-07-16": [dt.date(2016, 7, 1), dt.date(2016, 7, 12)]
    },
    "CH":{
        "12:2028-08-10": [dt.date(2028,7,16), dt.date(2028,7,26)],
        "03:2031-08-04": [dt.date(2031,7,10),dt.date(2031,7,20)],
        "13:2030-08-21": [dt.date(2030,7,27),dt.date(2030,8,6)],
        "22:2015-08-01": [dt.date(2015,7,7), dt.date(2015,7,17)],
        "27:2028-08-06": [dt.date(2028,7,12),dt.date(2028,7,22)]
    }
}

def preproc_boost(boost, output_path,area,read_type):
    """preprocesses all boosted cases for a specified area. TREFHT and Z500. Saves the output in location specified by output_path"""
    #define preproc for that area
    def prep(ds):
        return preprocess(ds,read_type,area=area)
    for case in areas[area]:
        print(case)
        mem = case[0:-11]
        date = areas[area][case][0]
        end_date = areas[area][case][1]
        delta = dt.timedelta(days=1)
        
        #file naming conventions
        if int(case[3:7]) < 2015:
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
                dates.append(date)
                with xr.open_mfdataset(f, preprocess=prep,concat_dim="member", combine="nested",parallel=True) as ds:
                    print("opened")
                    print([(fi[fi_len[0]:fi_len[1]]) for fi in f])
                    ds["member"] = [int(fi[fi_len[0]:fi_len[1]]) for fi in f]
                    ds = ds.set_coords('member')
                    print("processed")
                    dss.append(ds)
            date += delta
        # gathering all lead times into one file
        ds=xr.concat(dss,dim="lead_time")
        # creating lead time coords by finding start_date - peak date
        peak_dayofyear = int(to_dt(case[3:]).timetuple().tm_yday)
        lead_times = [int(date.timetuple().tm_yday)- peak_dayofyear for date in dates]
        ds["lead_time"] = lead_times
        ds = ds.set_coords('lead_time')
        print("processed")
        ds.to_netcdf(output_path+f"TREFHTMX_{area}_boosted_{case}.nc")
        print("saved")
    return None