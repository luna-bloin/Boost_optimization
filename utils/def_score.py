#imports
import sys
sys.path.append("../utils")
import alloc as sc
import utils as ut
import matplotlib.pyplot as plt
import xarray as xr
import csv
from tqdm import tqdm

def temp_max(ds, mn, anom = True,rolling_mean = 5):
    #rolling mean
    ds_roll = ds.rolling(time=rolling_mean,center=True).mean().TREFHTMX
    #if you want max of anomaly from climatology, it is calculated here
    if anom != False:
        ds_roll = ds_roll.groupby("time.dayofyear")-mn
    ds_max = ds_roll.max("time")
    return ds_max

def nb_heatw_days(ds,clim):
    ds_summer = ds.groupby("time.season")["JJA"].TREFHTMX.compute()
    #finding 90th percentile of summer temperatures (=defined as heatwave)
    q_90 = clim.groupby("time.season")["JJA"].quantile(0.95).compute()
    number_of_heatw_days = ds_summer.where(ds_summer >= q_90,drop=True).count(dim='time')
    return number_of_heatw_days