# imports
import sys
sys.path.append("../utils")
import score as sc
import utils as ut
import matplotlib.pyplot as plt
import xarray as xr
import csv
from tqdm import tqdm

# Configurations
try: 
    area = sys.argv[1] 
    restrict = eval(sys.argv[2])
    together = eval(sys.argv[3])
    anomaly_ds = eval(sys.argv[4])
except IndexError:
    area = "PNW"
    restrict = False # lead time to only -15 --> -10 days 
    together = False # should algorithm treat all cases together or separately
    anomaly_ds = False # should evens be absolute values or anomalies
print(f"{area}, restrict = {restrict}, together = {together}, anomaly = {anomaly_ds}")
n_top = [1,2,4,6,8,10,15,20,25,30]
n_alloc = [1,2,4,6,8,10,15,20,25,30]
n_batch = [1,2,4,6,8,10,15,20,25,30]
len_loop = 4
bootstrap = 100

# Paths
boost = "/net/meso/climphys/cesm212/boosting/archive/"
in_path = '/net/xenon/climphys/lbloin/optim_boost/'

# For plotting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size":14,
    "font.serif": ["Palatino"]})

# === READING IN NECESSARY FILES ===
print("Reading in necessary files")
# Read in climatology data
clim = ut.to_cel(xr.open_dataset(in_path + f"TREFHTMX_Z500_x5d_{area}_clim_full.nc").TREFHTMX_x5d)
mn = ut.to_cel(xr.open_dataset(f"/net/xenon/climphys/lbloin/optim_boost/TREFHTMX_Z500_x5d_{area}_clim.nc").TREFHTMX_x5d)
std = clim.groupby("time.dayofyear").std(("ens", "time"))

# Reading in boosted run info
ds_boost_info = {}
for mem_typ in ["100","300"]:
    boost_size = {}
    with open(f"../inputs/boost/boost_{area}_{mem_typ}.csv" ,"r") as f:
        rd = csv.reader(f,delimiter=";")
        for i,row in enumerate(rd):
            if i == 0:
                continue
            boost_size[row[0]] = [row[1],row[2]]
    ds_boost_info[mem_typ] = boost_size

#Reading in original run + mean
origs_all = {}
for mem_typ in ["100","300"]:
    origs = {}
    for case in tqdm(ds_boost_info[mem_typ]):
        orig = ut.to_cel(xr.open_dataset(in_path+f"TREFHTMX_Z500_x5d_{area}_2005-2035_ens{int(case[0:2])}.nc").groupby("time.season")["JJA"].sel(time=case[3:]).TREFHTMX_x5d)
        origs[case] = orig
    origs_all[mem_typ] = origs

# Reading in boosted data and preparing events 
ds_boost_all = {}
for mem_typ in ["100","300"]:
    ds_boost = {}
    for case in tqdm(ds_boost_info[mem_typ]):
        # find day of peak in parent run
        peak_date = ((origs_all[mem_typ][case].groupby("time.dayofyear")-mn).groupby("time.dayofyear")/std).idxmax()
        # open boost, create event (5-day rolling mean maximum temperature within 5 days of orig peak
        ds = xr.open_dataset(in_path+f"TREFHTMX_Z500_x5d_{area}_boosted_{case}.nc")
        if anomaly_ds == False:
            anomaly = False
        elif anomaly_ds == True:
            anomaly = mn
        ds_boost[case] = sc.create_event(ds,mem_typ, peak_date, anomaly = anomaly)
    ds_boost_all[mem_typ] = ds_boost
        
# ==================================================
print("scoring")
# === Scoring algorithm ===
save_adds = area
if anomaly_ds == True:
    save_adds = f"{save_adds}_anomaly"

print(save_adds)
for mem_typ in ["100","300"]:
    if together == False:
        for case in ds_boost_all[mem_typ]:
            print(case)
            ds = ds_boost_all[mem_typ][case] 
            sc.score_diff_config(ds, 
                                 n_top, 
                                 n_alloc, 
                                 n_batch, 
                                 len_loop, 
                                 bootstrap, 
                                 save_adds,
                                 restrict = restrict,
                                 together = case
                                )
    else:
        ds = xr.concat(list(ds_boost_all[mem_typ].values()), dim="case")
        ds["case"] = list(ds_boost_all[mem_typ].keys())
        sc.score_diff_config(ds, 
                             n_top, 
                             n_alloc, 
                             n_batch, 
                             len_loop, 
                             bootstrap, 
                             save_adds,
                             restrict = restrict,
                             together = together
                             )
                        
        