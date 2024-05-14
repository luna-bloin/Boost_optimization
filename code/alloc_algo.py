# imports
import sys
sys.path.append("../utils")
import alloc as ac
import utils as ut
import def_score as sd
import xarray as xr
import csv
from tqdm import tqdm
import glob
import datetime as dt

# Configurations
try: 
    area = sys.argv[1] 
    nb_heatw_day = eval(sys.argv[2])
    temp_max = eval(sys.argv[3])
    temp_max_anom = eval(sys.argv[4])
except IndexError:
    area = "PNW"
    nb_heatw_day = True # score type 1: total number of heatwave days in summer
    temp_max = False # score type 2: maximum temperature around unperturbed peak
    temp_max_anom = False # score_type 3: maximum temperature relative to climatology around unperturbed peak
print(f"{area}, Scoring number of heatwave days = {nb_heatw_day}, Maximum temperature = {temp_max}, Maximum temperature with anomaly = {temp_max_anom}")


# Parameters (list of numbers indicate a varying parameter)
n_alloc = [1,2,4,6,8,10,15,20,25,30]
n_batch = [1,2,4,6,8,10,15,20,25,30]
n_start_batch = [1,2,4,6,8,10,15,20,25,30]
len_loop = 1
bootstrap = 100

# Paths
boost = "/net/meso/climphys/cesm212/boosting/archive/"
in_path = '/net/xenon/climphys/lbloin/optim_boost/'

# === READING IN NECESSARY FILES ===
print("Reading in necessary files")
# Read in climatology data
clim = xr.open_dataset(in_path + f"TREFHTMX_{area}_2005-2035.nc").TREFHTMX
mn = clim.groupby("time.dayofyear").mean().rolling(dayofyear=20,center=True).mean()
# read in boosted data
files = glob.glob(in_path+f"TREFHTMX_{area}_boosted_*.nc")
cases = [file[-16:-3] for file in files]
# keep only values around peak
dss = []
for file in files:
    peak = file[-13:-3]
    dss.append(xr.open_dataset(file).convert_calendar("proleptic_gregorian").sel(time = slice(ut.to_dt(peak) - dt.timedelta(days = 5), ut.to_dt(peak) + dt.timedelta(days = 5))))
boosted_data_around_peak = xr.concat(dss, dim="case")
boosted_data_around_peak["case"] = cases
boosted_data_around_peak=boosted_data_around_peak.stack(event=("case","lead_time")).rename({"event":"lead_ID"})
#keep all values
boosted_data = xr.open_mfdataset(files,concat_dim="case", combine="nested",parallel=True)
boosted_data["case"] = cases
boosted_data = boosted_data.stack(event=("case","lead_time")).rename({"event":"lead_ID"})

# === Defining one (or several) score(s) ===
print("Getting scores")
to_score = {}
if nb_heatw_day == True:
    to_score["heatw_day"] = sd.nb_heatw_days(boosted_data, clim)
if temp_max == True:
    to_score["temp_max"] = sd.temp_max(boosted_data_around_peak, mn, anom = False)
if temp_max_anom == True:
    to_score["temp_max_anom"] = sd.temp_max(boosted_data_around_peak, mn, anom = True)

# === Run allocation algorithm for all scores chosen (with bootstrap) ===
print("Allocation algorithm")
for score in to_score:
    ac.score_diff_config(to_score[score], 
                         n_alloc, 
                         n_batch,
                         n_start_batch,
                         len_loop, 
                         bootstrap, 
                         area,
                         score_typ=score
                         )
    
# # Reading in boosted run info
# ds_boost_info = {}
# for mem_typ in ["100","300"]:
#     boost_size = {}
#     with open(f"../inputs/boost/boost_{area}_{mem_typ}.csv" ,"r") as f:
#         rd = csv.reader(f,delimiter=";")
#         for i,row in enumerate(rd):
#             if i == 0:
#                 continue
#             boost_size[row[0]] = [row[1],row[2]]
#     ds_boost_info[mem_typ] = boost_size

# #Reading in original run + mean
# origs_all = {}
# for mem_typ in ["100","300"]:
#     origs = {}
#     for case in tqdm(ds_boost_info[mem_typ]):
#         orig = ut.to_cel(xr.open_dataset(in_path+f"TREFHTMX_Z500_x5d_{area}_2005-2035.nc").groupby("time.season")["JJA"].sel(member=int(case[0:2]),time=case[3:]).TREFHTMX_x5d)
#         origs[case] = orig
#     origs_all[mem_typ] = origs

# # Reading in boosted data and preparing events 
# ds_boost_all = {}
# for mem_typ in ["100","300"]:
#     ds_boost = {}
#     for case in tqdm(ds_boost_info[mem_typ]):
#         # find day of peak in parent run
#         if peak_anom == True:
#             peak_date = ((origs_all[mem_typ][case].groupby("time.dayofyear")-mn).groupby("time.dayofyear")/std).idxmax()
#         else:
#             peak_date = ((origs_all[mem_typ][case].groupby("time.dayofyear") - mn).groupby("time.dayofyear")+mn).idxmax()
#         # open boost, create event (5-day rolling mean maximum temperature within 5 days of orig peak
#         ds = xr.open_dataset(in_path+f"TREFHTMX_Z500_x5d_{area}_boosted_{case}.nc")
#         ds["start_date"] = [f"2{date}"for date in ds.start_date.values] #TODO: fix this bug for CH
#         if anomaly_ds == False:
#             anomaly = False
#         elif anomaly_ds == True:
#             anomaly = mn
#         ds_boost[case] = sc.create_event(ds,mem_typ, peak_date, anomaly = anomaly)
#     ds_boost_all[mem_typ] = ds_boost
        
# # ==================================================
# print("scoring")
# # === Scoring algorithm ===
# save_adds = area
# if anomaly_ds == True:
#     save_adds = f"{save_adds}_anomaly"

# print(save_adds)
# for mem_typ in ["100","300"]:
#     if together == False:
#         for case in ds_boost_all[mem_typ]:
#             print(case)
#             ds = ds_boost_all[mem_typ][case] 
#             sc.score_diff_config(ds, 
#                                  n_top, 
#                                  n_alloc, 
#                                  n_batch, 
#                                  len_loop, 
#                                  bootstrap, 
#                                  save_adds,
#                                  restrict = restrict,
#                                  together = case
#                                 )
#     else:
#         ds = xr.concat(list(ds_boost_all[mem_typ].values()), dim="case")
#         ds["case"] = list(ds_boost_all[mem_typ].keys())
#         sc.score_diff_config(ds, 
#                              n_top, 
#                              n_alloc, 
#                              n_batch, 
#                              len_loop, 
#                              bootstrap, 
#                              save_adds,
#                              restrict = restrict,
#                              together = together
#                              )
                        
        