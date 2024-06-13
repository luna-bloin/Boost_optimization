# imports
import sys
sys.path.append("../utils")
import utils as ut
import xarray as xr
import glob
import pandas as pd

# === Script explanation ===
# Preprocesses the output from preprocess_to_time_series to get an event with a score, ready to use in either allocation_algorithm.py or bootstrap_allocation.py
# Reads in file, applies given running mean (and calculates lead time from start date), and calculates score(s). Saves output in outputs folder
# param roll is the rolling mean to be applied
# param with_lead_ID is True if you want to stack case and lead time into one metric called lead_ID (to treat all cases together)
# param nb_heatw_day is True if you want the score to be the total number of heatwave days in summer
# param temp_max is True if you want the score to be the maximum temperature around unperturbed peak
# param temp_max is True if you want the score to be the maximum temperature around unperturbed peak
# ==========================


# Configurations
try: 
    area = sys.argv[1] 
    roll = sys.argv[2] 
    with_lead_ID = eval(sys.argv[3])
    nb_heatw_day = eval(sys.argv[4])
    temp_max = eval(sys.argv[5])
    temp_max_anom = eval(sys.argv[6])
except IndexError:
    area = "CH"
    roll = 3
    with_lead_ID = True # whether to stack case and lead time into one metric called lead_ID    
    nb_heatw_day = True # score type 1: total number of heatwave days in summer
    temp_max = True # score type 2: maximum temperature around unperturbed peak
    temp_max_anom = True # score_type 3: maximum temperature relative to climatology around unperturbed peak

print(f"{area}, rolling mean {roll} Scoring number of heatwave days = {nb_heatw_day}, Maximum temperature = {temp_max}, Maximum temperature with anomaly = {temp_max_anom}, with stacking to lead_ID = {with_lead_ID}")

# Paths
in_path = '/net/xenon/climphys/lbloin/optim_boost/'

# === READING IN NECESSARY FILES ===
print("Reading in climatology")
# Read in climatology data
clim = xr.open_dataset(in_path + f"TREFHTMX_{area}_2005-2035.nc").TREFHTMX
mn = clim.groupby("time.dayofyear").mean().rolling(dayofyear=20,center=True).mean().mean("member")
# read in boosted data
files = glob.glob(in_path+f"TREFHTMX_{area}_boosted_*.nc")
cases = [file[-10:-3] for file in files]


# === Defining one (or several) score(s) ===
print("Reading in boosting and getting scores")
to_score = {} # dictionary of types of event scores to save
# open and process boosted data
for around_peak in [True,False]:
    boost = []
    for file in files:
        case = file[-10:-3]
        parent = clim.sel(member=int(case[0:2]),time=case[3:7]).rolling(time=roll, center=True).mean().convert_calendar("proleptic_gregorian")
        peak = parent.idxmax().values
        ds = xr.open_dataset(file).convert_calendar("proleptic_gregorian").rolling(time=roll, center=True).mean()
        ds["start_date"] = [(pd.to_datetime(ld)-peak).days for ld in ds.start_date.values]
        ds = ds.rename({"start_date":"lead_time"})
        if around_peak == True:
            ds = ds.sel(time = slice(peak - pd.Timedelta(days = 5), peak + pd.Timedelta(days = 5)))# keep only values around peak
        boost.append(ds)
    boost = xr.concat(boost, dim="case")
    boost["case"] = cases
    if with_lead_ID == True:
        boost = ut.multi_to_single_index(boost)
    if around_peak == True:
        if temp_max == True:
            to_score["temp_max"] = boost.max("time").TREFHTMX
        if temp_max_anom == True:
            anom = boost.groupby("time.dayofyear")-mn
            to_score["temp_max_anom"] = anom.max("time").TREFHTMX
    else:
        if nb_heatw_day == True:
            ds_summer = boost.groupby("time.season")["JJA"].TREFHTMX.compute()
            #finding 90th percentile of summer temperatures (=defined as heatwave)
            q_90 = clim.groupby("time.season")["JJA"].quantile(0.9).compute()
            number_of_heatw_days = ds_summer.where(ds_summer >= q_90,drop=True).count(dim='time')
            to_score["heatw_day"] = number_of_heatw_days

print("Saving as netcdf")
for score in to_score:
    print(f"saving {score}")
    to_score[score].to_dataset(name="score").to_netcdf(f"{in_path}boosted_{area}_{score}_lead_ID_{with_lead_ID}_roll{roll}.nc")