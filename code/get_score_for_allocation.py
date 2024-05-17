# imports
import sys
sys.path.append("../utils")
import utils as ut
import def_score as sd
import xarray as xr
import glob
import datetime as dt

# Configurations
try: 
    area = sys.argv[1] 
    nb_heatw_day = eval(sys.argv[2])
    temp_max = eval(sys.argv[3])
    temp_max_anom = eval(sys.argv[4])
except IndexError:
    area = "CH"
    nb_heatw_day = True # score type 1: total number of heatwave days in summer
    temp_max = True # score type 2: maximum temperature around unperturbed peak
    temp_max_anom = True # score_type 3: maximum temperature relative to climatology around unperturbed peak
print(f"{area}, Scoring number of heatwave days = {nb_heatw_day}, Maximum temperature = {temp_max}, Maximum temperature with anomaly = {temp_max_anom}")

# Paths
in_path = '/net/xenon/climphys/lbloin/optim_boost/'

# === READING IN NECESSARY FILES ===
print("Reading in necessary files")
# Read in climatology data
clim = xr.open_dataset(in_path + f"TREFHTMX_{area}_2005-2035.nc").TREFHTMX
mn = clim.groupby("time.dayofyear").mean().rolling(dayofyear=20,center=True).mean().mean("member")
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
boosted_data_around_peak = ut.multi_to_single_index(boosted_data_around_peak)
#keep all values
boosted_data = xr.open_mfdataset(files,concat_dim="case", combine="nested",parallel=True)
boosted_data["case"] = cases
boosted_data = ut.multi_to_single_index(boosted_data)

# === Defining one (or several) score(s) ===
print("Getting scores")
to_score = {}
if nb_heatw_day == True:
    to_score["heatw_day"] = sd.nb_heatw_days(boosted_data, clim)
if temp_max == True:
    to_score["temp_max"] = sd.temp_max(boosted_data_around_peak, mn, anom = False)
if temp_max_anom == True:
    to_score["temp_max_anom"] = sd.temp_max(boosted_data_around_peak, mn, anom = True)

print("Saving as netcdf")
for score in to_score:
    to_score[score].to_netcdf(f"{in_path}boosted_{area}_{score}.nc")