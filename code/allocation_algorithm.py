# imports
import sys
sys.path.append("../utils")
import alloc as ac
import xarray as xr

# Configurations
try: 
    file_to_open = sys.argv[1]
    n_alloc = eval(sys.argv[2])
    n_batch = eval(sys.argv[3])
except IndexError:
    file_to_open = '/net/xenon/climphys/lbloin/optim_boost/boosted_CH_temp_max.nc'
    n_alloc = 10
    n_batch = 10

print(f"opening {file_to_open}, allocation length = {n_alloc}, batch size = {n_batch}")

# === READING IN BOOSTED FILES ===
print("Reading in boosted data files")
# Read in boosted data from the screening/previous allocation rounds and stack along case/lead time to get one ID
screening_data = xr.open_dataset(file_to_open).stack(for_sorting=("member","lead_ID")).dropna(dim="for_sorting")
screening_data_sorted = screening_data.sortby("TREFHTMX",ascending=False).TREFHTMX

# === Run allocation algorithm ===
print("Allocation algorithm")
# find score from screening phase
scores_screening = ac.score_mean(screening_data_sorted)
# find allocation given screening input
lead_ID_dict = ac.find_alloc_weighted(screening_data_sorted,n_alloc,n_batch)
print(lead_ID_dict)
    
