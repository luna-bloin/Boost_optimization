# imports
import sys
sys.path.append("../utils")
import alloc as ac
import xarray as xr

# Configurations
try: 
    file_to_open = sys.argv[1]
    n_top = eval(sys.argv[2])
    n_batch = eval(sys.argv[3])
    alloc_type = sys.argv[4]
except IndexError:
    file_to_open = '/net/xenon/climphys/lbloin/optim_boost/boosted_CH_temp_max_lead_ID_True.nc'
    n_top = 10
    n_batch = 10
    alloc_type = "Weighted"

print(f"opening {file_to_open}, allocation length = {n_top}, batch size = {n_batch}")

# === READING IN BOOSTED FILES ===
# Read in boosted data from the screening/previous allocation rounds and stack along case/lead time to get one ID
screening_data = xr.open_dataset(file_to_open).stack(for_sorting=("member","lead_ID")).dropna(dim="for_sorting")
screening_data_sorted = screening_data.sortby("score",ascending=False).score

# === Run allocation algorithm ===
# find allocation given screening input
lead_ID_dict = ac.find_alloc(alloc_type,
                             screening_data.lead_ID,
                             screening_data_sorted,
                             n_top,
                             n_batch
                            )
print(lead_ID_dict)
    
