# imports
import sys
sys.path.append("../utils")
import bootstrap_alloc as ba
import xarray as xr

# Configurations
try: 
    to_open = sys.argv[1]
    n_top = eval(sys.argv[2])
    n_batch = eval(sys.argv[3])
    n_start_batch = eval(sys.argv[4])
    len_loop = eval(sys.argv[5])
    bootstrap = eval(sys.argv[6])
except IndexError:
    to_open = 'CH_temp_max_lead_ID_True_roll3'
    n_top = [3,5,7,10,12,15,20,25,30,50]
    n_batch = [5,10,20,30,50,75,100,200,300,500]
    n_start_batch = [3,5,7,10,20,25]
    len_loop = 2
    bootstrap = 1000
print(f"Bootstrap sweep for {to_open}:n_top ={n_top},n_batch={n_batch},n_start_batch={n_start_batch},len_loop={len_loop},bootstrap={bootstrap}")

# Paths
in_path = '/net/xenon/climphys/lbloin/optim_boost/'
# === READING IN BOOSTED FILES ===
print("Reading in boosted data files")
# Read in boosted data from the screening/previous allocation rounds and stack along case/lead time to get one ID
ds = xr.open_dataset(f"{in_path}boosted_{to_open}.nc")
# for now, lead times -20 to -10 and only members 1 to 100
ds = ds.sel(member=slice(1,100))

# === Run allocation algorithm for all scores chosen (with bootstrap) ===
print("Allocation algorithm")
ba.score_diff_config(ds.score, 
                     n_top, 
                     n_batch,
                     n_start_batch,
                     len_loop, 
                     bootstrap, 
                     to_open,
                     )
    
