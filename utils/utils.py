import glob
import csv
import datetime as dt
import xarray as xr

def read_area(area):
    """returns a list of min_lat,max_lat,min_lon,max_lon needed to crop the corresponding area"""
    pth = glob.glob(f"../inputs/areas/{area}.csv")
    if pth == []:
        print("no files found")
        return None
    with open(pth[0],"r") as f:
        rd = csv.reader(f)
        rows = []
        for row in rd:
            rows.append(row)
    areas= [ int(rows[1][0]), int(rows[1][1]),int(rows[1][2]), int(rows[1][3]) ]
    return areas

def read_regionmask(area):
    pth = glob.glob(f"../inputs/areas/{area}.csv")
    with open(pth[0],"r") as f:
        rd = csv.reader(f)
        for row in rd:
            return int(row[0])
            
def to_000(mem,length=3):
    while len(str(mem)) < length:
        mem = f"0{mem}"
    return mem

def to_cel(da):
    return da - 273.15

def to_dt(string):
    return dt.datetime(int(string[0:4]),int(string[5:7]),int(string[8:]))

def concat_to_ds(list,dim,typ_name):
    ds = xr.concat(list,dim=dim)
    ds[dim] = typ_name
    return ds

def multi_to_single_index(ds,dims=("case","lead_time"),new_name="lead_ID"):
    stacked = ds.stack(event=dims)
    coords = [str(st) for st in stacked["event"].values]
    stacked = stacked.reset_index("event")
    stacked["event"] = coords
    stacked = stacked.drop_vars(dims)
    stacked = stacked.rename({"event":new_name})
    return stacked