# imports
import sys
sys.path.append("../utils")
import preproc as pc

#source location
in_path = "/net/meso/climphys/cesm212/"
# processed data
output_path = '/net/xenon/climphys/lbloin/optim_boost/'
# boost data
boost = "/net/meso/climphys/cesm212/boosting/archive/"

# configurations
try: 
    areas = [sys.argv[1]]  # string of area to preprocess
except IndexError:
    areas = ["PNW","CH","MID","PAR","global"]
print(areas)

# preprocess
for area in areas:
    print(area)
    # deciding if area should be selected through regionmask or just cut out box
    if area == "CH":
        read_type = "regionmask"
    else:
        read_type = "box"
    print(read_type)
    print("preprocessing climatology")
    pc.preproc_clim(in_path, output_path,area,read_type)
    print("preprocessing unperturbed runs")
    pc.preproc_unpert(in_path, output_path,area,read_type)
    print("preprocessing boosted runs")
    pc.preproc_boost(boost, output_path,area,read_type)

