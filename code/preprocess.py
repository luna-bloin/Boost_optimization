# imports
import sys
sys.path.append("../utils")
import preproc as pc

#source location
in_path = "/net/meso/climphys/cesm212/" 
# processed data
output_path = '/net/xenon/climphys/lbloin/optim_boost/'
# boost data
boost_path = "/net/meso/climphys/cesm212/boosting/archive/"

# configurations
try: 
    areas = [sys.argv[1]]  # string of area to preprocess
    unpert = eval(sys.argv[2]) # whether to preproc this (either true or false)
    boost = eval(sys.argv[3]) # whether to preproc this (either true or false)
except IndexError:
    areas = ["PNW","CH","MID","PAR","global"]
    unpert = True
    boost = True
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
    if unpert == True:
        print("preprocessing unperturbed runs")
        pc.preproc_unpert(in_path, output_path,area,read_type)
    if boost == True:
        print("preprocessing boosted runs")
        pc.preproc_boost(boost_path, output_path,area,read_type)

