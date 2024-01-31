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

for area in ["PNW","MID","PAR","global"]:
    print(area)
    pc.preproc_clim(in_path, output_path,area)
    pc.preproc_unpert(in_path, output_path,area)
    pc.preproc_boost(boost, output_path,area)

