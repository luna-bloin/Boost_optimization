import utils as ut 
import xarray as xr
import numpy as np
import datetime as dt
from tqdm import tqdm
import csv
from numpy.random import default_rng,randint
rng = default_rng()

def create_event(ds,mem_typ,peak_date,restrict=False):
    """returns the 5-day running mean maximum temperature of each member in ds, within 5 days of parent peak"""
    orig_peak = peak_date.dayofyear.values
    dates = ds.start_date.values
    lead_times = [int(ut.to_dt(dat).timetuple().tm_yday-orig_peak) for dat in dates]
    ds["start_date"] = lead_times
    ds=ds.rename({"start_date":"lead_time"})
    if restrict != False:
        ds = ds.sel(lead_time=slice(-15,-10))
    return ut.to_cel(ds.TREFHTMX_x5d).sel(member=range(1,int(mem_typ)+1)).sel(time=slice(peak_date-dt.timedelta(days=5),peak_date+dt.timedelta(days=5))).max("time")

def rank_tops(ds, top=50,dims=('member', 'lead_time')):
    """takes a dataset ds and finds the highest value events over dimensions dim, of length top. Returns a dataset with only these events"""
    ds_stack=ds.stack(event=dims)
    top_runs = ds_stack.sortby(ds_stack,ascending=False).dropna(dim="event").isel(event=slice(0,top))
    return top_runs

def lead_time_sample(ds,lead_dict,len_top,mem_sample=False):
    """samples events from a dataset ds. Batch size is determined for each lead time in the dict lead_dict. mem_sample is optionally included if you want another set to draw from than just the members of the ds. Returns the highest members (length =len_top), and a dict of events that weren't chosen"""
    maxes = [] #list of sampled datasets for each lead time
    non_chosen = {} #dict of non-sampled members (per lead time)
    # if type(mem_sample) != bool:
    #     print(mem_sample.keys())
    for lead_time in lead_dict:
        #choose sample to draw from
        # print(lead_time)
        if mem_sample == False:
            sample = ds.member.values
        else:
            sample = mem_sample[lead_time]
        #choose batch size
        batch_size = lead_dict[lead_time]
        if batch_size > len(sample):
            batch_size = len(sample)
        #draw batch from sample of size batch_size
        batch = rng.choice(sample, size=batch_size,replace=False)
        # only keep maximum values over time
        maxes_batch = ds.sel(lead_time=int(lead_time),member=batch)
        maxes.append(maxes_batch)   
        # non-sampled members
        non_chosen[lead_time] = list(set(sample) - set(batch))
    maxes_tot = xr.concat(maxes,dim="lead_time")
    # find top runs within all the sampled members (across lead times -> batch_size*len(lead_time))
    return rank_tops(maxes_tot,top=len_top),non_chosen

def score_ds(maxes_tot,top_list):
    """scores the sampled events by comparing them to a ground truth. Returns #events found in ground truth/length of sample"""
    score = 0
    for mx in maxes_tot:
        mx_list = f'{mx.lead_time.values}:{mx.member.values}'
        if mx_list in top_list:
            score += 1
    score_avg = score/len(maxes_tot)
    return score_avg

def find_alloc(maxes_tot,batch_size):
    """allocates amount of new samples to draw for each lead time, by giving batch_size number of new samples for each event in that lead time that was in the top selection"""
    occ_per_lead_time = {}
    for mx in maxes_tot:
        if f"{mx.lead_time.values}" in occ_per_lead_time.keys():
            occ_per_lead_time[f"{mx.lead_time.values}"] += batch_size
        else:
            occ_per_lead_time[f"{mx.lead_time.values}"] = batch_size
    return occ_per_lead_time

def find_alloc_weighted(maxes_tot,len_alloc,batch_size):
    """allocates amount of new samples to draw for each lead time, by weighting the top runs by how far they are from the rank 1 event"""
    rank_list = maxes_tot[0:len_alloc] #top events
    occ_per_lead_time = {}
    if len(rank_list) > 1:
        total_dist = rank_list[0]-rank_list[-1]
        relative_dists = np.zeros(len_alloc)
        # find the ratio of relative distande/total distance
        for i,rk in enumerate(rank_list):
            relative_dists[i] = ((rk - rank_list[-1])/total_dist)
        # normalize weights so total new allocated rund correspond ~ to len_alloc*batch_size
        weights = relative_dists*(len_alloc*batch_size)/sum(relative_dists)
        for i, rk in enumerate(rank_list):
            if f"{rk.lead_time.values}" in occ_per_lead_time.keys():
                occ_per_lead_time[f"{rk.lead_time.values}"] += int(weights[i])
            else:
                occ_per_lead_time[f"{rk.lead_time.values}"] = int(weights[i])
    else:
        occ_per_lead_time[f"{rank_list[0].lead_time.values}"] = batch_size
    return occ_per_lead_time

def find_random_alloc(size, lead_times):
    """allocates amount of new samples to draw for each lead time, by drawing random samples from any lead time, in total amounting to size"""
    occ_per_lead_time = {}
    for i in range(size):
        lead_time = rng.choice(lead_times, size=1,replace=True)[0]
        if f"{lead_time}" in occ_per_lead_time:
            occ_per_lead_time[f"{lead_time}"] += 1
        else:
            occ_per_lead_time[f"{lead_time}"] = 1
    for ld in lead_times.values:
        if f"{ld}" not in occ_per_lead_time.keys():
            occ_per_lead_time[f"{ld}"] = 0
    return occ_per_lead_time

def sample_score_alloc(ds,len_loop,batch_size,len_top,top_list,len_alloc,alloc_type="Random"):
    """takes a dataset ds and performs sampling, scoring and allocation for len_loop rounds"""
    scores = np.zeros(len_loop)
    # loop over number of rounds
    for i in range(len_loop):
        # first round: simple sampling
        if i == 0:
            lead_dict = {}
            for lead in ds.lead_time:
                lead_dict[f"{lead.values}"] = batch_size
            # sample events
            maxes_tot = lead_time_sample(ds,lead_dict,len_top)
            to_analyze = maxes_tot[0]
        else:
            # sample events from pool of non-chosen events, combine and sort all sampled events (from previous rounds)
            maxes_tot = lead_time_sample(ds,lead_dict,len_top,mem_sample=maxes_tot[1])
            combed = xr.combine_nested([to_analyze,maxes_tot[0]],concat_dim="event")
            to_analyze = combed.sortby(combed,ascending=False)
        #scoring
        scores[i] = score_ds(to_analyze[0:len_top],top_list)
        #find allocation for next round
        if alloc_type == "Basic":
            lead_dict = find_alloc(to_analyze[0:len_alloc],batch_size)
        elif alloc_type == "Random":
            lead_dict = find_random_alloc(len_alloc*batch_size,ds.lead_time)
        elif alloc_type == "Weighted":
            lead_dict = find_alloc_weighted(to_analyze,len_alloc,batch_size)    
        else:
            print("input valid score type")
            break
    return scores

def score_algo(ds,len_loop,batch_size,len_top,len_alloc,bootstrap,alloc_type="Random"):
    """Perform scoring algo for dataset ds, scoring according to its ground truth, and performing a bootstrap for the result"""
    #find ground truth
    top_runs = rank_tops(ds,top=len_top) # top len_top runs in boosted ensemble
    top_list = [] #same, in form of a list
    for top in top_runs:
        top_list.append(f'{top.lead_time.values}:{top.member.values}')
    # run a sampling, scoring and allocating loop, nb of times = bootstrap
    scores = [np.zeros(len_loop) for i in range(bootstrap)]
    for bt in tqdm(range(bootstrap)):
        scores[bt] =  sample_score_alloc(ds,len_loop,batch_size,len_top,top_list,len_alloc,alloc_type=alloc_type)
    score_list = [np.transpose(scores)[i] for i in range(len(np.transpose(scores)))]
    return score_list

def score_diff_config(ds,n_top,n_alloc,n_batch,len_loop,bootstrap,restrict=False):
    """takes ds_boost containging different cases, and runs the scoring algo for a range of different configurations. n_top, n_alloc, and len_top are lists of the numbers wanted to loop over.Saves the output in csv files."""
    # varying len_top (length of ground truth list - affects score only)
    for len_top in n_top:
        print(f"Top ground truth length {len_top}")
        # varying len_alloc (how many top performing events will be used for allocation in next round
        for len_alloc in n_alloc:
            print(f"Allocation length {len_alloc}")
            # varying batch size (for each top performing event, how many new events to sample
            for batch_size in n_batch:
                print(f"Batch size {batch_size}")
                #allocating in three different ways
                alloc_types = ["Random","Basic","Weighted"]
                #writing output to csv
                save_file = f"../outputs/csvs/score_{case}_batch{batch_size}_alloc{len_alloc}_top{len_top}.csv"
                if restrict == True:
                    save_file = f"../outputs/csvs/score_{case}_batch{batch_size}_alloc{len_alloc}_top{len_top}_restricted.csv"
                with open(save_file,"w") as f:
                    wrt = csv.writer(f)
                    header = [f"Round {x}" for x in range(1,len_loop+1)]
                    header.insert(0, "Allocation_type") 
                    wrt.writerow(header)
                    for i,alloc in enumerate(alloc_types):
                        score = score_algo(ds,len_loop,batch_size,len_top,len_alloc,bootstrap,alloc_type=alloc)
                        score.insert(0,alloc)
                        wrt.writerow(score)
                    #also write total number of events to csv
                    total_size = np.zeros(len_loop)
                    total_size[0] = int(len(ds.lead_time)*batch_size)
                    for i in range(1,len_loop):
                        total_size[i] = int(total_size[i-1] + len_alloc*batch_size)
                    total_size = list(total_size)
                    total_size.insert(0,"Total_size")
                    wrt.writerow(total_size)
    return None