import utils as ut 
import xarray as xr
import numpy as np
import datetime as dt
from tqdm import tqdm
import csv
from numpy.random import default_rng,randint
rng = default_rng()

def create_event(ds,mem_typ,peak_date,anomaly=False):
    """returns the 5-day running mean maximum temperature of each member in ds, within 5 days of parent peak. If anomaly is not bool, pass it as the dataset of mean climatology, to create anomaly dataset"""
    orig_peak = peak_date.dayofyear.values
    dates = ds.start_date.values
    lead_times = [int(ut.to_dt(dat).timetuple().tm_yday-orig_peak) for dat in dates]
    ds["start_date"] = lead_times
    ds=ut.to_cel(ds).rename({"start_date":"lead_time"})
    if type(anomaly) != bool:
        ds = ds.groupby("time.dayofyear")- anomaly
    max_in_window =  ds.TREFHTMX_x5d.sel(member = range(1,int(mem_typ) + 1)).sel(time = slice(peak_date - dt.timedelta(days = 5), peak_date + dt.timedelta(days = 5))).max("time")
    return max_in_window
        

def rank_tops(ds, top=50,dims=('member', 'lead_time'),ret_full = False):
    """takes a dataset ds and finds the highest value events over dimensions dim, of length top. Returns a dataset with only these events"""
    ds_stack=ds.stack(event=dims)
    top_runs = ds_stack.sortby(ds_stack,ascending=False).dropna(dim="event").isel(event=slice(0,top))
    if ret_full == True:
        return top_runs
    else:
        return top_runs.min().values

def lead_time_sample(ds,lead_dict,len_top,mem_sample=False):
    """samples events from a dataset ds. Batch size is determined for each lead time in the dict lead_dict. mem_sample is optionally included if you want another set to draw from than just the members of the ds. Returns the highest members (length =len_top), and a dict of events that weren't chosen"""
    maxes = [] #list of sampled datasets for each lead time
    non_chosen = {} #dict of non-sampled members (per lead time)
    for lead_time in lead_dict:
        #choose sample to draw from
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
        maxes_batch = ds.sel(lead_time=eval(lead_time),member=batch)
        maxes.append(maxes_batch)   
        # non-sampled members
        non_chosen[lead_time] = list(set(sample) - set(batch))
    maxes_tot = xr.concat(maxes,dim="lead_time")
    # find top runs within all the sampled members (across lead times -> batch_size*len(lead_time))
    return rank_tops(maxes_tot,top=len_top,dims=ds.dims,ret_full = True),non_chosen

def score_ds(maxes_tot,top_thresh):
    """scores the sampled events by comparing them to a ground truth. Returns #events found in ground truth/length of sample"""
    score = 0
    for mx in maxes_tot:
        if mx >= top_thresh:
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
        # normalize weights so total new allocated round correspond ~ to len_alloc*batch_size
        weights = [int(r*(len_alloc*batch_size)/sum(relative_dists)) for r in relative_dists]
        # add 1 more sample to allocate for each weight until total new allocated round correspond exactly to len_alloc*batch_size
        i = 0
        while (len_alloc*batch_size) > sum(weights):
            weights[i] += 1
            if i < len(weights):
                i + 1
            else:
                i = 0
            i + 1
        # allocate to dict of lead times
        for i, rk in enumerate(rank_list):
            if f"{rk.lead_time.values}" in occ_per_lead_time.keys():
                occ_per_lead_time[f"{rk.lead_time.values}"] += weights[i]
            else:
                occ_per_lead_time[f"{rk.lead_time.values}"] = weights[i]        
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

def sample_score_alloc(ds,len_loop,batch_size,len_top,top_thresh,len_alloc,alloc_type="Random"):
    """takes a dataset ds and performs sampling, scoring and allocation for len_loop rounds"""
    scores = np.zeros(len_loop)
    lead_dicts = []
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
        scores[i] = score_ds(to_analyze[0:len_top],top_thresh)
        # what lead times were allocated
        lead_dicts.append(lead_dict)
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
    return scores, lead_dicts

def score_algo(ds,len_loop,batch_size,len_top,len_alloc,bootstrap,top_thresh,alloc_type="Random"):
    """Perform scoring algo for dataset ds, scoring according to its ground truth, and performing a bootstrap for the result"""
    # run a sampling, scoring and allocating loop, nb of times = bootstrap
    #scores = [np.zeros(len_loop) for i in range(bootstrap)]
    score_info = []
    for bt in tqdm(range(bootstrap)):
        # find scores and chosen leads
        score,lead_time_all_rounds =  sample_score_alloc(ds,len_loop,batch_size,len_top,top_thresh,len_alloc,alloc_type=alloc_type)
        # information on which lead times were chosen
        lead_times = [ld for ld in lead_time_all_rounds[0].keys()] #since round 1 spans all lead times, we get the lead time info from here
        lead_data = np.zeros((len(lead_time_all_rounds), len(lead_times)))
        for i,round in enumerate(lead_time_all_rounds):
            for lead_time in round:
                j = lead_times.index(lead_time)
                lead_data[i,j] = round[lead_time]
        # store all info in dataset
        score_info.append(xr.Dataset(
            {
                "chosen_leads": (["round", "lead_time"], lead_data),
                "score": (["round"], score),
            },
            coords={
                "round": range(0,len(lead_time_all_rounds)),
                "lead_time": lead_times,
            },
        ))
    score_info = xr.concat(score_info,dim="bootstrap")
    #score_list = [np.transpose(scores)[i] for i in range(len(np.transpose(scores)))]
    return score_info

def score_diff_config(ds,n_top,n_alloc,n_batch,len_loop,bootstrap,save_adds,together,restrict):
    """takes ds_boost containging different cases, and runs the scoring algo for a range of different configurations. n_top, n_alloc, and len_top are lists of the numbers wanted to loop over.Saves the output in csv files."""
    # varying len_top (length of ground truth list - affects score only)
    score_info_top = []
    # adding changes to name depending on configuration
    if together==True:
        save_adds += f"_mem_typ{len(ds.member.values)}"
        ds_calc = ds.stack(event=("lead_time","case")).rename({"lead_time":"lt","event":"lead_time"})
    else:
        ds_calc = ds
        save_adds += f"_{together}" #save case name
    if restrict == True:
        save_adds += "_restricted"
        ds_calc = ds_calc.sel(lead_time=slice(-15,10))
    for len_top in n_top:
        print(f"Top ground truth length {len_top}")
        #find ground truth
        top_thresh = rank_tops(ds,top=len_top,dims=ds.dims,ret_full = False) # top len_top runs in boosted ensemble
        # varying len_alloc (how many top performing events will be used for allocation in next round
        score_info_len_alloc = []
        for len_alloc in n_alloc:
            print(f"Allocation length {len_alloc}")
            # varying batch size (for each top performing event, how many new events to sample
            score_info_batch = []
            for batch_size in n_batch:
                print(f"Batch size {batch_size}")
                #allocating in three different ways
                alloc_types = ["Random","Basic","Weighted"]
                #calculating scores for all allocation types
                scores = []
                for i,alloc in enumerate(alloc_types):
                    score_info = score_algo(ds_calc,
                                       len_loop,
                                       batch_size,
                                       len_top,
                                       len_alloc,
                                       bootstrap,
                                       top_thresh,
                                       alloc_type=alloc)
                    scores.append(score_info)
                score_info_batch.append(ut.concat_to_ds(scores,"alloc_type",alloc_types))
            score_info_len_alloc.append(ut.concat_to_ds(score_info_batch,"batch_size",n_batch))
        score_info_top.append(ut.concat_to_ds(score_info_len_alloc,"allocation_size",n_alloc))
    score_info = ut.concat_to_ds(score_info_top,"top_size",n_top)
    score_info.to_netcdf(f"../outputs/score_info/score_info_{save_adds}.nc")
    return None