import utils as ut 
import xarray as xr
import numpy as np
import datetime as dt
from tqdm import tqdm
import csv
from numpy.random import default_rng,randint
rng = default_rng()

def rank_tops(ds, top=50,dims=('member', 'lead_time'),ret_full = False):
    """takes a dataset ds and finds the highest value events over dimensions dim, of length top. Returns a dataset with only these events"""
    ds_stack=ds.stack(event=dims)
    top_runs = ds_stack.sortby(ds_stack,ascending=False).dropna(dim="event").isel(event=slice(0,top))
    if ret_full == True:
        return top_runs
    else:
        return top_runs.min().values

def lead_ID_sample(ds,lead_dict,mem_sample=False):
    """samples events from a dataset ds. Batch size is determined for each lead time in the dict lead_dict. mem_sample is optionally included if you want another set to draw from than just the members of the ds. Returns the highest members (length =len_top), and a dict of events that weren't chosen"""
    maxes = [] #list of sampled datasets for each lead time
    non_chosen = {} #dict of non-sampled members (per lead time)
    for lead_ID in lead_dict:
        #choose sample to draw from
        if mem_sample == False:
            sample = ds.member.values
        else:
            sample = mem_sample[lead_ID]
        #choose batch size
        batch_size = lead_dict[lead_ID]
        if batch_size > len(sample):
            batch_size = len(sample)
        #draw batch from sample of size batch_size
        batch = rng.choice(sample, size=batch_size,replace=False)
        # only keep maximum values over time
        maxes_batch = ds.sel(lead_ID=eval(lead_ID),member=batch)
        maxes.append(maxes_batch)   
        # non-sampled members
        non_chosen[lead_ID] = list(set(sample) - set(batch))
    maxes_tot = xr.concat(maxes,dim="lead_ID")
    # find top runs within all the sampled members (across lead times -> batch_size*len(lead_time))
    return rank_tops(maxes_tot,dims=ds.dims,ret_full = True), non_chosen

def score_max(maxes_tot):
    """TODO: write this"""
    return maxes_tot.max().values

def score_sum(maxes_tot,top=10):
    """TODO: write this"""
    top_scores = maxes_tot.sortby(maxes_tot,ascending = False)[0:top]
    return top_scores.sum().values

def find_alloc_static(maxes_tot,len_alloc,batch_size):
    """allocates amount of new samples to draw for each lead time, by giving batch_size number of new samples for each event in that lead time that was in the top selection"""
    rank_list = maxes_tot[0:len_alloc] #top events
    occ_per_lead_ID = {}
    for mx in maxes_tot:
        if f"{mx.lead_ID.values}" in occ_per_lead_ID.keys():
            occ_per_lead_ID[f"{mx.lead_ID.values}"] += batch_size
        else:
            occ_per_lead_ID[f"{mx.lead_ID.values}"] = batch_size
    return occ_per_lead_ID

def find_alloc_weighted(maxes_tot,len_alloc,batch_size):
    """allocates amount of new samples to draw for each lead time, by weighting the top runs by how far they are from the rank 1 event"""
    rank_list = maxes_tot[0:len_alloc] #top events
    occ_per_lead_ID = {}
    if len(rank_list) > 1:
        total_dist = rank_list[0]-rank_list[-1]
        if total_dist == 0 : # if all values are the same
            return find_alloc_static(maxes_tot, len_alloc,batch_size)
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
            if f"{rk.lead_ID.values}" in occ_per_lead_ID.keys():
                occ_per_lead_ID[f"{rk.lead_ID.values}"] += weights[i]
            else:
                occ_per_lead_ID[f"{rk.lead_ID.values}"] = weights[i]        
    else:
        occ_per_lead_ID[f"{rank_list[0].lead_ID.values}"] = batch_size
    return occ_per_lead_ID

def find_random_alloc(size, lead_IDs):
    """allocates amount of new samples to draw for each lead time, by drawing random samples from any lead time, in total amounting to size"""
    occ_per_lead_ID = {}
    for i in range(size):
        lead_ID = rng.choice(lead_IDs, size=1,replace=True)[0]
        if f"{lead_ID}" in occ_per_lead_ID:
            occ_per_lead_ID[f"{lead_ID}"] += 1
        else:
            occ_per_lead_ID[f"{lead_ID}"] = 1
    for ld in lead_IDs.values:
        if f"{ld}" not in occ_per_lead_ID.keys():
            occ_per_lead_ID[f"{ld}"] = 0
    return occ_per_lead_ID

def find_alloc(alloc_type,lead_IDs,to_analyze,len_alloc,batch_size):
    #find allocation for next round
    if alloc_type == "Static":
        lead_dict = find_alloc_static(to_analyze,len_alloc,batch_size)
    elif alloc_type == "Random":
        lead_dict = find_random_alloc(len_alloc*batch_size,lead_IDs)
    elif alloc_type == "Weighted":
        lead_dict = find_alloc_weighted(to_analyze,len_alloc,batch_size)    
    else:
        print("input valid score type")
    return lead_dict

def screening(ds,batch_size_start):
    lead_dict = {}
    for lead in ds.lead_ID:
        lead_dict[f"{lead.values}"] = batch_size_start
    # sample events
    maxes_tot = lead_ID_sample(ds,lead_dict)
    return maxes_tot

def sample_score_alloc(ds,lead_dict,maxes_tot,len_loop,batch_size,len_alloc,alloc_type="Random"):
    """takes a dataset ds and performs sampling, scoring and allocation for len_loop rounds"""
    scores = np.zeros(len_loop)
    lead_dicts = []
    to_analyze = maxes_tot[0]
    # loop over number of rounds
    for i in range(len_loop):
        # sample events from pool of non-chosen events, combine and sort all sampled events (from previous rounds)
        maxes_tot = lead_ID_sample(ds,lead_dict,mem_sample=maxes_tot[1])
        combed = xr.combine_nested([to_analyze,maxes_tot[0]],concat_dim="event")
        to_analyze = combed.sortby(combed,ascending=False)
        #scoring
        scores[i] = score_sum(to_analyze)
        # what lead times were allocated
        lead_dicts.append(lead_dict)
        #allocation for next round
        if i < len_loop - 1:
            lead_dict = find_alloc(alloc_type,ds.lead_ID,to_analyze,len_alloc,batch_size)
    return scores, lead_dicts

def score_algo(ds,len_loop,batch_size,batch_start_size,len_alloc,bootstrap):
    """Perform scoring algo for dataset ds, scoring according to its ground truth, and performing a bootstrap for the result"""
    # run a sampling, scoring and allocating loop, nb of times = bootstrap
    score_info_boot = []
    lead_list = [f"{ld}" for ld in ds.lead_ID.values] #list of of all lead IDs for dataset
    for bt in tqdm(range(bootstrap)):
        # screening phase (similar for all three allocation algorithms)
        results_screening = screening(ds,batch_start_size)
        scores_screening = score_sum(results_screening[0])
        # find scores and chosen leads for different allocation types
        alloc_types = ["Random","Static","Weighted"]
        score_info = []
        for alloc in alloc_types:
            lead_dict= find_alloc(alloc,ds.lead_ID,results_screening[0],len_alloc,batch_size)
            score, lead_ID_all_rounds =  sample_score_alloc(ds,
                                                           lead_dict,
                                                           results_screening,
                                                           len_loop,
                                                           batch_size,
                                                           len_alloc,
                                                           alloc_type=alloc)
            # information on which lead times were chosen
            lead_data = np.zeros((len(lead_ID_all_rounds), len(lead_list)))
            for i,round in enumerate(lead_ID_all_rounds):
                for lead_ID in round:
                    j = lead_list.index(lead_ID)
                    lead_data[i,j] = round[lead_ID]
            # store all info in dataset
            score_info.append(xr.Dataset(
                {
                    "chosen_leads": (["round", "lead_ID"], lead_data),
                    "score": (["round"], score),
                },
                coords={
                    "round": range(0,len(lead_ID_all_rounds)),
                    "lead_ID": lead_list,
                },
            ))
        score_info_alloc_type = ut.concat_to_ds(score_info,"alloc_type",alloc_types)
        score_info_alloc_type["screening_score"] = scores_screening
        score_info_boot.append(score_info_alloc_type)
    score_info = xr.concat(score_info_boot,dim="bootstrap")
    return score_info

def score_diff_config(ds,n_alloc,n_batch,n_batch_start,len_loop,bootstrap,area,score_typ="temp_max"):
    """takes ds_boost containging different cases, and runs the scoring algo for a range of different configurations. n_top, n_alloc, and len_top are lists of the numbers wanted to loop over.Saves the output in csv files."""
    score_info_batch_start = [] 
    # varying batch size for screening phase
    for batch_start_size in n_batch_start:
        print(f"Batch size for screening = {batch_start_size}")
        # varying len_alloc (how many top performing events will be used for allocation in next round)
        score_info_len_alloc = []
        for len_alloc in n_alloc:
            print(f"Allocation length {len_alloc}")
            # varying batch size (for each top performing event, how many new events to sample)
            scores_batch = []
            for batch_size in n_batch:
                print(f"Batch size {batch_size}")
                #calculating aggregated scores for all three allocation types
                score_info = score_algo(ds,
                                   len_loop,
                                   batch_size,
                                   batch_start_size,     
                                   len_alloc,
                                   bootstrap,
                                   )
                scores_batch.append(score_info)
            score_info_len_alloc.append(ut.concat_to_ds(scores_batch,"batch_size",n_batch))
        score_info_batch_start.append(ut.concat_to_ds(score_info_len_alloc,"allocation_size",n_alloc))
    score_info = ut.concat_to_ds(score_info_batch_start,"start_batch_size",n_batch_start)
    score_info.to_netcdf(f"../outputs/score_info/score_info_{area}_{score_typ}.nc")
    return None