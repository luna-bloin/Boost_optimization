import utils as ut 
import alloc as ac
import xarray as xr
import numpy as np
from tqdm import tqdm
from numpy.random import default_rng,randint
rng = default_rng()


def lead_ID_sample(ds,lead_dict,mem_sample=False):
    """Samples events from a dataset ds without replacement (between rounds). Batch size is determined for each lead time in the dict lead_dict.
        :param ds: dataset that contains boosted events with dimensions lead_ID (either just lead time or stacked lead_time and case)
        :param lead_dict: dictionary of lead_IDs, with information on how many new realizations to sample (based on screening)
        :param mem_sample: optional. a limited sample to draw from (to not draw same event twice between rounds)
        returns [dataset of sampled events, dictionary of non-chosen events]"""
    samples = [] #list of sampled datasets for each lead time
    non_chosen = {} #dict of non-sampled members (per lead time)
    lead_ID_0_size = [] #list of lead_IDs with size 0 
    for lead_ID in lead_dict:
        #choose sample to draw from
        if mem_sample == False:
            available_events = ds.member.values
        else:
            available_events = mem_sample[lead_ID]
        #choose batch size
        batch_size = lead_dict[lead_ID]
        if batch_size > len(available_events):
            batch_size = len(available_events)
        if batch_size ==0:
            lead_ID_0_size.append(lead_ID)
            non_chosen[lead_ID] = available_events
            continue
        #draw batch from sample of size batch_size
        batch = rng.choice(available_events, size=batch_size,replace=False)
        # only keep maximum values over time
        sample_batch = ds.sel(lead_ID=lead_ID,member=batch)
        sample_da = xr.DataArray(sample_batch.values, dims="value").pad(value=(0,len(ds.member.values)-len(batch)))
        samples.append(sample_da)   
        # non-sampled members
        non_chosen[lead_ID] = list(set(available_events) - set(batch))
    for ld in lead_ID_0_size:
        del lead_dict[ld]
    samples = xr.concat(samples,dim="lead_ID")
    samples["lead_ID"] = list(lead_dict.keys())
    return samples.stack(event=("lead_ID","value")).dropna(dim="event"), non_chosen

def lead_ID_sample_replace(ds,lead_dict):
    """Samples events from a dataset ds with replacement (between rounds). Batch size is determined for each lead time in the dict lead_dict.
        :param ds: dataset that contains boosted events with dimensions lead_ID (either just lead time or stacked lead_time and case)
        :param lead_dict: dictionary of lead_IDs, with information on how many new realizations to sample (based on screening)
        returns [dataset of sampled events, dictionary of non-chosen events]"""
    samples = [] #list of sampled datasets for each lead time
    non_chosen = {} #dict of non-sampled members (per lead time)
    lead_ID_0_size = [] #list of lead_IDs with size 0 
    for lead_ID in lead_dict:
        all_lead_IDs = ds.sel(lead_ID=lead_ID).member.values
        #choose batch size
        batch_size = lead_dict[lead_ID]
        if batch_size > len(all_lead_IDs):
            batch_size = len(all_lead_IDs)
        if batch_size ==0:
            lead_ID_0_size.append(lead_ID)
            continue
        #draw batch from sample of size batch_size
        batch = rng.choice(all_lead_IDs, size=batch_size,replace=False)
        # only keep maximum values over time
        sample_batch = ds.sel(lead_ID=lead_ID,member=batch)
        sample_da = xr.DataArray(sample_batch.values, dims="value").pad(value=(0,len(ds.member.values)-len(batch)))
        samples.append(sample_da)   
        non_chosen[lead_ID] = all_lead_IDs # all are non-chosen, since we are replacing events (i.e. we can choose an event several times)
    for ld in lead_ID_0_size:
        del lead_dict[ld]
    samples = xr.concat(samples,dim="lead_ID")
    samples["lead_ID"] = list(lead_dict.keys())
    return samples.stack(event=("lead_ID","value")).dropna(dim="event"), non_chosen


def screening(ds,n_batch_start,replace=False):
    """Performs screening (samples using blind boosting)
        :param ds: dataset that contains boosted events with dimensions lead_ID (either just lead time or stacked lead_time and case)
        :param n_batch_start: value of n_batch_start (batch size for screening)
        :param replace: whether or not to replace event when randomly sampled
        returns [dataset of sampled events, dictionary of non-chosen events]"""
    lead_dict = {}
    for lead in ds.lead_ID:
        lead_dict[f"{lead.values}"] = n_batch_start
    # sample events
    if replace == True:
        sampled = lead_ID_sample_replace(ds,lead_dict)
    else:
        sampled = lead_ID_sample(ds,lead_dict)
    return sampled,lead_dict


def sample_score_alloc(ds,lead_dict,sampled,n_top,n_batch,len_loop,alloc_type="Random",replace=False):
    """Randomly samples n_batch realizations, scores them and allocates for all rounds of the allocation algorithm  
        :param ds: dataset that contains boosted events with dimensions lead_ID (either just lead time or stacked lead_time and case)
        :param lead_dict: dictionary of lead_IDs, with information on how many new realizations to sample (based on screening)
        :param sampled: [dataset of sampled events, dictionary of non-chosen events]
        :param n_top: values of n_top (length of top events to use for allocation)
        :param n_batch: value of n_batch (batch size for each allocation round)
        :param len_loop: how many rounds of allocation to perform
        :param alloc_type: type of allocation ("Random", "Static"or "Weighted")
        :param replace: whether or not to replace event when randomly sampled
        returns the resulting dataset"""
    scores = []
    lead_dicts = []
    to_analyze = sampled[0]
    # loop over number of rounds
    for i in range(len_loop):
        # sample events from pool of non-chosen events, combine and sort all sampled events (from previous rounds)
        if replace == True:
            sampled = lead_ID_sample_replace(ds,lead_dict)
        else:
            sampled = lead_ID_sample(ds,lead_dict,mem_sample=sampled[1])
        combed = xr.combine_nested([to_analyze,sampled[0]],concat_dim="event")
        to_analyze = combed.sortby(combed,ascending=False)
        #scoring
        scores.append(sampled[0].values)
        # what lead times were allocated
        lead_dicts.append(lead_dict)
        #allocation for next round
        if i < len_loop - 1:
            lead_dict = ac.find_alloc(alloc_type,ds.lead_ID,to_analyze,n_top,n_batch)
    return scores, lead_dicts

def score_algo(ds,n_top,n_batch,n_batch_start,len_loop,bootstrap,replace = False):
    """Runs the screening + allocation algorithm for set parameters, in a bootstrapped way. builds an xarray dataset for the results. 
        :param ds: dataset that contains boosted events with dimensions lead_ID (either just lead time or stacked lead_time and case)
        :param n_top: values of n_top (length of top events to use for allocation)
        :param n_batch: value of n_batch (batch size for each allocation round)
        :param n_batch_start: value of n_batch_start (batch size for screening round)
        :param len_loop: how many rounds of allocation to perform
        :param bootstrap: how many times are you bootstrapping the process
        :param replace: whether or not to replace event when randomly sampled
        returns the resulting dataset"""
    # run a sampling, scoring and allocating loop, nb of times = bootstrap
    score_info_boot = []
    lead_list = [f"{ld}" for ld in ds.lead_ID.values] #list of of all lead IDs for dataset
    for bt in tqdm(range(bootstrap)):
        # screening phase (similar for all three allocation algorithms)
        results_screening, lead_dict_screening = screening(ds,n_batch_start,replace=replace)
        # calculate scores
        scores_screening = results_screening[0].values
        # find scores and chosen leads for different allocation types
        alloc_types = ["Static","Weighted"]
        score_info = []
        for alloc in alloc_types:
            lead_dict= ac.find_alloc(alloc,ds.lead_ID,results_screening[0].sortby(results_screening[0],ascending=False),n_top,n_batch)
            scores, lead_ID_all_rounds =  sample_score_alloc(ds,
                                                           lead_dict,
                                                           results_screening,
                                                           n_top,
                                                           n_batch,
                                                           len_loop,
                                                           alloc_type=alloc,
                                                           replace=replace)
            #add screening scores to the beginning of list of scores per round
            scores.insert(0,scores_screening)
            lead_ID_all_rounds.insert(0,lead_dict_screening)
            # information on which lead times were chosen
            lead_data = np.zeros((len(lead_ID_all_rounds), len(lead_list)))
            for i,round in enumerate(lead_ID_all_rounds):
                for lead_ID in round:
                    j = lead_list.index(lead_ID)
                    lead_data[i,j] = round[lead_ID]
            #make sure all arrays have same length
            to_pad = np.max([len(sc) for sc in scores])
            padded_score = [np.pad(arr, (0, to_pad - len(arr)), constant_values=np.nan) for arr in scores]
            # store all info in dataset
            score_info.append(xr.Dataset(
                {
                    "chosen_leads": (["round", "lead_ID"], lead_data),
                    "score": (["round","distribution_value"], padded_score),
                },
                coords={
                    "round": range(len(scores)),
                    "lead_ID": lead_list,
                    "distribution_value":range(to_pad)
                },
            ))
        score_info_alloc_type = ut.concat_to_ds(score_info,"alloc_type",alloc_types)
        score_info_alloc_type["screening_score"] = scores_screening
        score_info_boot.append(score_info_alloc_type)
    score_info = xr.concat(score_info_boot,dim="bootstrap")
    return score_info

def score_diff_config(ds,n_tops,n_batchs,n_batch_starts,len_loop,bootstrap,save_info,replace=False):
    """Runs the screening + allocation algorithm for a range of parameters, in a bootstrapped way. saves results as .nc file in folder outputs. 
        :param ds: dataset that contains boosted events with dimensions lead_ID (either just lead time or stacked lead_time and case)
        :param n_tops: list of values of n_top (length of top events to use for allocation
        :param n_batchs: list of values of n_batch (batch size for each allocation round)
        :param n_batch_starts: list of values of n_batch_start (batch size for screening round)
        :param len_loop: how many rounds of allocation to perform
        :param bootstrap: how many times are you bootstrapping the process
        :param save_info: what to save results .nc file
        :param replace: whether or not to replace event when randomly sampled
        returns None"""
    score_info_batch_start = [] 
    # varying batch size for screening phase
    for n_batch_start in n_batch_starts:
        print(f"Batch size for screening = {n_batch_start}")
        # varying batch size (for each top performing event, how many new events to sample)
        scores_batch = []
        for n_batch in n_batchs:
            print(f"Batch size {n_batch}")
            # varying n_top (how many top performing events will be used for allocation in next round)
            score_info_n_top = []
            n_top_used = []
            for n_top in n_tops:
                if n_top > n_batch:
                    break
                print(f"Allocation length {n_top}")
                n_top_used.append(n_top)
                #calculating aggregated scores for all three allocation types
                score_info = score_algo(ds,
                                   n_top,
                                   n_batch,
                                   n_batch_start,     
                                   len_loop,
                                   bootstrap,
                                   replace = replace
                                   )
                score_info_n_top.append(score_info)
            scores_batch.append(ut.concat_to_ds(score_info_n_top,"top_length",n_top_used))
        score_info_batch_start.append(ut.concat_to_ds(scores_batch,"batch_size",n_batchs))
    score_info = ut.concat_to_ds(score_info_batch_start,"start_batch_size",n_batch_starts)
    score_info.to_netcdf(f"../outputs/score_info/score_info_{save_info}_replace{replace}.nc")
    return None