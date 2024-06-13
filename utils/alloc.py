import numpy as np

def find_alloc_static(top_events,n_top,n_batch):
    """allocates amount of new samples to draw for each lead time, by giving n_batch number of new samples for each event in that lead time that was in the top selection"""
    rank_list = top_events[0:n_top] #top events
    occ_per_lead_ID = {} # dict containing amount of new realizations per lead time
    # need to allocate n_batch/n_top per top event, but can only allocate round numbers. excess will be distributated among the top of the top
    new_batch = int(n_batch/n_top)
    diff = n_batch - new_batch*n_top
    for i,mx in enumerate(rank_list):
        #distribute excess (if there is any)
        if diff > 0:
            allocate_here = new_batch + 1
            diff -= 1
        else:
            allocate_here = new_batch
        #allocate new realizationa
        if f"{mx.lead_ID.values}" in occ_per_lead_ID.keys():
            occ_per_lead_ID[f"{mx.lead_ID.values}"] += allocate_here
        else:
            occ_per_lead_ID[f"{mx.lead_ID.values}"] = allocate_here
    if diff > 0:
        occ_per_lead_ID[f"{rank_list[0].lead_ID.values}"] += diff
    return occ_per_lead_ID

def find_alloc_weighted(top_events,n_top,n_batch):
    """allocates amount of new samples to draw for each lead time, by weighting the top runs by how far they are from the top 1 event"""
    rank_list = top_events[0:n_top] #top events 
    occ_per_lead_ID = {}
    if len(rank_list) > 1:
        total_dist = rank_list[0]-rank_list[-1]
        if total_dist == 0 : # if all values are the same
            return find_alloc_static(top_events,n_top,n_batch)
        relative_dists = np.zeros(n_top)
        # find the ratio of relative distande/total distance
        for i,rk in enumerate(rank_list):
            relative_dists[i] = (rk - rank_list[-1])
            
        # normalize weights so total new allocated round correspond ~ to n_batch
        weights = [int(r*(n_batch)/sum(relative_dists)) for r in relative_dists]
        # add 1 more sample to allocate for each weight until total new allocated round correspond exactly to n_batch
        diff = n_batch-sum(weights)
        # allocate to dict of lead times
        for i, rk in enumerate(rank_list):
            if diff > 0:
                allocate_here = weights[i] + 1
                diff -= 1
            else:
                allocate_here = weights[i]
            if f"{rk.lead_ID.values}" in occ_per_lead_ID.keys():
                occ_per_lead_ID[f"{rk.lead_ID.values}"] += allocate_here
            else:
                occ_per_lead_ID[f"{rk.lead_ID.values}"] = allocate_here
        if diff > 0:
            occ_per_lead_ID[f"{rank_list[0].lead_ID.values}"] += diff 
    else:
        occ_per_lead_ID[f"{rank_list[0].lead_ID.values}"] = n_batch
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

def find_alloc(alloc_type,lead_IDs,top_events,n_top,n_batch):
    """TODO: write description"""
    #find allocation for next round
    if alloc_type == "Static":
        lead_dict = find_alloc_static(top_events,n_top,n_batch)
    elif alloc_type == "Random":
        lead_dict = find_random_alloc(n_batch,lead_IDs)
    elif alloc_type == "Weighted":
        lead_dict = find_alloc_weighted(top_events,n_top,n_batch)    
    else:
        print("input valid score type")
    return lead_dict

