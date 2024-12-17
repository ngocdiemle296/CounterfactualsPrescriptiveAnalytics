import pandas as pd
import numpy as np
from collections import Counter
from dateutil import parser

def check_loop(list_activities):
    # Check if traces contain repeated activities
    temp = {**Counter(list_activities)}
    if all(i == 1 for i in list(temp.values())): #No Loop
        return(0)
    else:
        return(1)
    
def get_activities(df, id, case_id_name, activity_column_name):
    # Return list of activities for each trace
    trace_df = df[df[case_id_name] == id]
    list_activities = trace_df[activity_column_name] 
    return list_activities

def get_resources(df, id, activity, activity_column_name, resource_column_name, case_id_name):
    # Extract resource or list of resources perform a certain activity
    trace_df = df[df[case_id_name] == id]
    res = trace_df[trace_df[activity_column_name] == activity][resource_column_name].tolist()
    return res

def finding_ids_with_loop(trace_ids, df, case_id_name, activity_column_name):
    # Return list of trace ids those have repeated activities
    ids_with_loop = []
    for id in trace_ids:
        list_activities = get_activities(df, id, case_id_name, activity_column_name)
        if check_loop(list_activities) == 0: #No Loop
            continue
        else:
            ids_with_loop.append(id)
    return ids_with_loop

def act_with_res(df, activity_column_name, resource_column_name):
    # Return list of resources performing activities in event log
    # Output: {act_1: [res_1], act_2: [res_2]}
    list_activities = df[activity_column_name].unique()
    result = {}
    for act in list_activities:
        res = df[df[activity_column_name] == act][resource_column_name].unique().tolist()
        if "missing" in res:
            res.remove("missing")
        result[act] = res
    return result

def res_freq(df, lst_activities, list_act_res, activity_column_name, resource_column_name):
    # Counting resource frequency performing list of activities
    # Output: {act: {res: freq}}
    res_freq = {}
    for act in lst_activities:
        lst_res = list_act_res[act]
        res_freq[act] = {}
        for res in lst_res:
            freq = df[(df[activity_column_name] == act) & (df[resource_column_name] == res)].shape[0]
            res_freq[act][res] = freq
        res_freq[act] = dict(sorted(res_freq[act].items(), key=lambda item: -item[1])) # Sorted with descending order
    return res_freq

def filter_res(act_res_freq, list_activities, filter_rate):
    # Filter resources based on contribution
    # Return list of resources have lower performance based on the filter rate
    # Requirement: "act_res_freq" has to order in DESCENDING order
    # Output: {act: [res_to_remove]}
    res_to_remove = {}
    for act in list_activities:
        sum_value = sum(act_res_freq[act].values()) 
        threshold = filter_rate*sum_value
        lst_res = list(act_res_freq[act].keys()) # Extract list of resources performing that activity
        res_to_remove[act] = []
        cummulate = 0
        for res in lst_res:
            cont = act_res_freq[act][res] 
            if cummulate <= threshold:
                cummulate += cont
            else:
                res_to_remove[act].append(res)
    return res_to_remove

def find_normal_res(list_activities, list_act_res, res_to_remove):
    result = {}
    for act in list_activities:
        result[act] = list(np.setdiff1d(list_act_res[act], res_to_remove[act])) #R\R-
    return result









# NOT USE!!
def res_with_loop(df, trace_ids_with_loop): #Confusing
    result = {}
    for id in trace_ids_with_loop:
        list_activities = get_activities(df, id)
        temp = {**Counter(list_activities)} # Extract activity frequency
        act_to_loop = [k for k in temp.keys() if temp[k] > 1] # List of repeated activities
        for act in act_to_loop:
            if act not in result.keys():
                res = get_resources(df, id, act, activity_column_name, resource_column_name)
                result[act] = [res[0]] # Getting first resource 
            else: 
                res = get_resources(df, id, act, activity_column_name, resource_column_name)
                res_1 = [res[0]]
                res_2 = result[act]
                result[act] = list(set(res_1 + res_2))
    return result



def get_res_of_loop(trace_df, activity_column_name, resource_column_name):
    # Return list of resources perform repeated activity for each trace
    list_activities = trace_df[activity_column_name] # Get list of activities
    temp = {**Counter(list_activities)} # Extract activity frequency
    act_to_loop = [k for k in temp.keys() if temp[k] > 1] # List of repeated activities
    res = {}
    for act in act_to_loop:
        res[act] = trace_df[trace_df[activity_column_name] == act][resource_column_name].tolist()
        res[act].pop(0) # Remove first resource (second time activity is performed is considered as loop => Counting from second resource)
    return res