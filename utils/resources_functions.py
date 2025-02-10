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
    """
    Generates a dictionary mapping each unique activity to a list of unique resources associated with it.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        activity_column_name (str): The name of the column containing activity names.
        resource_column_name (str): The name of the column containing resource names.
        
    Returns:
        dict: A dictionary where keys are unique activities and values are lists of unique resources associated with each activity.
    """
    
    list_activities = df[activity_column_name].unique()
    result = {}
    for act in list_activities:
        res = df[df[activity_column_name] == act][resource_column_name].unique().tolist()
        if "missing" in res:
            res.remove("missing")
        result[act] = res
    return result

def res_freq(df, lst_activities, list_act_res, activity_column_name, resource_column_name):
    """
    Calculate the frequency of resources performing a list of activities.

    Parameters:
        df (pd.DataFrame): The dataframe containing the activity and resource data.
        lst_activities (list): A list of activities to consider.
        list_act_res (dict): A dictionary where keys are activities and values are lists of resources.
        activity_column_name (str): The name of the column in the dataframe that contains activity names.
        resource_column_name (str): The name of the column in the dataframe that contains resource names.

    Returns:
        dict: A nested dictionary where the keys are activities and the values are dictionaries.
              These inner dictionaries have resources as keys and their corresponding frequencies as values,
              sorted in descending order of frequency.
    """
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
    """
    Filters resources based on their contribution to activities and returns a list of resources 
    that have lower performance based on the specified filter rate.

    Parameters:
    act_res_freq (dict): A dictionary where keys are activities and values are dictionaries 
                         of resources with their corresponding frequencies. The dictionary 
                         should be ordered in descending order of frequencies.
    list_activities (list): A list of activities to be considered for filtering.
    filter_rate (float): A threshold rate (between 0 and 1) used to determine which resources 
                         to filter out based on their cumulative contribution.

    Returns:
    dict: A dictionary where keys are activities and values are lists of resources to be removed 
          based on the filter rate.
    """

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
    """
    Finds the normal resources for each activity by removing infrequent resources.
    
    Parameters:
        list_activities (list): A list of activities.
        list_act_res (dict): A dictionary where keys are activities and values are lists of resources associated with those activities.
        res_to_remove (dict): A dictionary where keys are activities and values are lists of resources to be removed from the corresponding activity's resources.
    
    Returns:
        dict: A dictionary where keys are activities and values are lists of resources after removing infrequent resources.
    """

    result = {}
    for act in list_activities:
        result[act] = list(np.setdiff1d(list_act_res[act], res_to_remove[act])) #R\R-
    return result