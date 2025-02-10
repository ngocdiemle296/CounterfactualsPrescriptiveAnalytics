import dice_ml
from dice_ml import Dice
from collections import Counter
import concurrent.futures
from interruptingcow import timeout
import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import time

from utils.transition_system import transition_system
from utils.resources_functions import get_resources, act_with_res
from utils.resources_functions import finding_ids_with_loop, act_with_res, res_freq, filter_res, find_normal_res

def create_DiCE_model(dataframe, continuous_features, outcome_name, dice_method, predictive_model, case_id_name, start_date_name, end_date_name):
    ## Create DiCE model
    data_for_dice = dataframe.drop([case_id_name, start_date_name, end_date_name, "remaining_time"], axis =1)
    data_model = dice_ml.Data(dataframe=data_for_dice,
                              continuous_features=continuous_features,
                              outcome_name=outcome_name)
    
    ml_backend = dice_ml.Model(model=predictive_model, backend="sklearn", model_type='regressor')
    method = dice_method  
    explainer = Dice(data_model, ml_backend, method=method) 
    return explainer

def filtering_resources_for_dice(train_data, activity_column_name, resource_column_name, threshold):
    # List of activities
    list_activities = list(train_data[activity_column_name].unique())
    # List of activities with resources respectively
    list_act_res = act_with_res(train_data, activity_column_name, resource_column_name)
    # Frequency of resources performing activities in the event log
    act_res_freq = res_freq(train_data, list_activities, list_act_res, activity_column_name, resource_column_name)
    # Extract list of resources with low performance
    res_to_remove = filter_res(act_res_freq, list_activities, threshold)
    # Extract list of resources with normal performance
    res_normal = find_normal_res(list_activities, list_act_res, res_to_remove)
    return res_normal


def CFE_for_a_single_query(explainer, query_instance, y_predicted, REDUCED_KPI_TIME, total_CFs, list_next_activity, range_resources, cols_to_vary):
    """
    Generates a list of Counterfactual examples (CFEs) for a query instance
    If no recommendations has found returns 0
    """
    total_time_predicted = y_predicted
    total_time_upper_bound = int(total_time_predicted * REDUCED_KPI_TIME) 

    try:
        cfe = explainer.generate_counterfactuals(query_instance,
                                                 total_CFs,
                                                 features_to_vary=cols_to_vary,
                                                 desired_range = [0, total_time_upper_bound],
                                                 permitted_range = {'NEXT_ACTIVITY': list_next_activity, 
                                                                    'NEXT_RESOURCE': range_resources})
    except:
        cfe = 0
    return cfe

def next_possible_activities(trace_history, transition_graph, WINDOW_SIZE):
    """
    Returns the list of possible next activities based on the transition graph and the trace history.
    """
    n = len(trace_history)
    pos_acts = []
    if  n <= WINDOW_SIZE:
        trace_to_compare = trace_history
        trace_to_str =  "".join(trace_to_compare)
        if trace_to_str in transition_graph.keys():
            pos_acts = transition_graph[trace_to_str]
        else:
            for ts in transition_graph.keys():
                ts_to_list = ts.split(", ")
                if (Counter(ts_to_list) == Counter(trace_to_compare)) and (ts_to_list[-1] == trace_to_compare[-1]):
                    pos_acts = transition_graph[ts]
    else:
        trace_to_compare = trace_history[-WINDOW_SIZE:] 
        for ts in transition_graph.keys():
            ts_to_list = ts.split(", ")
            if (Counter(ts_to_list) == Counter(trace_to_compare)) and (ts_to_list[-1] == trace_to_compare[-1]):
                pos_acts = transition_graph[ts]
    return list(pos_acts)

def range_res(res_normal, next_possible_activities):
    # Extract range of resources for DiCE by combining all resources of next possible activities
    new_list = []
    for act in next_possible_activities:
        list_res = res_normal[act]
        new_list = list(set(new_list + list_res))
        del list_res
    return list(set(new_list))

def evaluate_single_query_results(cfe_single_query, res_normal): 
    """
    Validating the results of CFEs by checking the compatibility
    of the next activities and resources.
    
    Parameters:
        cfe_single_query (object): An object containing the counterfactual examples generated for a single query.
        res_normal (dict): A dictionary where keys are activities and values are lists of compatible resources.
    
    Returns:
        tuple: A tuple containing:
            - int: The number of valid counterfactual examples.
            - DataFrame: A DataFrame of the valid counterfactual examples.
    """
    # Extracting only generated counterfactual examples
    cfe_single_query_df = cfe_single_query.cf_examples_list[0].final_cfs_df

    # Checking the compatibility of next activities and next resources 
    valid_idx = [] # Indices of valid CFE 
    for idx, (act, res) in enumerate(cfe_single_query_df[['NEXT_ACTIVITY', 'NEXT_RESOURCE']].values):
        if res in res_normal[act]:
            valid_idx.append(idx)
                 
    # Evaluating number of valid CFEs
    if len(valid_idx) == 0: #There is NO valid CFE for this query.
        return 0, cfe_single_query_df.iloc[valid_idx].reset_index(drop=True)
    else:
        return len(valid_idx), cfe_single_query_df.iloc[valid_idx].reset_index(drop=True)
    
def dice_recommendations(train_data, test_log, test_data, threshold, predictive_model, continuous_features, dice_method, REDUCED_KPI_TIME, TOTAL_CFS, result_df, WINDOW_SIZE, cols_to_vary, case_id_name, activity_column_name, resource_column_name, outcome_name, start_date_name, end_date_name, time_limit):
    """
    Generate DiCE recommendations for counterfactual explanations in prescriptive analytics.
    """

    test_log_ids = test_log[case_id_name].unique()
    print("Filtering resources...")
    res_normal = filtering_resources_for_dice(train_data, activity_column_name, resource_column_name, threshold)
    print("Building transition graph...")
    transition_graph = transition_system(train_data, case_id_name=case_id_name, activity_column_name=activity_column_name, window_size=WINDOW_SIZE)
    print("Creating DiCE model...")
    explainer = create_DiCE_model(train_data, continuous_features, outcome_name, dice_method, predictive_model, case_id_name, start_date_name, end_date_name)
    n = len(test_log_ids)
    result_df[case_id_name] = test_log_ids
    result_df["true_outcome"] = ""
    result_df["predicted_outcome"] = ""
    result_df["valid_cfes"] = ""
    result_df["init_next_activity"] = ""
    result_df["Next_activity"] = ""
    result_df["Next_resource"] = ""
    result_df["predicted_rec"] = ""
    print("Start generating recommendations...")
    for i, idx in enumerate(tqdm.tqdm(test_log_ids)):
        print(idx, n - 1)
        trace_df = test_log[test_log[case_id_name] == idx]
        trace_history = trace_df[activity_column_name].tolist() # List of prefix history
        current_execution = test_data[test_data[case_id_name] == idx]
        next_activity = current_execution['NEXT_ACTIVITY'].values[0] # Suppose to be next activity
        next_resource = current_execution['NEXT_RESOURCE'].values[0] # Suppose to be next resource
        true_outcome = current_execution[outcome_name].values[0]
        result_df['init_next_activity'][i] = next_activity
        result_df['true_outcome'][i] = true_outcome
        
        query_instance = current_execution.drop([case_id_name, outcome_name, start_date_name, end_date_name, "remaining_time"], axis = 1)

        predicted_time =  int(predictive_model.predict(query_instance)[0])
        result_df['predicted_outcome'][i] = predicted_time
        
        # Possible next activities and resources
        next_possible_activity = next_possible_activities(trace_history, transition_graph, WINDOW_SIZE)
        range_next_resources = range_res(res_normal, next_possible_activity)         
        
        try:
            cfe_single_query = func_timeout(time_limit, CFE_for_a_single_query, args= (explainer, query_instance, predicted_time, REDUCED_KPI_TIME, TOTAL_CFS, 
                                                          next_possible_activity, range_next_resources, cols_to_vary))
        except:
            cfe_single_query = 0

      
        # DiCE CAN generate recommendation
        if cfe_single_query != 0: 
            # Evaluate for a single query 
            n_valid_cfes, cfe_single_query_evaluate = evaluate_single_query_results(cfe_single_query, res_normal)

            if n_valid_cfes != 0: 
                result_df['valid_cfes'][i] = n_valid_cfes
                # Finding best activity, best resource
                smallest_total_time = cfe_single_query_evaluate['leadtime'].min()
                best_act, best_res = cfe_single_query_evaluate[cfe_single_query_evaluate['leadtime'] == smallest_total_time][['NEXT_ACTIVITY', 'NEXT_RESOURCE']].values[0] 
                      
                result_df["Next_activity"][i] = best_act
                result_df["Next_resource"][i] = best_res
                result_df["predicted_rec"][i] = smallest_total_time

            else: # No valid CFEs
                result_df['valid_cfes'][i] = 0
                result_df["Next_activity"][i] = next_activity
                result_df["Next_resource"][i] = next_resource
                result_df["predicted_rec"][i] = predicted_time

        else: # No CFEs
            result_df['valid_cfes'][i] = 0
            result_df["Next_activity"][i] = next_activity
            result_df["Next_resource"][i] = next_resource
            result_df["predicted_rec"][i] = predicted_time
        n = n - 1
    return result_df