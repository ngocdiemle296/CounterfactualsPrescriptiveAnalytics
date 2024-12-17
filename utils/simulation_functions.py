import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil import parser
from utils.pre_processing import getting_total_time

def preparing_data_for_simulation(result_df, test_log, path_name, case_id_name, end_date_name):
    # Generating dataframe with repl_id, act_1, res_1, starting_time
    test_log_ids = result_df[case_id_name].unique()
    simu_df = pd.DataFrame(columns=["case:concept:name", "repl_id", "act_1", "res_1", "starting_time"])
    simu_df["case:concept:name"] = test_log_ids
        
    for i, idx in enumerate(test_log_ids):
        trace_df = test_log[test_log[case_id_name] == idx]
        trace_history = trace_df[case_id_name].tolist() # List of prefix history
        simu_df['repl_id'][i] = len(trace_history) - 1
        simu_df["act_1"][i] = result_df["Next_activity"][i]
        simu_df["res_1"][i] = result_df["Next_resource"][i]
        simu_df["starting_time"][i] = trace_df[-1:][end_date_name].values[0]
    
    simu_df.to_csv(path_name, index=False)

def extract_valid_idx(rec_df, sim, case_id_name, activity_column_name):
    # Extract only ids of traces those were generated with the simulator
    list_idx = []
    sim_ids = sim[case_id_name].unique()
    for idx in sim_ids:
        sub_df = sim[sim[case_id_name] == idx].reset_index(drop=True)
        if rec_df[rec_df["case:concept:name"] == idx]["act_1"].values[0] == sub_df[activity_column_name][0]:
            list_idx.append(idx)
    return list_idx


def simulation_generation(path_test_log, path_simulation_folder, path_simulation_result, path_rec, case_id_name, n_sim, start_date_name, end_date_name, activity_column_name, resource_column_name):
    """ 
        path_test_log: path to dataset contains only running traces (from beginning to split time)
        path_simulation_folder:  path to folder contains simulation results from simulator
        path_simulation_result: path to folder contains final results after combing simulation data with running traces history 
        path_rec: path to recommendation file
    """
    
    rec_df = pd.read_csv(path_rec)
    # Loading test log (Only running traces)
    test_logg = pd.read_csv(path_test_log)
    
    # BAC
    test_loggg = test_logg[[case_id_name, start_date_name, end_date_name, activity_column_name, resource_column_name]]
    
    for i in tqdm(range(n_sim)):
        sim = pd.read_csv(path_simulation_folder + "sim_{}.csv".format(i+1)) 
        new_sim = sim.rename(columns={'start:timestamp': start_date_name, 
                                  'time:timestamp': end_date_name,
                                  "case:concept:name": case_id_name,
                                  "concept:name": activity_column_name,
                                  "org:resource": resource_column_name})
                   
        # IDs of traces that be able to simulate
        id_in_sim = extract_valid_idx(rec_df, new_sim, case_id_name, activity_column_name)
        # Extract only traces in simulation data (excluding traces cannot be simulated)
        # new_sim =  sim.loc[sim[case_id_name].isin(id_in_sim)].reset_index(drop=True) # CHANGE HERE! 1
        new_test_log =  test_loggg.loc[test_loggg[case_id_name].isin(id_in_sim)].reset_index(drop=True) # CHANGE HERE! 2

        # Combing together (for traces cannot be simulated, we keep the exact traces in simulation data)
        simu = pd.concat([new_test_log, new_sim], axis=0).sort_values(by = [case_id_name, start_date_name]).reset_index(drop=True)
        # simu = pd.concat([new_sim], axis=0).sort_values(by = [case_id_name, start_date_name]).reset_index(drop=True)

        # Assigning different ID to differentiate
        for j in range(len(simu)):
            simu[case_id_name][j] = str(simu[case_id_name][j]) + "_" + str(i+1)
        
        # Saving 
        simu.to_csv(path_simulation_result + "sim_{}.csv".format(i+1), index=False)


def generating_test_simulation(n_sim, path_simulation_result, resource_column_name, start_date_name, end_date_name, case_id_name):
    # Combining data from simulation results (n_sim datasets) and computing total time
    dataframes = []
    for i in range(n_sim):
        sim = pd.read_csv(path_simulation_result + "sim_{}.csv".format(i+1))
        dataframes.append(sim)
    
    test_data_simulation = pd.concat(dataframes, ignore_index=True)

    for i in range(len(test_data_simulation)):
        val = test_data_simulation[resource_column_name][i]
        if val == 'NotDef':
            test_data_simulation[resource_column_name][i] = 'missing'
        # elif val.isnumeric() == True:
        #     test_data_simulation[resource_column_name][i] = str(int(val))

    test_data_simulation[start_date_name] = [parser.parse(i).replace(tzinfo=None) for i in test_data_simulation[start_date_name]]
    test_data_simulation[end_date_name] = [parser.parse(i).replace(tzinfo=None) for i in test_data_simulation[end_date_name]]

    # Computing total time
    df = getting_total_time(test_data_simulation, case_id_name, start_date_name)
    df.to_csv(path_simulation_result + "test_simulation.csv", index = False)

    return df

# Getting list of activities from train/test dataset with respective total time
def getting_history_from_df_with_resource(data, case_id_name, activity_column_name, resource_column_name, outcome_name):
    trace_ids = data[case_id_name].unique()
    history = {}
    for idx in trace_ids:
        trace_df = data[data[case_id_name] == idx]
        trace_activity = trace_df[activity_column_name].tolist()
        trace_resource = trace_df[resource_column_name].tolist()
        total_time = trace_df[outcome_name].values[-1]
        history[idx] = (total_time, trace_activity, trace_resource)
    return history

def getting_final_data(n_sim, path_simulation_result, path_rec_data, dice_params, activity_column_name, resource_column_name, start_date_name, end_date_name, case_id_name, outcome_name):
    df = generating_test_simulation(n_sim, path_simulation_result, resource_column_name, start_date_name, end_date_name, case_id_name)
    sim_history = getting_history_from_df_with_resource(df, case_id_name, activity_column_name, resource_column_name, outcome_name)
    result_df = pd.read_csv(path_rec_data + "results_bac_" + dice_params + ".csv")
    # result_df = pd.read_csv(path_rec_data + dice_params + ".csv")

    result = {}
    for i, idx in enumerate(result_df[case_id_name]):
        kpi = []
        for j in range(n_sim):
            idx_sim = str(idx) + "_" + str(j+1)
            if idx_sim in sim_history.keys():
                kpi.append(sim_history[idx_sim][0])
        result[idx] = kpi

    result_df["kpi_followed"] = ""

    for i, idx in enumerate(result_df[case_id_name]):
        if not result[idx]: # or result_df["valid_cfes"][i] == 0:
            true_outcome = result_df["true_outcome"][i]
            result_df["kpi_followed"][i] = true_outcome
        else:
            kpi = result[idx]
            avg_kpi = np.mean(kpi)
            result_df["kpi_followed"][i] = avg_kpi

    result_df.to_csv(path_rec_data + "result_simulation_" + dice_params + ".csv", index=False)
    return result_df