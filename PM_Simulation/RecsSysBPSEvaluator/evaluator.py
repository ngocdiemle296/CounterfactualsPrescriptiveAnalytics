import numpy as np
import pandas as pd
from datetime import datetime
from RecsSysBPSEvaluator.src.prefix_utils import return_prefixes_from_recommendations_df
from RecsSysBPSEvaluator.src.state_utils import return_attributes_from_recommendations_df, return_act_res_recommendations
from RecsSysBPSEvaluator.PetriNetRecsBPS.src.PetriNetBPS import SimulatorParameters, SimulatorEngine
from tqdm import tqdm


def apply(log, net, initial_marking, final_marking, recommendations_df, res_availability, split_time, data_attributes, categorical_attributes, n_sim=10, mode_ex_time='resource'):

    parameters = SimulatorParameters(net, initial_marking, final_marking)
    parameters.discover_from_eventlog(log, mode_ex_time, mode_trans_weights='data_attributes', data_attributes=data_attributes, categorical_attributes=categorical_attributes, history_weights='binary')

    simulator = SimulatorEngine(net, initial_marking, final_marking, parameters)

    prefixes = return_prefixes_from_recommendations_df(recommendations_df, log)
    if data_attributes:
        attributes = return_attributes_from_recommendations_df(recommendations_df, log, data_attributes)
    else:
        attributes = [[] for _ in range(len(recommendations_df))]


    simulations = []
    top_recomm = []
    for j in range(n_sim):
        print(f'SIM. {j+1}')
        sim_j = []
        for i in tqdm(range(len(recommendations_df))):
            act_rec, res_rec, start_time = return_act_res_recommendations(recommendations_df, res_availability, i, top_k = 3)
            if act_rec:
                recommendations = {
                    "starting_time": start_time,
                    "activities": [act_rec],
                    "resources": [res_rec],
                    "prefixes": [prefixes[i]],
                    "attributes": [attributes[i]]
                }
                log_data = simulator.simulate(1, 
                                            remove_head_tail=0, 
                                            starting_time=recommendations['starting_time'], 
                                            resource_availability=res_availability,
                                            recommendations=recommendations)
                if j == 0:
                    top_recomm.append(simulator.top_k_rec[0])
            
            sim_j.append(log_data)
        
        simulations.append(sim_j)


    if 0 in top_recomm:
        print(str(round(top_recomm.count(0)/len(recommendations_df)*100, 2)) + "% not possible recommendations.")


    simulations_df = []
    for j in range(n_sim):
        sims = []
        for i in range(len(recommendations_df)):
            if top_recomm[i] == 0:
                continue
            case_id = recommendations_df.iloc[i]['case:concept:name']
            sim = simulations[j][i]
            sim['case:concept:name'] = case_id
            sims.append(sim)
        sim_j = pd.concat(sims)
        sim_j.sort_values(by='time:timestamp', inplace=True)
        sim_j.index = range(len(sim_j))
        simulations_df.append(sim_j)

    return simulations_df