from RecsSysBPSEvaluator import evaluator
import warnings
warnings.filterwarnings('ignore')


def run_simulation_recommendations(log, net, initial_marking, final_marking, recommendations_df, split_time, res_availability, data_attributes, categorical_attributes, n_sim, path_simulations=None):

    simulations_df = evaluator.apply(log, net, initial_marking, final_marking, recommendations_df, res_availability, split_time, data_attributes, categorical_attributes, n_sim)

    if path_simulations:
        for j in range(len(simulations_df)):
            sim_j = simulations_df[j]
            sim_j.to_csv(path_simulations + f'/sim_{j+1}.csv', index=False)

    return simulations_df