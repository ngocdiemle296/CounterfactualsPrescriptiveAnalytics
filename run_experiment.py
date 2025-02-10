import pandas as pd
import argparse
import joblib
from utils.dice_functions import dice_recommendations
from utils import get_features

outcome_name = "leadtime"

def load_case_study(case_study):
    data_path = f"./case_studies/{case_study}/"
    train_data = pd.read_csv(data_path + "train_data.csv")
    test_data = pd.read_csv(data_path + "test_log_only_last_act_with_tsplit.csv") 
    test_log = pd.read_csv(data_path + "test_log_with_tsplit.csv")
    return train_data, test_data, test_log

def get_case_study_features(case_study):
    data_path = f"./case_studies/{case_study}/"
    predictive_model = joblib.load(data_path + "catboost_time.joblib")
    case_id_name, activity_column_name, resource_column_name, continuous_features = get_features(case_study)
    return predictive_model, case_id_name, activity_column_name, resource_column_name, continuous_features

def run_experiment(case_study, reduced_threshold, num_cfes):
    print(f"Running experiment for case study: {case_study}")
    print(f"Reduction threshold: {reduced_threshold}")
    print(f"Number of counterfactual explanations: {num_cfes}")

    print(f"Loading data...")
    train_data, test_data, test_log = load_case_study(case_study)

    print(f"Getting features...")
    predictive_model, case_id_name, activity_column_name, resource_column_name, continuous_features, start_date_name, end_date_name = get_case_study_features(case_study)

    print(f"Generating recommendations...")
    result_df = pd.DataFrame() # Empty dataframe for storing results
    recommendations_df = dice_recommendations(train_data, test_log, test_data, 
                              threshold = 0.99, 
                              predictive_model = predictive_model, 
                              continuous_features = continuous_features, 
                              dice_method = "random", 
                              REDUCED_KPI_TIME = reduced_threshold, TOTAL_CFS = num_cfes, 
                              result_df = result_df, 
                              WINDOW_SIZE = 3, 
                              cols_to_vary = ["NEXT_ACTIVITY", "NEXT_RESOURCE"], 
                              case_id_name = case_id_name, 
                              activity_column_name = activity_column_name, 
                              resource_column_name = resource_column_name, 
                              outcome_name = outcome_name,
                              start_date_name = start_date_name, 
                              end_date_name = end_date_name,
                              time_limit = 50*60)
    recommendations_df.to_csv(f"{case_study}_{reduced_threshold}_{num_cfes}_recommendations.csv", index=False) # Save results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment with specified parameters.")
    parser.add_argument("--case_study", type=str, required=True, help="Specify the case study.")
    parser.add_argument("--reduced_threshold", type=float, required=True, help="Set the reduction threshold.")
    parser.add_argument("--num_cfes", type=int, required=True, help="Set the number of counterfactual explanations.")

    args = parser.parse_args()
    
    run_experiment(args.case_study, args.reduced_threshold, args.num_cfes)
