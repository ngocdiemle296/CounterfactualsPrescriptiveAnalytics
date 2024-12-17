import pandas as pd
import numpy as np
import pm4py
from utils.pre_processing_functions import prepare_data_and_add_features, add_next_act_res, preprocessing_activity_frequency, getting_total_time
from dateutil import parser


# def data_pre_processing(path_to_df, case_id_position, start_date_position, date_format, end_date_position):
#     print("Loading dataset...")
#     df = pd.read_csv(path_to_df)
#     df[df.columns[start_date_position]] = [parser.parse(i) for i in df.iloc[:,start_date_position]] 
#     df[df.columns[end_date_position]] = [parser.parse(i) for i in df.iloc[:,end_date_position]] 
#     print("Start pre-processing data...")
#     df = prepare_data_and_add_features(df, case_id_position, start_date_position, 
#                                        date_format, end_date_position)
#     print("Finished pre-processing!")
#     return df


def data_pre_processing(df, case_id_position, start_date_position, date_format, end_date_position):
    print("Loading dataset...")
    # df = pd.read_csv(path_to_df)
    df[df.columns[start_date_position]] = [parser.parse(i) for i in df.iloc[:,start_date_position]] 
    df[df.columns[end_date_position]] = [parser.parse(i) for i in df.iloc[:,end_date_position]] 
    print("Start pre-processing data...")
    df = prepare_data_and_add_features(df, case_id_position, start_date_position, 
                                       date_format, end_date_position)
    print("Finished pre-processing!")
    return df

def adding_features(df, activity_column_name, resource_column_name, case_id_name, start_date_name):
    print("Adding total time...")
    df = getting_total_time(df, case_id_name, start_date_name)
    print("Adding activity frequency...")
    df = preprocessing_activity_frequency(df, activity_column_name, case_id_name, start_date_name)
    print("Adding next activity and next resource...")
    df = add_next_act_res(df, activity_column_name, resource_column_name, case_id_name)
    print("Finished adding features!")
    return df

