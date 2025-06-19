import os
import sys
sys.path.append(os.path.abspath('.')) # to run files that are away
os.environ["WANDB_SILENT"] = "true"  # Suppress WandB logs

libraries = ["torch", "numpy", "polars"]
modules   = {lib: sys.modules.get(lib) for lib in libraries}

if not modules["torch"]:
    import torch
if not modules["numpy"]:
    import numpy as np
if not modules["polars"]:
    import polars as pl

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from ASM_problem.load_and_rename_files import LogFilesProcessor, WaferFilesProcessor
from ASM_problem.prediction_methods import MultiOutputModelPredictor, DataPreprocessor
from typing import List, Union

from key_params import main_folder, NUM_WAFERS, dict_of_wafer_files, dict_of_log_files, step_col_name, COMMON_ID_COLS, COMMON_ID_COLS_MOD
log_processor = LogFilesProcessor(COMMON_ID_COLS_MOD, COMMON_ID_COLS)
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_preprocess_wafer_data(dict_of_wafer_files, main_folder: str, save: bool = False) -> tuple:
    """Merge water files, split by RC, and create y and radius dataframes"""
    processor = WaferFilesProcessor()
    master_df = processor.load_and_merge_wafer_files(dict_of_wafer_files)
    if save:
        master_df.write_parquet(f"{main_folder}/parquet_files/master_wafer_file.parquet")
    wafer_df_dict = processor.split_wafer_df_by_rc(master_df)

    y_dict, radius_dict, radius_wide_dict = {}, {}, {}
    for i, df in wafer_df_dict.items():
        y_dict[i], radius_dict[i] = processor.split_1_wafer_df_to_y_and_radius_df(df)
        r_df = radius_dict[i].sort("marathon_run").with_columns(
            pl.arange(0, pl.len()).over("marathon_run").alias("radius_idx"))
        radius_wide_dict[i] = r_df.pivot(
            values="Radius (mm)", index="marathon_run", on="radius_idx", aggregate_function="first").sort("marathon_run")
    return master_df, wafer_df_dict, y_dict, radius_wide_dict

def load_and_process_log_files(dict_of_log_files, log_processor: LogFilesProcessor, unique_marathon_runs_list: list, step_col_name: str,main_folder: str,save: bool = False) -> pl.DataFrame:
    """Read all stepfile CSVs, concat, then optionally save to parquet"""
    df_list = []

    for entry in dict_of_log_files.values():
        df = log_processor.read_csv_and_rename_cols(entry['path'])
        df = log_processor.add_marathon_and_step_columns(df, entry['marathon'], entry['step'], step_col_name)
        df = log_processor.remove_runs_not_found_in_wafer_df(df, unique_marathon_runs_list)
        df = log_processor.insert_step_cols_after_run(df, step_col_name)
        df = log_processor.cast_int_and_float_to_float64(df)
        df = log_processor.drop_single_value_cols(df, step_col_name)
        df = log_processor.append_step_suffix_to_cols(df, step_col_name, entry['step'])
        df_list.append(df)

    master_log_df = pl.concat(df_list, how="diagonal")
    master_log_df = log_processor.reorder_cols(master_log_df, step_col_name)
    log_df        = master_log_df.rename({c: c.strip().lower() for c in master_log_df.columns})

    if save:
        log_df.write_parquet(f"{main_folder}/parquet_files/master_log_file.parquet")
    return log_df

def split_and_save_log_df_by_wafer(log_df, num_wafers, main_folder, log_processor: LogFilesProcessor, overwrite = False) -> None:
    """Split log_df into 4 parquet files, 1 for each wafer.
    saving to parquet is easier to handle than a dict of dataframes"""

    common_cols   = [c for c in log_df.columns if not c.startswith("rc")]
    # log_processor = LogFilesProcessor(COMMON_ID_COLS_MOD, COMMON_ID_COLS)

    for i in range(num_wafers):
        wafer_cols = [c for c in log_df.columns if c.startswith(f"rc{i+1}")]
        df         = log_df.select(common_cols + wafer_cols).with_columns(pl.lit(i+1).alias("wafer"))

        # reorder to put 'wafer' after 'step_id'
        if "step_id" in df.columns:
            cols = df.columns.copy()
            cols.remove("wafer")
            insert_idx = cols.index("step_id") + 1
            cols = cols[:insert_idx] + ["wafer"] + cols[insert_idx:]
            df   = df.select(cols)

        # Drop constant-valued numeric columns
        df = log_processor.remove_constant_valued_cols(df)

        filepath = f"{main_folder}/parquet_files/wafer_{i+1}_log.parquet"
        if overwrite or not os.path.exists(filepath):
            df.write_parquet(filepath)

def _compute_log_df_grouped_stats(log_df: pl.DataFrame, col_to_group_by: Union[str, List[str]]):
    """Group log_df by specified column(s) and compute statistics (mean, std, min, max, median, skew, kurtosis) for numeric columns
    Excludes columns containing certain substrings or non-numeric types"""

    stat_map = {"mean":   lambda col: pl.col(col).mean(),
                "std":    lambda col: pl.col(col).std(),
                "min":    lambda col: pl.col(col).min(),
                "max":    lambda col: pl.col(col).max(),
                "median": lambda col: pl.col(col).median(),
                "skew":   lambda col: pl.col(col).skew(),
                "kurt":   lambda col: pl.col(col).kurtosis()
                }

    exclude_substrings = ['process time', '#run', 'wafer', 'step_id', 'marathon_run']
    allowed_types= {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64,
                    pl.UInt32, pl.Int16, pl.UInt16, pl.Int8, pl.UInt8}
    numeric_cols = [c for c in log_df.columns
                    if not any(sub in c.lower() for sub in exclude_substrings)
                    and log_df.schema[c] in allowed_types]
    agg_exprs    = [func(col).alias(f"{col}_{stat_name}")
                    for stat_name, func in stat_map.items()
                    for col in numeric_cols]
    result_df    = log_df.group_by(col_to_group_by).agg(agg_exprs)

    # we are already doing the below filtering above, lets remove it
    # filtered_log_df = result_df.filter(pl.col(col_to_group_by).is_in(y_df[col_to_group_by]))
    # return filtered_log_df

    return result_df

def train_models(y_df_dict, radius_wide_dict, main_folder, num_wafers, device):
    predictor    = MultiOutputModelPredictor(device)
    preprocessor = DataPreprocessor()

    total_rmse_lgb, total_rmse_cat = 0, 0
    for wafer_idx in range(num_wafers):
        wafer_log_df           = pl.read_parquet(f"{main_folder}/parquet_files/wafer_{wafer_idx+1}_log.parquet")
        wafer_log_stats_df     = _compute_log_df_grouped_stats(wafer_log_df, 'marathon_run')
        wafer_log_stats_df2    = wafer_log_stats_df.with_columns(pl.lit(wafer_idx+1).alias("wafer"))
        wafer_log_stats_df_with_radius = wafer_log_stats_df2.join(radius_wide_dict[wafer_idx], on="marathon_run", how="left")
        log_stats_df_reordered = wafer_log_stats_df_with_radius.select(["wafer"] + [c for c in wafer_log_stats_df_with_radius.columns if c != "wafer"])

        X = log_stats_df_reordered.to_pandas().drop(columns=["wafer"], errors='ignore')
        y = y_df_dict[wafer_idx].sort("marathon_run").drop("marathon_run").to_pandas()

        X_train: pd.DataFrame
        y_train: np.ndarray
        X_val: pd.DataFrame
        y_val: np.ndarray
        X_train, y_train, X_val, y_val, y_scaler = preprocessor.scale_and_split_data(X, y)

        nan_rows = X_train[X_train.isnull().any(axis=1)]
        print(nan_rows.head(20))



        print(f"====== Wafer {wafer_idx+1} ======")
        rmse_lgb, _ = predictor.predict_lightgbm(X_train, y_train, X_val, y_val)
        print(f"LGBM RMSE: {rmse_lgb:.3f}")
        total_rmse_lgb += rmse_lgb

        # rmse_cat, _ = predictor.predict_catboost(X_train, y_train, X_val, y_val)
        # print(f"Catboost RMSE: {rmse_cat:.3f}")
        # total_rmse_cat += rmse_cat

    print(f"Avg LGBM RMSE: {total_rmse_lgb / num_wafers:.3f}")
    # print(f"Avg Catboost RMSE: {total_rmse_cat / num_wafers:.3f}")


"""ML predictions"""
master_df, wafer_df_dict, y_df_dict, radius_wide_dict=load_and_preprocess_wafer_data(dict_of_wafer_files, main_folder, save=False)
unique_marathon_runs_list = list(master_df["marathon_run"].unique())
log_df = load_and_process_log_files(dict_of_log_files, log_processor, unique_marathon_runs_list, step_col_name, main_folder, save=False)
split_and_save_log_df_by_wafer(log_df, NUM_WAFERS, main_folder, log_processor, overwrite = True)
train_models(y_df_dict, radius_wide_dict, main_folder, NUM_WAFERS, device)


# TODO: null values in log_df_with_wafer_col, what to do with them?