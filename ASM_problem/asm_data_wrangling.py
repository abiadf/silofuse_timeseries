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

import gc
import catboost as cb
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from ASM_problem.load_files import LogFilesProcessor, WaferFilesProcessor
from ASM_problem.prediction_methods import MultiOutputModelPredictor, DataPreprocessor
from typing import List, Union

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% ###############################

from key_params import main_folder, NUM_WAFERS, dict_of_wafer_files, dict_of_log_files, step_col_name, COMMON_ID_COLS, COMMON_ID_COLS_MOD

# %% ###############################
"""Load WAFER data files"""

# wafer_processor = WaferFilesProcessor()
# master_wafer_df = wafer_processor.load_and_merge_wafer_files(dict_of_wafer_files)

# unique_marathon_runs_list = list(master_wafer_df["marathon_run"].unique())

# # Save to parquet
# should_we_save_wafer = False
# if should_we_save_wafer:
#     wafer_saving_location = f"{main_folder}/parquet_files/master_wafer_file.parquet"
#     master_wafer_df.write_parquet(wafer_saving_location)
#     print(f"Consolidated data saved to: {wafer_saving_location}")

# wafer_df_dict = wafer_processor.split_wafer_df_by_rc(master_wafer_df)

# y_df_dict       = {}
# radius_df_dict  = {}
# radius_wide_dict= {}

# for i, df in wafer_df_dict.items():
#     y_df_dict[i], radius_df_dict[i] = wafer_processor.split_1_wafer_df_to_y_and_radius_df(df)

# for i, df in radius_df_dict.items():
#     radius_df   = (df.sort("marathon_run").with_columns(pl.arange(0, pl.len()).over("marathon_run").alias("radius_idx")))
#     radius_wide = radius_df.pivot(values = "Radius (mm)",
#                                   index  = "marathon_run",
#                                   columns= "radius_idx",
#                                   aggregate_function="first").sort("marathon_run")
#     radius_wide_dict[i] = radius_wide

# del i, df, radius_df_dict, radius_df, radius_wide
# gc.collect()


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



# %% ##############################
"""Load+process LOG data files"""

# log_processor = LogFilesProcessor(COMMON_ID_COLS_MOD, COMMON_ID_COLS)
# df_list       = []

# for entry in dict_of_log_files.values():
#     file_path = entry['path']
#     step_id   = entry['step']
#     marathon  = entry['marathon']
#     print(f"Processing {file_path}, step {step_id}, marathon {marathon}...")

#     df = log_processor.read_csv_and_rename_cols(file_path)
#     df = log_processor.add_marathon_and_step_columns(df, marathon, step_id, step_col_name)
#     filtered_df = log_processor.remove_runs_not_found_in_wafer_df(df, unique_marathon_runs_list)
#     df = log_processor.insert_step_cols_after_run(filtered_df, step_col_name)
#     df = log_processor.cast_int_and_float_to_float64(df)
#     df = log_processor.drop_single_value_cols(df, step_col_name)
#     df = log_processor.append_step_suffix_to_cols(df, step_col_name, step_id)
#     df_list.append(df)
# del entry, file_path, step_id, df, filtered_df
# gc.collect()

# master_log_df = pl.concat(df_list, how="diagonal")
# print(f"Master log df shape: {master_log_df.shape}")

# # # Fill nulls with 0 in numeric cols except COMMON_ID_COLS_MOD
# # for col in master_log_df.columns:
# #     if master_log_df[col].dtype.is_numeric() and col not in COMMON_ID_COLS_MOD:
# #         master_log_df = master_log_df.with_columns(pl.col(col).fill_null(0))

# master_log_df = log_processor.reorder_cols(master_log_df, step_col_name)
# log_df        = master_log_df.rename({c: c.strip().lower() for c in master_log_df.columns})

# # Save to parquet
# should_we_save = False
# if should_we_save:
#     saving_location = f"{main_folder}/parquet_files/master_log_file.parquet"
#     master_log_df.write_parquet(saving_location)
#     print(f"Consolidated data saved to: {saving_location}")

# del df_list, master_log_df
# gc.collect()

# ===============
log_processor = LogFilesProcessor(COMMON_ID_COLS_MOD, COMMON_ID_COLS)

def load_and_process_log_files(dict_of_log_files,unique_marathon_runs_list: list, step_col_name: str,main_folder: str,save: bool = False) -> pl.DataFrame:
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


# %% ##############################
"""Split log_df into 4 parquet files, 1 for each wafer"""

# wafer_log_df_dict= {}
# common_cols      = [c for c in log_df.columns if not c.startswith("rc")]

# for i in range(NUM_WAFERS):
#     wafer_cols    = [c for c in log_df.columns if c.startswith(f"rc{i+1}")]
#     cols_to_keep  = common_cols + wafer_cols
#     log_df_with_wafer_col = log_df.select(cols_to_keep).with_columns(pl.lit(i+1).alias("wafer"))

#     # reorder to put 'wafer' after 'step_id'
#     if "step_id" in (cols := log_df_with_wafer_col.columns):
#         cols.remove("wafer")
#         insert_idx = cols.index("step_id") + 1
#         cols = cols[:insert_idx] + ["wafer"] + cols[insert_idx:]
#         log_df_with_wafer_col = log_df_with_wafer_col.select(cols)

#     # # Fill nulls in numeric columns (except common id cols)
#     # for col in log_df_with_wafer_col.columns:
#     #     if log_df_with_wafer_col[col].dtype.is_numeric() and col not in COMMON_ID_COLS_MOD:
#     #         log_df_with_wafer_col = log_df_with_wafer_col.with_columns(pl.col(col).fill_null(0))

#     # Drop constant-valued numeric columns
#     log_df_with_wafer_col = log_processor.remove_constant_valued_cols(log_df_with_wafer_col)

#     # save to parquet, easier to handle than a dict of dataframes
#     log_df_with_wafer_col.write_parquet(f"{main_folder}/parquet_files/wafer_{i+1}_log.parquet")

#     del log_df_with_wafer_col
#     gc.collect()

# del log_df, wafer_cols, cols_to_keep
# gc.collect()
# ==============

def split_and_save_log_df_by_wafer(log_df, num_wafers, main_folder) -> None:
    """Split log_df into 4 parquet files, 1 for each wafer.
    saving to parquet is easier to handle than a dict of dataframes"""

    common_cols   = [c for c in log_df.columns if not c.startswith("rc")]
    log_processor = LogFilesProcessor(COMMON_ID_COLS_MOD, COMMON_ID_COLS)

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
        df.write_parquet(f"{main_folder}/parquet_files/wafer_{i+1}_log.parquet")


# %% ##############################
# reshape LOG_df to X

def compute_log_df_grouped_stats(log_df: pl.DataFrame, col_to_group_by: Union[str, List[str]]):
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


# =============
# # to remove below block when the next cell works, so we read directly from parquet instead of handling dicts
# summary_log_df_dict = {}
# for i, wafer_log_df in wafer_log_df_dict.items():
#     summary_log_df_dict[i] = compute_log_df_grouped_stats(wafer_log_df, 'marathon_run')
#     summary_log_df_dict[i] = summary_log_df_dict[i].with_columns(pl.lit(i+1).alias("wafer"))

#     # join radius col from radius_df_dict[i]
#     summary_log_df_dict[i] = summary_log_df_dict[i].join(radius_wide_dict[i], on="marathon_run", how="left")
#     cols                   = summary_log_df_dict[i].columns
#     summary_log_df_dict[i] = summary_log_df_dict[i].select(["wafer"] + [c for c in cols if c != "wafer"])
# =============



# %% ##############################
"""ML predictions"""

# rmse_lgb_total = 0
# rmse_cat_total = 0
# # rmse_rf_total  = 0
# # rmse_xgb_total = 0
# # rmse_hgb_total = 0
# # rmse_elas_total= 0

# predictor    = MultiOutputModelPredictor(device)
# preprocessor = DataPreprocessor()
# # X_full_pd, y_full_pd = preprocessor.join_logs_and_wafer_df(log_df, wafer_df, y_df)

# count = 0
# for i in range(NUM_WAFERS):
#     wafer_log_df= pl.read_parquet(f"{main_folder}/parquet_files/wafer_{i+1}_log.parquet")
#     summary_df  = compute_log_df_grouped_stats(wafer_log_df, 'marathon_run')
#     summary_df  = summary_df.with_columns(pl.lit(i+1).alias("wafer"))
#     summary_df_with_radius  = summary_df.join(radius_wide_dict[i], on="marathon_run", how="left")
#     cols        = summary_df_with_radius.columns
#     summary_df2 = summary_df_with_radius.select(["wafer"] + [c for c in cols if c != "wafer"])

#     X_full_pd   = summary_df2.to_pandas().drop(columns=["wafer"], errors='ignore')
#     y_full_pd   = y_df_dict[i].sort("marathon_run").drop("marathon_run").to_pandas()
#     # X_train_final, y_train, X_val_final, y_val, y_scaler = scale_and_split_data(X_full_pd, y_full_pd)
#     X_train_final, y_train, X_val_final, y_val, y_scaler = preprocessor.scale_and_split_data(X_full_pd, y_full_pd)

#     print(f"====== Wafer {i+1} ======")
#     rmse_lgb, y_pred_lgb = predictor.predict_lightgbm(X_train_final, y_train, X_val_final, y_val)
#     print(f"LGBM RMSE: {rmse_lgb:.3f}")

#     rmse_cat, y_pred_cat = predictor.predict_catboost(X_train_final, y_train, X_val_final, y_val, device)
#     print(f"Catboost RMSE: {rmse_cat:.3f}")

#     del wafer_log_df, summary_df, X_full_pd, y_full_pd, X_train_final, y_train, X_val_final, y_val, cols
#     gc.collect()

#     rmse_lgb_total += rmse_lgb
#     rmse_cat_total += rmse_cat
#     count += 1

# print(f"Average LGBM RMSE: {rmse_lgb_total / count:.3f}")
# print(f"Average Catboost RMSE: {rmse_cat_total / count:.3f}")

# ==============

def train_models(y_df_dict, radius_wide_dict, main_folder, num_wafers, device):
    predictor = MultiOutputModelPredictor(device)
    preprocessor = DataPreprocessor()

    total_rmse_lgb, total_rmse_cat = 0, 0
    for i in range(num_wafers):
        wafer_log_df           = pl.read_parquet(f"{main_folder}/parquet_files/wafer_{i+1}_log.parquet")
        wafer_log_stats_df     = compute_log_df_grouped_stats(wafer_log_df, 'marathon_run')
        wafer_log_stats_df2    = wafer_log_stats_df.with_columns(pl.lit(i+1).alias("wafer"))
        wafer_log_stats_df_with_radius = wafer_log_stats_df2.join(radius_wide_dict[i], on="marathon_run", how="left")
        log_stats_df_reordered = wafer_log_stats_df_with_radius.select(["wafer"] + [c for c in wafer_log_stats_df_with_radius.columns if c != "wafer"])

        X = log_stats_df_reordered.to_pandas().drop(columns=["wafer"], errors='ignore')
        y = y_df_dict[i].sort("marathon_run").drop("marathon_run").to_pandas()

        X_train, y_train, X_val, y_val, _ = preprocessor.scale_and_split_data(X, y)

        print(f"====== Wafer {i+1} ======")
        rmse_lgb, _ = predictor.predict_lightgbm(X_train, y_train, X_val, y_val)
        print(f"LGBM RMSE: {rmse_lgb:.3f}")
        total_rmse_lgb += rmse_lgb

        rmse_cat, _ = predictor.predict_catboost(X_train, y_train, X_val, y_val, device)
        print(f"Catboost RMSE: {rmse_cat:.3f}")
        total_rmse_cat += rmse_cat

    print(f"Avg LGBM RMSE: {total_rmse_lgb / num_wafers:.3f}")
    print(f"Avg Catboost RMSE: {total_rmse_cat / num_wafers:.3f}")


master_df, wafer_df_dict, y_df_dict, radius_wide_dict=load_and_preprocess_wafer_data(dict_of_wafer_files, main_folder, save=False)
unique_marathon_runs_list = list(master_df["marathon_run"].unique())

log_df = load_and_process_log_files(dict_of_log_files, unique_marathon_runs_list, step_col_name, main_folder, save=False)
split_and_save_log_df_by_wafer(log_df, NUM_WAFERS, main_folder)
train_models(y_df_dict, radius_wide_dict, main_folder, NUM_WAFERS, device)



# TODO: null values in log_df_with_wafer_col, what to do with them?

# # Fill nulls in numeric columns (except common id cols)
# for col in log_df_with_wafer_col.columns:
#     if log_df_with_wafer_col[col].dtype.is_numeric() and col not in COMMON_ID_COLS_MOD:
#         log_df_with_wafer_col = log_df_with_wafer_col.with_columns(pl.col(col).fill_null(0))