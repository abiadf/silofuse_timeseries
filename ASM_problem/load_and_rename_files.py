
import polars as pl
import pandas as pd
from typing import Tuple, Union

class LogFilesProcessor:
    def __init__(self, common_id_cols_mod, common_id_cols):
        self.COMMON_ID_COLS_MOD = common_id_cols_mod
        self.COMMON_ID_COLS     = common_id_cols

    def read_csv_and_rename_cols(self, file_path: str) -> pl.DataFrame:
        """Read CSV into Polars DataFrame and title-case column names after stripping spaces
        NOTE: a known polars issue that it cant use the 'decimal' parameter in read_csv, so we load into pandas first"""
        pdf = pd.read_csv(file_path, decimal='.')
        df  = pl.from_pandas(pdf)
        df  = df.rename({c: c.strip().title() for c in df.columns})
        return df

    def add_marathon_and_step_columns(self, df: pl.DataFrame, marathon: Union[str, int], step_id: int, step_col_name: str) -> pl.DataFrame:
        """Add marathon, step_id columns, and create 'marathon_run' by combining marathon and '#Run'."""
        df = df.with_columns([
            pl.lit(marathon).alias("marathon"),
            pl.lit(step_id).alias(step_col_name)])
        df = df.with_columns([
            (pl.col("marathon").cast(pl.Utf8) + "_" + pl.col("#Run").cast(pl.Utf8)).alias("marathon_run")])
        return df

    def remove_runs_not_found_in_wafer_df(self, df: pl.DataFrame, unique_marathon_runs_list: list) -> pl.DataFrame:
        """Keep only rows where 'marathon_run' exists in the provided list of valid wafer runs
        removing runs early on makes the processing faster/lighter"""
        df = df.filter(pl.col("marathon_run").is_in(unique_marathon_runs_list))
        return df

    def insert_step_cols_after_run(self, df: pl.DataFrame, step_col_name: str) -> pl.DataFrame:
        """Reorder columns to insert step_col_name and 'marathon_run' after '#Run'."""
        cols = df.columns
        insert_idx = cols.index("#Run") + 1
        cols.remove(step_col_name)
        cols.remove("marathon_run")
        cols.insert(insert_idx, step_col_name)
        cols.insert(insert_idx + 1, "marathon_run")
        df = df.select(cols)
        return df

    def cast_int_and_float_to_float64(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cast integer and float columns to Float64 type."""
        df = df.with_columns([
            pl.col(col).cast(pl.Float64)
            for col in df.columns
            if df[col].dtype in [pl.Int64, pl.Float32, pl.Int32]])
        return df

    def drop_single_value_cols(self, df: pl.DataFrame, step_col_name: str) -> pl.DataFrame:
        """Drop columns with a single unique value, excluding common ID columns and step_col_name."""
        single_val_cols = [
            col for col in df.columns
            if col not in self.COMMON_ID_COLS_MOD + [step_col_name]
            and df[col].n_unique() == 1]
        df = df.drop(single_val_cols)
        return df

    def append_step_suffix_to_cols(self, df: pl.DataFrame, step_col_name: str, step_id: int) -> pl.DataFrame:
        """Rename non-ID columns by appending '_step{step_id}' suffix"""
        append_step_suffix_to_cols = {
            col: f"{col}_step{step_id}"
            for col in df.columns
            if col not in self.COMMON_ID_COLS_MOD + self.COMMON_ID_COLS + [step_col_name]}
        df = df.rename(append_step_suffix_to_cols)
        return df

    def reorder_cols(self, master_log_df: pl.DataFrame, step_col_name: str) -> pl.DataFrame:
        """Reorder columns to have common ID columns, step_col_name first, then others; warn if columns missing."""
        desired_order = self.COMMON_ID_COLS_MOD + [step_col_name] + [
            col for col in master_log_df.columns if col not in self.COMMON_ID_COLS_MOD + [step_col_name]]
        missing = [col for col in desired_order if col not in master_log_df.columns]
        if missing:
            print("Missing columns before select:", missing)
        master_log_df = master_log_df.select(desired_order)
        return master_log_df
    
    def remove_constant_valued_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        """Drop numeric columns with a single unique value (aka constant-valued columns)"""
        constant_cols = [
            col for col in df.columns
            if df[col].dtype.is_numeric() and df[col].n_unique() == 1]
        df = df.drop(constant_cols)
        return df


class WaferFilesProcessor:
    # def __init__(self, common_id_cols_mod):
        # self.COMMON_ID_COLS_MOD = common_id_cols_mod

    @staticmethod
    def load_and_merge_wafer_files(dict_of_wafer_files: dict) -> pl.DataFrame:
        dfs = []
        for d in dict_of_wafer_files.values():
            pdf = pd.read_csv(
                d['path'],
                decimal='.',                    # adjust if needed
                na_values=["", "NA", "null"],)  # treat as nulls
            df = pl.from_pandas(pdf).with_columns([
                pl.lit(d['marathon']).alias("marathon"),
                (pl.lit(d['marathon']).cast(pl.Utf8) + "_" + pl.col("#Run").cast(pl.Utf8)).alias("marathon_run")])
            dfs.append(df)

        wafer_master_df = pl.concat(dfs, how="vertical")

        cols = wafer_master_df.columns.copy()
        for c in ["marathon", "marathon_run"]:
            cols.remove(c)
        insert_idx = cols.index("#Run") + 1
        cols[insert_idx:insert_idx] = ["marathon", "marathon_run"]

        return wafer_master_df.select(cols)

    @staticmethod
    def split_1_wafer_df_to_y_and_radius_df(wafer_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Pivot wafer_df to create:
        - y_df (output): Spatial property averaged by marathon_run and Site #
        - radius_df (output): marathon_run and Radius (mm) columns
        - wafer_df (pl.DataFrame): df with SINGLE wafer data"""

        pivot_df = wafer_df.pivot(values = "Radius (mm)",
                                  index  = ["marathon_run", "wafer"],#"RC"],
                                  columns= "Site #",
                                  aggregate_function="mean")
        pivot_df = pivot_df.rename({str(c): f"site_{c}" for c in pivot_df.columns if isinstance(c, int) or c.isdigit()})

        y_df = wafer_df.pivot(values = "Spatial property (nm)",
                              index  = "marathon_run",
                              columns= "Site #",
                              aggregate_function="mean")
        y_df      = y_df.sort("marathon_run")

        radius_df = wafer_df.select(["marathon_run", "Radius (mm)"])

        # =================
        # added this part
        # print(radius_df.shape)
        # radius_df = radius_df.join(
        #     radius_df.group_by(["marathon_run", "Radius (mm)"]).count().filter(pl.col("count") == 1),
        #     on=["marathon_run", "Radius (mm)"], how="inner")

        # print(radius_df.shape)
        # =================

        counts = wafer_df.group_by(["marathon_run", "Site #"]).agg(pl.count()).rename({"count": "cnt"})
        dupes  = counts.filter(pl.col("cnt") > 1).select(["marathon_run", "Site #"])
        if dupes.height > 0:
            raise KeyError(f"{dupes.height} Duplicate entries found in wafer_df")

        return y_df, radius_df

    @staticmethod
    def split_wafer_df_by_rc(wafer_df: pl.DataFrame) -> dict:
        """Splits wafer_df by the 'RC' col values and returns a dict of dfs, each has as key (0-3) and wafer/RC value (1-4)"""

        # wafer_df_dict = {i: df.with_columns(pl.lit(i).alias("wafer"))
        #                  for i, (_, df) in enumerate(wafer_df.partition_by("RC", as_dict=True).items())}
        # wafer_df_dict = {rc_val: df.with_columns(pl.col("RC").alias("wafer"))
        #                 for rc_val, df in wafer_df.partition_by("RC", as_dict=True).items()}
        # wafer_df_dict = {rc_val: df.rename({"RC": "wafer"})
        #                  for rc_val, df in wafer_df.partition_by("RC", as_dict=True).items()}
        # the 'rename' method^ was causing memory issues

        wafer_df_dict = {}
        for rc_val in wafer_df["RC"].unique():
            df_subset = wafer_df.filter(pl.col("RC") == rc_val).with_columns(
                pl.col("RC").alias("wafer"))  # keep original RC values in the column
            wafer_df_dict[int(rc_val) - 1] = df_subset
        return wafer_df_dict

