
import polars as pl

class LogFilesProcessor:
    def __init__(self, common_id_cols_mod):
        self.COMMON_ID_COLS_MOD = common_id_cols_mod

    def read_csv_and_rename_cols(self, file_path):
        df = pl.read_csv(file_path, ignore_errors=True)
        df = df.rename({c: c.strip().title() for c in df.columns})
        return df

    def add_cols_to_df(self, df, marathon, step_id, step_col_name):
        df = df.with_columns([
            pl.lit(marathon).alias("marathon"),
            pl.lit(step_id).alias(step_col_name)])
        df = df.with_columns([
            (pl.col("marathon").cast(pl.Utf8) + "_" + pl.col("#Run").cast(pl.Utf8)).alias("marathon_run")])
        return df

    def rearrange_and_insert_cols(self, df, step_col_name):
        cols = df.columns
        insert_idx = cols.index("#Run") + 1
        cols.remove(step_col_name)
        cols.remove("marathon_run")
        cols.insert(insert_idx, step_col_name)
        cols.insert(insert_idx + 1, "marathon_run")
        df = df.select(cols)
        return df

    def cast_numeric_cols(self, df):
        df = df.with_columns([
            pl.col(col).cast(pl.Float64)
            for col in df.columns
            if df[col].dtype in [pl.Int64, pl.Float32, pl.Int32]])
        return df

    def drop_single_value_cols(self, df, step_col_name):
        single_val_cols = [
            col for col in df.columns
            if col not in self.COMMON_ID_COLS_MOD + [step_col_name]
            and df[col].n_unique() == 1]
        df = df.drop(single_val_cols)
        return df

    def rename_mapping(self, df, step_col_name, step_id):
        rename_mapping = {
            col: f"{col}_step{step_id}"
            for col in df.columns
            if col not in self.COMMON_ID_COLS_MOD + [step_col_name]}
        df = df.rename(rename_mapping)
        return df

    def reorder_cols(self, master_df, step_col_name):
        desired_order = self.COMMON_ID_COLS_MOD + [step_col_name] + [
            col for col in master_df.columns if col not in self.COMMON_ID_COLS_MOD + [step_col_name]]
        missing = [col for col in desired_order if col not in master_df.columns]
        if missing:
            print("Missing columns before select:", missing)
        master_df = master_df.select(desired_order)
        return master_df


class WaferFilesProcessor:
    # def __init__(self, common_id_cols_mod):
        # self.COMMON_ID_COLS_MOD = common_id_cols_mod

    @staticmethod
    def load_and_process_wafer_files(dict_of_wafer_files: dict) -> pl.DataFrame:
        """Read CSVs + add 'marathon' and 'marathon_run' cols + move them to right after '#run' """

        dfs = [pl.read_csv(d['path'], ignore_errors=True).with_columns([
            pl.lit(d['marathon']).alias("marathon"),
            (pl.lit(d['marathon']).cast(pl.Utf8) + "_" + pl.col("#Run").cast(pl.Utf8)).alias("marathon_run")])
            for d in dict_of_wafer_files.values()]
        wafer_master_df = pl.concat(dfs, how="vertical")

        cols = wafer_master_df.columns
        for c in ["marathon", "marathon_run"]:
            cols.remove(c)
        insert_idx = cols.index("#Run") + 1
        cols[insert_idx:insert_idx] = ["marathon", "marathon_run"]
        return wafer_master_df.select(cols)



