import pandas as pd
import polars as pl

def count_missing_values_in_df(df) -> None:
    """Count NaNs (floats) and nulls (all types) in pandas or Polars DataFrame."""

    if isinstance(df, pd.DataFrame):
        float_cols = [col for col, dt in df.dtypes.items() if pd.api.types.is_float_dtype(dt)]

        # NaNs in float columns
        nan_per_col = df[float_cols].isna().sum()
        cols_with_nan = (nan_per_col > 0).sum()
        rows_with_nan = df[float_cols].isna().any(axis=1).sum()
        total_nans = nan_per_col.sum()

        # Nulls in all columns (in pandas, NaN == null)
        null_per_col = df.isnull().sum()
        cols_with_nulls = (null_per_col > 0).sum()
        rows_with_nulls = df.isnull().any(axis=1).sum()
        total_nulls = null_per_col.sum()

    elif isinstance(df, pl.DataFrame):
        float_cols = [col for col, dt in zip(df.columns, df.dtypes) if dt.is_float()]

        if float_cols:
            nan_per_col = df.select([pl.col(col).is_nan().sum().alias(col) for col in float_cols])
            cols_with_nan = sum(val > 0 for val in nan_per_col.row(0))
            rows_with_nan = df.filter(
                pl.fold(False, lambda acc, e: acc | e, [pl.col(c).is_nan() for c in float_cols])).height
            total_nans = nan_per_col.to_series().sum()
        else:
            cols_with_nan = rows_with_nan = total_nans = 0

        null_per_col = df.select([pl.col(col).is_null().sum().alias(col) for col in df.columns])
        cols_with_nulls = sum(val > 0 for val in null_per_col.row(0))
        rows_with_nulls = df.filter(
            pl.fold(False, lambda acc, e: acc | e, [pl.col(c).is_null() for c in df.columns])).height
        total_nulls = null_per_col.to_series().sum()

    else:
        raise TypeError("Unsupported DataFrame type. Pass pandas or Polars DataFrame.")

    print(f"# columns with NaNs: {cols_with_nan}")
    print(f"# rows with NaNs: {rows_with_nan}")
    print(f"Total NaNs: {total_nans}")
    print(f"# columns with nulls: {cols_with_nulls}")
    print(f"# rows with nulls: {rows_with_nulls}")
    print(f"Total nulls: {total_nulls}")
