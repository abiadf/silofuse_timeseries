from typing import Tuple

import numpy as np
import polars as pl
import pandas as pd
import catboost as cb
import xgboost as xgb

import itertools
from itertools import product
from lightgbm import LGBMRegressor, early_stopping

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline


class MultiOutputModelPredictor:
    def __init__(self, device):
        self.device = device
        self.device_str = 'GPU' if self.device.type == 'cuda' else 'CPU'


    def predict_linear_reg(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        model         = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)
        y_pred_linreg = model.predict(X_val)
        rmse_linreg   = mean_squared_error(y_val, y_pred_linreg) ** 0.5
        return rmse_linreg, y_pred_linreg


    def predict_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        n_targets = y_train.shape[1]
        y_pred    = np.zeros(y_val.shape)
        
        for i in range(n_targets):
            model = LGBMRegressor(objective       = 'regression',
                                  verbosity       = -1,
                                  n_estimators    = 1000,
                                  learning_rate   = 0.1,
                                  num_leaves      = 31,
                                  max_depth       = 3,
                                  min_data_in_leaf= 1,
                                  device          = self.device_str)
            model.fit(X_train, y_train[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
            y_pred[:, i] = model.predict(X_val)
        
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        return rmse, y_pred


    def tune_lightgbm_hyperparams_manual(self, X_train, y_train):
        n_targets = y_train.shape[1]
        param_grid = {
            'n_estimators':     [100, 200],
            'learning_rate':    [0.01, 0.05],
            'num_leaves':       [31, 50],
            'max_depth':        [-1, 10],
            'min_data_in_leaf': [20, 50]}
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        combos = list(product(*param_grid.values()))
        results = []

        print(f"Total combos: {len(combos)}")
        for idx, combo in enumerate(combos, 1):
            params = dict(zip(param_grid.keys(), combo))
            print(f"Trying combo {idx}/{len(combos)}: {params}")
            
            val_scores = []
            for train_idx, val_idx in kf.split(X_train):
                # Use .iloc only if X_train is a DataFrame
                if hasattr(X_train, 'iloc'):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                else:
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]

                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                y_pred = np.zeros(y_val.shape)
                for i in range(n_targets):
                    model = LGBMRegressor(objective='regression',
                                        verbosity=-1,
                                        device=self.device_str,
                                        **params)
                    model.fit(X_tr, y_tr[:, i], eval_set=[(X_val, y_val[:, i])],
                            callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
                    y_pred[:, i] = model.predict(X_val)

                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                val_scores.append(rmse)

            avg_rmse = np.mean(val_scores)
            print(f"Avg RMSE: {avg_rmse}")
            results.append((params, avg_rmse))

        best_params = min(results, key=lambda x: x[1])[0]
        print(f"Best params: {best_params}")
        return best_params


    def predict_catboost(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        """CatBoost only accepts uppercase 'task_type', beware of that"""
        model = MultiOutputRegressor(cb.CatBoostRegressor(verbose    = 0,
                                                          iterations = 100,
                                                          task_type  = 'CPU'))#self.device_str))
        model.fit(X_train, y_train)
        y_pred_cat = model.predict(X_val)
        rmse_cat = mean_squared_error(y_val, y_pred_cat) ** 0.5
        return rmse_cat, y_pred_cat

    def tune_catboost_hyperparams(self, X_train: np.ndarray, y_train: np.ndarray):

        param_grid = {
            'iterations': [50],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6],
            'l2_leaf_reg': [3, 7],
            'border_count': [32, 64],
            'bagging_temperature': [0, 1]}

        best_score  = float('inf')
        best_params = None
        best_model  = None

        combos = list(itertools.product(
            param_grid['iterations'],
            param_grid['learning_rate'],
            param_grid['depth'],
            param_grid['l2_leaf_reg'],
            param_grid['border_count'],
            param_grid['bagging_temperature'],))

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        total = len(combos)
        for idx, (iterations, learning_rate, depth, l2_leaf_reg, border_count, bagging_temperature) in enumerate(combos, 1):
            print(f"Combo {idx}/{total}: iter={iterations}, lr={learning_rate}, depth={depth}, l2={l2_leaf_reg}, border={border_count}, bagging_temp={bagging_temperature}")
            
            cv_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                base_model = cb.CatBoostRegressor(
                    verbose=0,
                    task_type='CPU',
                    thread_count=1,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    depth=depth,
                    l2_leaf_reg=l2_leaf_reg,
                    border_count=border_count,
                    bagging_temperature=bagging_temperature,
                    random_seed=42,
                    early_stopping_rounds=30)
                multi_model = MultiOutputRegressor(base_model)
                multi_model.fit(X_tr, y_tr)
                preds = multi_model.predict(X_val)
                rmse = root_mean_squared_error(y_val, preds)
                cv_scores.append(rmse)

            avg_rmse = np.mean(cv_scores)
            print(f"Avg RMSE: {avg_rmse:.4f}")

            if avg_rmse < best_score:
                best_score = avg_rmse
                best_params = {
                    'iterations': iterations,
                    'learning_rate': learning_rate,
                    'depth': depth,
                    'l2_leaf_reg': l2_leaf_reg,
                    'border_count': border_count,
                    'bagging_temperature': bagging_temperature,}
                best_model = multi_model

        print("Best params:", best_params)
        print(f"Best CV RMSE: {best_score:.4f}")
        return best_model


    @staticmethod
    def predict_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', verbosity=0))
        model.fit(X_train, y_train)
        y_pred_xgb = model.predict(X_val)
        rmse_xgb = mean_squared_error(y_val, y_pred_xgb) ** 0.5
        return rmse_xgb, y_pred_xgb

    @staticmethod
    def predict_randomforest(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        rf_model  = MultiOutputRegressor(RandomForestRegressor(random_state=42))
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_val)
        rmse_rf   = mean_squared_error(y_val, y_pred_rf) ** 0.5
        return rmse_rf, y_pred_rf

    @staticmethod
    def predict_hgb(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100))
        model.fit(X_train, y_train)
        y_pred_hgb = model.predict(X_val)
        rmse_hgb = mean_squared_error(y_val, y_pred_hgb) ** 0.5
        return rmse_hgb, y_pred_hgb

    @staticmethod
    def predict_elasticnet(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        imputer = SimpleImputer(strategy="mean")
        base_model = make_pipeline(imputer, ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000))
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        y_pred_elas = model.predict(X_val)
        rmse_elas = mean_squared_error(y_val, y_pred_elas) ** 0.5
        return rmse_elas, y_pred_elas




class DataPreprocessor:
    def __init__(self):
        self.y_scaler = StandardScaler()
        self.x_scaler = StandardScaler()

    def join_logs_and_wafer_df(self, log_df: pl.DataFrame, wafer_df: pl.DataFrame, y_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        X_full = log_df.join(wafer_df, on="marathon_run", how="inner", suffix="_df2")
        X_full_pd = X_full.to_pandas()
        y_full_pd = y_df.sort("marathon_run").drop("marathon_run").to_pandas()
        return X_full_pd, y_full_pd

    def scale_and_split_data(self, X_full_pd: pl.DataFrame, y_full_pd: pl.DataFrame) -> Tuple[pl.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, StandardScaler]:
        y_full_scaled = self.y_scaler.fit_transform(y_full_pd)
        X_train, X_val, y_train, y_val = train_test_split(X_full_pd, y_full_scaled, test_size=0.2, random_state=42)

        cols_to_scale = [c for c in X_train.select_dtypes(include=np.number).columns if c != "marathon_run"]
        X_train[cols_to_scale] = self.x_scaler.fit_transform(X_train[cols_to_scale])
        X_val[cols_to_scale] = self.x_scaler.transform(X_val[cols_to_scale])

        X_train_final = X_train.drop(columns=["marathon_run"])
        X_val_final = X_val.drop(columns=["marathon_run"])

        return X_train_final, y_train, X_val_final, y_val, self.y_scaler

