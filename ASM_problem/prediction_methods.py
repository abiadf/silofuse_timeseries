
import numpy as np
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from lightgbm import LGBMRegressor, early_stopping
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


class MultiOutputModelPredictor:
    def __init__(self, device):
        self.device = device

    @staticmethod
    def predict_lightgbm(X_train, y_train, X_val, y_val):
        n_targets = y_train.shape[1]
        y_pred    = np.zeros(y_val.shape)
        
        for i in range(n_targets):
            model = LGBMRegressor(objective='regression',
                                verbosity=-1,
                                n_estimators=1000,
                                learning_rate=0.05,
                                num_leaves=31,
                                max_depth=-1,
                                min_data_in_leaf=20)
            model.fit(X_train, y_train[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
            y_pred[:, i] = model.predict(X_val)
        
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        return rmse, y_pred

    def predict_catboost(self, X_train, y_train, X_val, y_val, device):
        catboost_task_type = 'GPU' if device.type == 'cuda' else 'CPU'
        model = MultiOutputRegressor(cb.CatBoostRegressor(verbose   = 0,
                                                        iterations= 100,
                                                        task_type = catboost_task_type))
        model.fit(X_train, y_train)
        y_pred_cat = model.predict(X_val)
        rmse_cat = mean_squared_error(y_val, y_pred_cat) ** 0.5
        return rmse_cat, y_pred_cat

    @staticmethod
    def predict_xgboost(X_train, y_train, X_val, y_val):
        model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', verbosity=0))
        model.fit(X_train, y_train)
        y_pred_xgb = model.predict(X_val)
        rmse_xgb = mean_squared_error(y_val, y_pred_xgb) ** 0.5
        return rmse_xgb, y_pred_xgb

    @staticmethod
    def predict_randomforest(X_train, y_train, X_val, y_val):
        rf_model  = MultiOutputRegressor(RandomForestRegressor(random_state=42))
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_val)
        rmse_rf   = mean_squared_error(y_val, y_pred_rf) ** 0.5
        return rmse_rf, y_pred_rf

    @staticmethod
    def predict_hgb(X_train, y_train, X_val, y_val):
        model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100))
        model.fit(X_train, y_train)
        y_pred_hgb = model.predict(X_val)
        rmse_hgb = mean_squared_error(y_val, y_pred_hgb) ** 0.5
        return rmse_hgb, y_pred_hgb

    @staticmethod
    def predict_elasticnet(X_train, y_train, X_val, y_val):
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

    def join_logs_and_wafer_df(self, log_df, wafer_df, y_df):
        X_full = log_df.join(wafer_df, on="marathon_run", how="inner", suffix="_df2")
        X_full_pd = X_full.to_pandas()
        y_full_pd = y_df.sort("marathon_run").drop("marathon_run").to_pandas()
        return X_full_pd, y_full_pd

    def scale_and_split_data(self, X_full_pd, y_full_pd):
        y_full_scaled = self.y_scaler.fit_transform(y_full_pd)
        X_train, X_val, y_train, y_val = train_test_split(X_full_pd, y_full_scaled, test_size=0.2, random_state=42)

        cols_to_scale = [c for c in X_train.select_dtypes(include=np.number).columns if c != "marathon_run"]
        X_train[cols_to_scale] = self.x_scaler.fit_transform(X_train[cols_to_scale])
        X_val[cols_to_scale] = self.x_scaler.transform(X_val[cols_to_scale])

        X_train_final = X_train.drop(columns=["marathon_run"])
        X_val_final = X_val.drop(columns=["marathon_run"])

        return X_train_final, y_train, X_val_final, y_val, self.y_scaler

