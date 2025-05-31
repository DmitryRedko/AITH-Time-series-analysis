import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd

class BoostingWrapper:
    def __init__(self, params=None, num_boost_round=100, early_stopping_rounds=10, verbose_eval=False):
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'seed': 42
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval

        self.model = None
        self.fitted = False
        self.residuals = None

        self.lag_features = [1, 7, 30, 90]
        self.exog_features = []  # будет определён в fit

    def fit(self, y, exog=None, val_data=None):
        self.y = y
        self.exog = exog

        # Сохраняем список экзогенных признаков
        if exog is not None:
            self.exog_features = list(exog.columns)

        df = self.make_features(y, exog)
        X_train = df.drop(columns=["y"])
        y_train = df["y"]

        dtrain = lgb.Dataset(X_train, label=y_train)

        callbacks = []
        valid_sets = None
        if val_data:
            X_val, y_val = val_data
            dval = lgb.Dataset(X_val, label=y_val)
            valid_sets = [dtrain, dval]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        callbacks.append(lgb.log_evaluation(period=1 if self.verbose_eval else 0))

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks
        )

        y_pred_train = self.model.predict(X_train, num_iteration=self.model.best_iteration)
        self.residuals = y_train - y_pred_train
        self.last_y = y.copy()
        self.fitted = True
        return self

    def make_features(self, y_series, exog):
        df = pd.DataFrame(index=y_series.index)
        df["y"] = y_series

        for lag in self.lag_features:
            df[f"lag_{lag}"] = y_series.shift(lag)

        if exog is not None:
            for col in self.exog_features:
                df[col] = exog[col]

        return df.dropna()

    def predict(self, steps, exog_future):
        if not self.fitted:
            raise ValueError("Модель не обучена")

        history = self.last_y.copy()
        preds = []

        for i in range(steps):
            current_time = exog_future.index[i]
            exog_row = exog_future.iloc[i]

            features = {}

            # Лаги
            for lag in self.lag_features:
                if len(history) >= lag:
                    features[f"lag_{lag}"] = history.iloc[-lag]
                else:
                    features[f"lag_{lag}"] = np.nan

            # Экзогенные признаки
            for col in self.exog_features:
                features[col] = exog_row[col]

            X_pred = pd.DataFrame([features])
            y_hat = self.model.predict(X_pred, num_iteration=self.model.best_iteration)[0]

            history.loc[current_time] = y_hat
            preds.append((current_time, y_hat))

        return pd.Series(dict(preds))

    def save(self, path):
        if not self.fitted:
            raise ValueError("Нечего сохранять — модель не обучена")
        joblib.dump((self.model, self.last_y, self.exog, self.residuals), path)

    def load(self, path):
        self.model, self.last_y, self.exog, self.residuals = joblib.load(path)
        self.fitted = True
        return self
