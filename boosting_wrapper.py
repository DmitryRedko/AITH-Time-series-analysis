import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd

class BoostingWrapper:
    def __init__(self, params=None, train_target = None, train_exog = None, num_boost_round=100, early_stopping_rounds=10, verbose_eval=False):
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

    def fit(self, y, exog=None, val_data=None):
        """
        y: pd.Series или np.array — целевая переменная
        exog: pd.DataFrame или None — экзогенные признаки 
        val_data: tuple (X_val, y_val) для валидации и ранней остановки
        """
        X = exog if exog is not None else pd.DataFrame(index=y.index)
        dtrain = lgb.Dataset(X, label=y)

        self.train_target = y
        self.train_exog = exog

        valid_sets = None
        if val_data:
            X_val, y_val = val_data
            dval = lgb.Dataset(X_val, label=y_val)
            valid_sets = [dtrain, dval]

        callbacks = []
        if self.early_stopping_rounds and val_data:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
        if self.verbose_eval:
            callbacks.append(lgb.log_evaluation(period=1))
        else:
            callbacks.append(lgb.log_evaluation(period=0))

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks
        )

        self.fitted = True
        return self

    def predict(self, steps, exog=None):
        if not self.fitted:
            raise ValueError("Модель не обучена")

        X_pred = exog if exog is not None else pd.DataFrame(index=range(steps))
        preds = self.model.predict(X_pred, num_iteration=self.model.best_iteration)
        return pd.Series(preds)

    def save(self, path):
        if not self.fitted:
            raise ValueError("Нечего сохранять — модель не обучена")
        joblib.dump((self.model, self.train_exog, self.train_target), path)


    def load(self, path):
        self.model, self.train_exog, self.train_target = joblib.load(path)
        self.fitted = True
        return self
