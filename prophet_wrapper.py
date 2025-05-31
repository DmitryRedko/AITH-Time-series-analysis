import logging
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
import pandas as pd

from prophet import Prophet
import joblib

class ProphetWrapper:
    def __init__(self, 
                 exog=None, 
                 growth: str = 'linear',
                 changepoints:  None = None,
                 n_changepoints: int = 25,
                 changepoint_range: float = 0.8,
                 yearly_seasonality: str = 'auto',
                 weekly_seasonality: str = 'auto',
                 daily_seasonality: str = 'auto',
                 holidays:  None = None,
                 seasonality_mode: str = 'additive',
                 seasonality_prior_scale: float = 10,
                 holidays_prior_scale: float = 10,
                 changepoint_prior_scale: float = 0.05,
                 mcmc_samples: int = 0,
                 interval_width: float = 0.8,
                 uncertainty_samples: int = 1000,
                 stan_backend:  None = None,
                 scaling: str = 'absmax',
                 holidays_mode:  None = None,
                 custom_seasonalities: list = None 
                ):
        
        self.model = Prophet(
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holidays,
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            mcmc_samples=mcmc_samples,
            interval_width=interval_width,
            uncertainty_samples=uncertainty_samples,
            stan_backend=stan_backend,
            scaling=scaling,
            holidays_mode=holidays_mode
        )

        if custom_seasonalities:
            for seasonality in custom_seasonalities:
                self.model.add_seasonality(
                    name=seasonality.get('name'),
                    period=seasonality.get('period'),
                    fourier_order=seasonality.get('fourier_order', 5), 
                    prior_scale=seasonality.get('prior_scale', 10) 
                )

        self.fitted = False
        self.results = None
        self.exog = None
        self.regressors = exog


    def fit(self, y, exog=None):
        if self.regressors:
            if exog is None:
                raise ValueError(f"Требуются дополнительные регрессоры: {self.regressors}")
            self.exog = exog
            missing = [reg for reg in self.regressors if reg not in exog.columns]
            if missing:
                raise ValueError(f"Отсутствуют необходимые регрессоры в обучающих данных: {missing}")

            for reg in self.regressors:
                self.model.add_regressor(reg)
                y[reg] = exog[reg].values  

        self.results = self.model.fit(y)
        self.fitted = True
        return self


    def predict(self, steps, exog_predict=None):
        if not self.fitted:
            raise ValueError("Модель не обучена")

        exog = None
        if self.regressors:
            if exog_predict is None:
                raise ValueError(f"Требуются дополнительные регрессоры: {self.regressors}")

            exog = pd.concat([self.exog, exog_predict], axis=0).reset_index(drop=True)
            exog = exog.astype(float)

            missing = [reg for reg in self.regressors if reg not in exog_predict.columns]
            if missing:
                raise ValueError(f"Отсутствуют необходимые регрессоры: {missing}")

        future = self.model.make_future_dataframe(periods=steps, freq='D')

        if self.regressors:
            if len(exog) != len(future):
                raise ValueError(f"Длина exog ({len(exog)}) не совпадает с длиной future ({len(future)})")
            for reg in self.regressors:
                future[reg] = exog[reg].values

        forecast = self.model.predict(future)
        return forecast['yhat'][-steps:].reset_index(drop=True)


    def save(self, path):
        if not self.fitted:
            raise ValueError("Нечего сохранять — модель не обучена")
        joblib.dump((self.model, self.exog, self.regressors), path)

    def load(self, path):
        self.model, self.exog, self.regressors = joblib.load(path)
        self.fitted = True
        return self
