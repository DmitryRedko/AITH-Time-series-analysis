from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

class SARIMAXWrapper:
    def __init__(self, order=(1, 0, 0), simple_differencing=False, seasonal_order=(0, 0, 0, 0),  exog = None, enforce_stationarity=True, enforce_invertibility=True):
        self.model = None
        self.results = None
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.exog = exog
        self.simple_differencing = simple_differencing

    def fit(self, y):
        self.model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            exog = self.exog,
            simple_differencing = self.simple_differencing
        )
        self.results = self.model.fit(disp=False)
        return self

    def predict(self, steps, exog=None):
        if exog is None:
            return self.results.get_forecast(steps=steps).predicted_mean
        else:
            return self.results.get_forecast(steps=steps, exog=exog).predicted_mean

    def save(self, path):
        if self.results is None:
            raise ValueError("Нечего сохранять — модель не обучена")
        joblib.dump(self.results, path)

    def load(self, path):
        self.results = joblib.load(path)
        return self
