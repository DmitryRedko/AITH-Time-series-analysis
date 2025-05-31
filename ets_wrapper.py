from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
import numpy as np

class ETSWrapper:
    def __init__(self, 
                 trend=None, 
                 seasonal=None, 
                 seasonal_periods=None, 
                 damped_trend=False, 
                 auto_fit_mode= False, 
                 auto_fit_seasonal_periods_candidates = [7, 14, 30, 90, 365],
                 verbose=False,
                 metric='aic',
                 trends = [None, 'add', 'mul'],
                 seasonals = [None, 'add', 'mul'],
                 damped_opts = [False, True]
        ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.model = None
        self.results = None
        self.auto_fit_mode = auto_fit_mode
        self.auto_fit_seasonal_periods_candidates = auto_fit_seasonal_periods_candidates
        self.verbose=verbose
        self.metric = metric
        self.trends = trends
        self.seasonals = seasonals
        self.damped_opts = damped_opts
    
    def fit(self, y):

        if self.auto_fit_mode:
            self.auto_fit(y)
        else:
            self.model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend
            )
            self.results = self.model.fit()
        
        return self
    
    def auto_fit(self, y):
        best_score = np.inf
        best_cfg = None

        for trend in self.trends:
            for seasonal in self.seasonals:
                for sp in self.auto_fit_seasonal_periods_candidates:
                    for damped in self.damped_opts:
                        if seasonal is not None and sp is None:
                            continue
                        try:
                            model = ExponentialSmoothing(
                                y,
                                trend=trend,
                                seasonal=seasonal,
                                seasonal_periods=sp,
                                damped_trend=damped
                            ).fit()
                            score = getattr(model, self.metric)
                            if score < best_score:
                                best_score = score
                                best_cfg = (trend, seasonal, sp, damped)
                            if self.verbose:
                                print(f"✅ {trend=}, {seasonal=}, {sp=}, {damped=} → {self.metric}={score:.2f}")
                        except Exception as e:
                            if self.verbose:
                                print(f"{trend=}, {seasonal=}, {sp=}, {damped=} → ошибка: {e}")

        if best_cfg is None:
            raise ValueError("Не удалось подобрать ни одну конфигурацию ETS")


        print(f"Лучшая конфигурация: {best_cfg} с {self.metric}={best_score:.2f}")

        self.trend, self.seasonal, self.seasonal_periods, self.damped_trend = best_cfg

        self.model = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend
        )
        self.results = self.model.fit()

        return self


    def predict(self, steps, exog = None):
        if self.results is None:
            raise ValueError("Модель не обучена")
        return self.results.forecast(steps)

    def save(self, path):
        if self.results is None:
            raise ValueError("Нечего сохранять — модель не обучена")
        joblib.dump(self.results, path)

    def load(self, path):
        self.results = joblib.load(path)
        return self
