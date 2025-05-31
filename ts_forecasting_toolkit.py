import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PowerTransformer
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
import seaborn as sns
from scipy.stats import ttest_1samp, wilcoxon
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.ndimage import median_filter
from sarimax_wrapper import SARIMAXWrapper
from prophet_wrapper import ProphetWrapper
from ets_wrapper import ETSWrapper
from boosting_wrapper import BoostingWrapper


class TimeSeriesForecastingToolkit:
    def __init__(self, df, date_col='date', event_cols=None, price_col='sell_price', period=7, year = None):
        self.df = df.copy()

        if year:
            self.df = self.df[self.df[date_col]>str(year)]

        self.date_col = date_col
        self.event_cols = event_cols or []
        self.price_col = price_col
        self.period = period
        self.lambda_ = None
        self.offset = 0
        self.model = None

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df.set_index(self.date_col, inplace=True)
        self.df['month'] = self.df.index.month
        self.df['year'] = self.df.index.year
        self.df['season'] = self.df['month'].apply(self.get_season)

    def get_season(self, month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    def log_transform(self, col):
        self.df['log_cnt'] = np.log(self.df[col] + 1)

    def boxcox_transform(self, col):
        y = self.df[col].copy()
        if (y <= 0).any():
            self.offset = abs(y.min()) + 1e-6
            y += self.offset
        y_transformed, self.lambda_ = boxcox(y)
        self.df['boxcox_cnt'] = y_transformed

    def inverse_log(self, series):
        return np.expm1(series)

    def inverse_boxcox(self, series):
        if self.lambda_ == 0:
            restored = np.exp(series)
        else:
            restored = np.power(series * self.lambda_ + 1, 1 / self.lambda_)
        return restored - self.offset

    def yeojohnson_transform(self, col):
        y = self.df[col].values.reshape(-1, 1)
        self.yeojohnson_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        y_transformed = self.yeojohnson_transformer.fit_transform(y)
        self.df['yeojohnson_cnt'] = y_transformed.flatten()

    def inverse_yeojohnson(self, series):
        y_inv = self.yeojohnson_transformer.inverse_transform(series.values.reshape(-1, 1))
        return y_inv.flatten()

    def get_df(self):
        return self.df

    def analyze(self, col=None, period = None, custom_df = None):
        
        if(custom_df is None):
            print("Use self-module df")
            df = self.df.copy()
        else:
            df = custom_df.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df.set_index(self.date_col, inplace=True)
            df['month'] = df.index.month
            df['year'] = df.index.year
            df['season'] = df['month'].apply(self.get_season)
        
        if period is None:
            period = self.period 

        if col is None:
            raise ValueError("Target column should be set")
    
        df[f"rolling_{col}_7"] = df[col].rolling(window=7, center=True).mean()
        df[f"rolling_{col}_30"] = df[col].rolling(window=30, center=True).mean()
        df[f"rolling_{col}_90"] = df[col].rolling(window=90, center=True).mean()

        stl = STL(df[col], period=period)
        res = stl.fit()
        df["trend"] = res.trend
        df["seasonal"] = res.seasonal
        df["resid"] = res.resid
        
        df["has_event"] = df[self.event_cols].notnull().any(axis=1) if self.event_cols else False
        season_colors = {
            "winter": "blue", "spring": "yellow", "summer": "green", "autumn": "red"
        }
        month_starts = df.drop_duplicates(subset=["year", "month"], keep="first")

        fig, ax1 = plt.subplots(figsize=(20, 6))
        for i in range(len(month_starts) - 1):
            ax1.axvspan(month_starts.index[i], month_starts.index[i + 1], color=season_colors[month_starts.iloc[i]["season"]], alpha=0.03)

        ax1.plot(df.index, df[col], label=col, color="black", linewidth=0.5)
        ax1.scatter(df[df["has_event"]].index, df[df["has_event"]][col], color='blue', label='Event', s=12)

        for x in month_starts.index:
            ax1.axvline(x=x, color='gray', linestyle='--', linewidth=0.7)

        if self.price_col in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(df.index, df[self.price_col], label=self.price_col, color='red', linewidth=1.2)
            ax2.set_ylabel(self.price_col)
            ax2.legend(loc='upper right')

        ax1.set_ylabel(col)
        ax1.legend(loc='upper left')
        plt.title("1. Оригинальный ряд")
        plt.tight_layout()
        plt.show()


        fig, ax = plt.subplots(figsize=(20, 6))
        for i in range(len(month_starts) - 1):
            ax.axvspan(month_starts.index[i], month_starts.index[i + 1], color=season_colors[month_starts.iloc[i]["season"]], alpha=0.03)

        ax.plot(df.index, df[col], label=col, color="black", linewidth=0.3)
        ax.plot(df.index, df[f"rolling_{col}_7"], label="Rolling Mean (7)", color="blue", linewidth=2, alpha=0.5)
        ax.plot(df.index, df[f"rolling_{col}_30"], label="Rolling Mean (30)", color="red", linewidth=2, alpha=0.5)
        ax.plot(df.index, df[f"rolling_{col}_90"], label="Rolling Mean (90)", color="green", linewidth=2, alpha=0.5)

        for x in month_starts.index:
            ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.7)

        ax.set_ylabel(col)
        ax.legend()
        plt.title("2. Скользящее среднее")
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
        for i, comp in enumerate([col, "trend", "seasonal", "resid"]):
            ax = axes[i]
            for j in range(len(month_starts) - 1):
                ax.axvspan(month_starts.index[j], month_starts.index[j + 1], color=season_colors[month_starts.iloc[j]["season"]], alpha=0.03)
            ax.plot(df.index, df[comp], color=['black', 'orange', 'green', 'red'][i])
            for x in month_starts.index:
                ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.7)
            ax.set_ylabel(comp)
            ax.set_title(comp.capitalize())
        plt.suptitle("3. STL-декомпозиция", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

    def remove_outliers(self, col='cnt', method="interpolate", window=5):
        ts = self.df[col].copy()

        q1 = ts.quantile(0.25)
        q3 = ts.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        is_outlier = (ts < lower) | (ts > upper)

        if method == "interpolate":
            ts_clean = ts.mask(is_outlier)
            ts_clean = ts_clean.interpolate(method='linear', limit_direction='both')
        elif method == "median":
            rolling_median = ts.rolling(window=window, center=True, min_periods=1).median()
            ts_clean = ts.copy()
            ts_clean[is_outlier] = rolling_median[is_outlier]
        else:
            raise ValueError(f"Unknown outlier removal method: {method}")

        self.df[col] = ts_clean

    def replace_outlier_streaks_with_rolling_mean(self, col, zero_streak=14, lower_bound=0, window=30, padding=2):
        series = self.df[col].copy()
        rolling = series.rolling(window=window, center=True, min_periods=1).mean()

        streak = 0
        segments = []

        for i, val in enumerate(series):
            if val <= lower_bound:
                streak += 1
            else:
                if streak >= zero_streak:
                    segments.append((i - streak, i))
                streak = 0

        if streak >= zero_streak:
            segments.append((len(series) - streak, len(series)))

        to_replace = set()
        for start, end in segments:
            extended_start = max(start - padding, 0)
            extended_end = min(end + padding, len(series))
            to_replace.update(range(extended_start, extended_end))

        to_replace = sorted(to_replace)
        series.iloc[to_replace] = rolling.iloc[to_replace]
        self.df[col] = series

        self.outlier_replace_log = (
            f"✅ Заменено {len(to_replace)} точек на скользящее среднее (окно = {window}) "
            f"в участках длиной ≥ {zero_streak}, где значения ≤ {lower_bound}, с расширением ±{padding}"
        )


    def stationarity_tests(self, d=0, D=0, period=None, col=None, plot=False):

        if col is None: 
            raise ValueError("Target column should be set")
        
        period = period or self.period or 30
        series = self.df[col].copy()
        if(D>0):
            print(f"=== Стационарность: d={d}, D={D}, period={period} ===")
        else:
            print(f"=== Стационарность: d={d}, D={D} ===")

        if d > 0:
            for _ in range(d):
                series = series.diff()
        if D > 0:
            for _ in range(D):
                series = series - series.shift(period)
        series = series.dropna()

        print("\n=== KPSS Test ===")
        statistic, p_value, _, _ = kpss(series)
        print(f"Statistic: {statistic}, p-value: {p_value}")
        print("✅ Стационарен" if p_value >= 0.05 else "❌ Нестационарен")

        print("\n=== ADF Test ===")
        stat, p_val, _, _, _, _ = adfuller(series)
        print(f"Statistic: {stat}, p-value: {p_val}")
        print("✅ Стационарен" if p_val < 0.05 else "❌ Нестационарен")

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))
            plot_acf(series, ax=axes[0], lags=32)
            plot_pacf(series, ax=axes[1], lags=32, method='ywm')
            axes[0].set_title("ACF")
            axes[1].set_title("PACF")
            plt.tight_layout()
            plt.show()

    def stl_decompose_transform(self, target_col_name, period=None):
        period = period or self.period
        series = self.df[target_col_name].copy()
        stl = STL(series, period=period)
        res = stl.fit()
        self.df[f"{target_col_name}_trend"] = res.trend
        self.df[f"{target_col_name}_seasonal"] = res.seasonal
        self.df[f"{target_col_name}_resid"] = res.resid
        print("✅ STL-декомпозиция завершена. Компоненты добавлены в DataFrame.")
        return self.df[[f"{target_col_name}_trend", f"{target_col_name}_seasonal", f"{target_col_name}_resid"]]

    def fit(self, col, model_type="sarimax", verbose=False, exog=None,  val_data=None, **kwargs):
        y = self.df[col].dropna()
        
        if model_type == "sarimax":
            self.model = SARIMAXWrapper(exog=exog, **kwargs).fit(y)
            if verbose:
                print(f"✅ SARIMAX-модель обучена")

        elif model_type == "prophet":
            df_prophet = y.reset_index().rename(columns={self.date_col: 'ds', col: 'y'})
            regressors = exog.columns.tolist() if exog is not None else None
            self.model = ProphetWrapper(exog=regressors, **kwargs).fit(df_prophet, exog=exog)
            if verbose:
                print(f"✅ Prophet-модель обучена")

        elif model_type == "boosting":
            self.train_target = y
            self.train_exog = exog
            self.model = BoostingWrapper(**kwargs).fit(self.train_target, exog=self.train_exog, val_data=val_data)
            if verbose:
                print(f"✅ Boosting-модель обучена")

        elif model_type == "ets":
            y = y.asfreq('D')  
            y = y.interpolate(method='linear')
            self.model = ETSWrapper(**kwargs).fit(y)
            if verbose:
                print(f"✅ ETS-модель обучена")
        else:
            raise NotImplementedError(f"Модель {model_type} не поддерживается")


    def predict(self, steps, exog=None):
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.predict(steps, exog)

    def save_model(self, path):
        if self.model is None:
            raise ValueError("Нет модели для сохранения")
        self.model.save(path)

    def load_model(self, path, model_type="sarimax"):
        if model_type == "sarimax":
            self.model = SARIMAXWrapper().load(path)
        elif model_type == "prophet":
            self.model = ProphetWrapper().load(path)
        elif model_type == "boosting":
            self.model = BoostingWrapper().load(path)
        elif model_type == "ets":
            self.model = ETSWrapper().load(path)
        else:
            raise NotImplementedError(f"Модель {model_type} не поддерживается")


    def evaluate_forecast(self, y_true, y_pred, verbose=True, title="Forecast vs Actual", plot = False,  figsize = (15,8)):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        if verbose:
            print(f"📊 Оценка прогноза:")
            print(f"MAE  = {mae:.2f}")
            print(f"RMSE = {rmse:.2f}")

        if plot:
            plt.figure(figsize=figsize)
            plt.plot(y_true, label='Actual', color='red')
            plt.plot(y_pred, label='Forecast', color='blue', linestyle='--')
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return {'mae': mae, 'rmse': rmse}

    def check_residuals(self, lags=40, plot=False, custom_residuals = None, verbose = True, return_pvals=True, return_resids = True):
        
        if custom_residuals is None:
            if self.model is None:
                raise ValueError("Модель не обучена")
            if hasattr(self.model, 'results') and hasattr(self.model.results, 'resid'):
                residuals = self.model.results.resid
            elif isinstance(self.model, ProphetWrapper) and self.model.fitted:
                df_train = self.model.model.history.copy()
                forecast_train = self.model.model.predict(df_train)
                residuals = df_train['y'] - forecast_train['yhat']
            elif isinstance(self.model, BoostingWrapper) and self.model.fitted:
                residuals = self.model.residuals
            else:
                raise ValueError("Остатки недоступны для данной модели")
        else:
            residuals = custom_residuals
        
        if residuals is None:
                raise ValueError("Остатки недоступны для данной модели")    

        if plot:
            fig, axes = plt.subplots(1, 4, figsize=(20, 4))

            axes[0].plot(residuals)
            axes[0].set_title("Остатки модели")
            axes[0].set_xlabel("Время")
            axes[0].set_ylabel("Остатки")
            axes[0].grid(True)

            plot_acf(residuals, ax=axes[1], lags=lags)
            axes[1].set_title("ACF остатков")

            sns.histplot(residuals, kde=True, ax=axes[2])
            axes[2].set_title("Гистограмма остатков")

            sm.qqplot(residuals, line='s', ax=axes[3])
            axes[3].set_title("QQ-график остатков")

            plt.tight_layout()
            plt.show()

        
        t_stat, t_pvalue = ttest_1samp(residuals, popmean=0)

        if verbose:
            print(f"t-тест Стьюдента на равенство 0: t-статистика = {t_stat:.3f}, p-значение = {t_pvalue:.4f}")
            if t_pvalue < 0.05:
                print("❌ Среднее значение остатков статистически значимо отличается от 0")
            else:
                print("✅ Среднее значение остатков не отличается от 0")


        wilcoxon_stat, wilcoxon_pvalue = wilcoxon(residuals)

        if verbose and t_pvalue > 0.05:
            try:
                print(f"Критерий Вилкоксона: статистика = {wilcoxon_stat:.3f}, p-значение = {wilcoxon_pvalue:.4f}")
                if wilcoxon_pvalue < 0.05:
                    print("❌ Медиана остатков статистически отличается от 0")
                else:
                    print("✅ Медиана остатков не отличается от 0")
            except ValueError as e:
                print(f"⚠️ Критерий Вилкоксона не применим: {e}")
            
        jb_stat, jb_pvalue, _, _ = jarque_bera(residuals)

        if verbose:
            print(f"Тест Жарка-Бера: статистика = {jb_stat:.2f}, p-значение = {jb_pvalue:.4f}")
            if jb_pvalue < 0.05:
                print("❌ Остатки не распределены нормально")
            else:
                print("✅ Остатки распределены нормально")

        lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        lb_stat = lb_test['lb_stat'].values[0]
        lb_pvalue = lb_test['lb_pvalue'].values[0]

        if verbose:
            print(f"Тест Льюнга-Бокса (lags={lags}): статистика = {lb_stat:.2f}, p-значение = {lb_pvalue:.4f}")
            if lb_pvalue < 0.05:
                print("❌ Остатки автокоррелированы")
            else:
                print("✅ Остатки не автокоррелированы")

        exog_vars = sm.add_constant(np.arange(len(residuals)))
        bp_test = het_breuschpagan(residuals, exog_vars)
        bp_stat, bp_pvalue = bp_test[0], bp_test[1]

        if verbose:
            print(f"Тест Бройша-Пагана: статистика = {bp_stat:.2f}, p-значение = {bp_pvalue:.4f}")
            if bp_pvalue < 0.05:
                print("❌ Есть гетероскедастичность")
            else:
                print("✅ Гомоскедастичность (гетероскедастичность не обнаружена)")

        if return_resids and return_pvals:
            return residuals, [t_pvalue, wilcoxon_pvalue, jb_pvalue, lb_pvalue, bp_pvalue]

        if return_pvals:
            return [t_pvalue, wilcoxon_pvalue, jb_pvalue, lb_pvalue, bp_pvalue]
        
        if return_resids:
            return residuals

    def interpolate_missing(self, col, method='linear', limit_direction='both', order=None, inplace=True):

        if col not in self.df.columns:
            raise ValueError(f"Столбец '{col}' отсутствует в DataFrame.")

        kwargs = {}
        if method in ['polynomial', 'spline']:
            if order is None:
                raise ValueError("Параметр 'order' обязателен для методов 'polynomial' и 'spline'")
            kwargs['order'] = order

        if inplace:
            self.df[col] = self.df[col].interpolate(method=method, limit_direction=limit_direction, **kwargs)
            print(f"✅ Пропущенные значения в '{col}' интерполированы методом '{method}' (order={order if 'order' in kwargs else 'n/a'}).")
        else:
            df_copy = self.df.copy()
            df_copy[col] = df_copy[col].interpolate(method=method, limit_direction=limit_direction, **kwargs)
            print(f"✅ Пропущенные значения в '{col}' интерполированы методом '{method}' (order={order if 'order' in kwargs else 'n/a'}).")
            return df_copy



