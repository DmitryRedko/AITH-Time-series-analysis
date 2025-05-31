# TimeSeriesForecastingToolkit

`TimeSeriesForecastingToolkit` — класс для предобработки, анализа, трансформации и прогнозирования временных рядов с использованием различных моделей (SARIMAX, Prophet, ETS, градиентный бустинг).

## Инициализация

```python
TimeSeriesForecastingToolkit(df, date_col='date', event_cols=None, price_col='sell_price', period=7, year=None)
```

* `df`: pandas DataFrame
* `date_col`: имя колонки с датами
* `event_cols`: список дополнительных регрессоров
* `price_col`: имя колонки с ценами/значениями
* `period`: период сезонности
* `year`: фильтрация по году (оставляет только строки после указанного года)

---

## Методы

### Анализ и визуализация

#### `analyze(col=None, period=None, custom_df=None)`

Анализ временного ряда: исходный график, скользящие средние, STL-декомпозиция с визуализацией по сезонам.

**Параметры:**

* `col`: название целевой колонки
* `period`: период сезонности
* `custom_df`: DataFrame для анализа (если не указан, используется self.df)

**Функции:**

* Строит график исходного ряда, выделяя события и сезоны
* Строит графики скользящих средних (7, 30, 90 дней)
* Выполняет STL-декомпозицию и отображает её компоненты

---

### Трансформации

#### `log_transform(col)`

Применяет логарифмическое преобразование `log(x+1)` к колонке `col`.

#### `boxcox_transform(col)`

Применяет Box-Cox трансформацию к `col`. Автоматически корректирует нулевые и отрицательные значения.

#### `yeojohnson_transform(col)`

Применяет Yeo-Johnson трансформацию к `col`. Подходит для значений любого знака.

#### `inverse_log(series)`

Обратное преобразование логарифма `exp(series) - 1`.

#### `inverse_boxcox(series)`

Обратное преобразование для Box-Cox с учетом сдвига и лямбда.

#### `inverse_yeojohnson(series)`

Обратное преобразование для Yeo-Johnson.

---

### Предобработка

#### `remove_outliers(col='cnt', method='interpolate', window=5)`

Удаляет выбросы методом интерполяции или заменой на медиану скользящего окна.

#### `replace_outlier_streaks_with_rolling_mean(...)`

Заменяет длинные участки нулей или малых значений на скользящее среднее.

#### `interpolate_missing(col, method='linear', ...)`

Интерполирует пропущенные значения в колонке `col`.

---

### Стационарность и декомпозиция

#### `stationarity_tests(d=0, D=0, period=None, col=None, plot=False)`

Проверка стационарности ряда с помощью тестов KPSS и ADF. Может отображать ACF и PACF графики.

#### `stl_decompose_transform(target_col_name, period=None)`

Применяет STL-декомпозицию ряда на тренд, сезонность и остатки.

---

### Обучение и прогнозирование

#### `fit(col, model_type="sarimax", exog=None, val_data=None, **kwargs)`

Обучение модели (`sarimax`, `prophet`, `boosting`, `ets`) на колонке `col`.

#### `predict(steps, exog=None)`

Получение прогноза на `steps` шагов вперёд.

#### `save_model(path)` / `load_model(path, model_type="sarimax")`

Сохранение и загрузка модели.

---

### Оценка качества

#### `evaluate_forecast(y_true, y_pred, verbose=True, plot=False)`

Рассчитывает MAE и RMSE, может построить график факта и прогноза.

#### `check_residuals(lags=40, plot=False, ...)`

Выполняет комплексную диагностику остатков модели:

* t-тест
* Wilcoxon-тест
* Jarque-Bera
* Ljung-Box
* Breusch-Pagan

Возвращает остатки и/или p-value всех тестов.

---

## Зависимости

* `pandas`, `numpy`, `scipy`, `seaborn`, `matplotlib`
* `statsmodels`, `sklearn`
* Собственные обёртки: `SARIMAXWrapper`, `ProphetWrapper`, `ETSWrapper`, `BoostingWrapper`

---

## Пример использования

```python
toolkit = TimeSeriesForecastingToolkit(df, date_col='date')
toolkit.yeojohnson_transform('cnt')
toolkit.fit(col='yeojohnson_cnt', model_type='sarimax', order=(1,1,1))
preds = toolkit.predict(steps=30)
toolkit.evaluate_forecast(y_true, preds)
toolkit.check_residuals()
```

---

## Обёртки моделей

### `ETSWrapper`

Обёртка для модели экспоненциального сглаживания (Exponential Smoothing, Holt-Winters).

**Параметры и поведение:**

* `trend`, `seasonal`, `seasonal_periods`, `damped_trend`: параметры модели, как в `ExponentialSmoothing`
* `auto_fit_mode`: если True, автоматически подбираются параметры по метрике (по умолчанию AIC)
* `auto_fit_seasonal_periods_candidates`: список кандидатов для автоподбора периодов сезонности
* `trends`, `seasonals`, `damped_opts`: список возможных вариантов для автоподбора
* `metric`: метрика оптимизации (`aic`, `bic`, `hqic` и т.д.)
* `verbose`: флаг логирования в процессе автоподбора

**Методы:**

* `fit(y)`: обучение модели; при `auto_fit_mode=True` запускает перебор конфигураций
* `predict(steps)`: прогноз на `steps` шагов вперёд
* `save(path)`: сохранение обученной модели с помощью `joblib`
* `load(path)`: загрузка модели из файла

**Примечание:** при автоподборе выводится лучшая конфигурация и её значение метрики.

### `ProphetWrapper`

Обёртка для модели Prophet от Facebook/Meta.

**Параметры:**

* Все параметры инициализации соответствуют Prophet (growth, seasonality, holidays, changepoints и др.)
* `exog`: список дополнительных регрессоров
* `custom_seasonalities`: возможность задать пользовательские сезонности

**Методы:**

* `fit(y, exog=None)`: обучение модели, включая регрессоры (если заданы)
* `predict(steps, exog_predict=None)`: прогноз на `steps` шагов вперёд с учётом регрессоров
* `save(path)`: сохранение модели с помощью `joblib`
* `load(path)`: загрузка модели из файла и восстановление состояния

**Примечание:**

* Регрессоры должны присутствовать как при обучении, так и при прогнозировании
* Метод `predict` возвращает только значения прогноза `yhat`

### `SARIMAXWrapper`

Обёртка для модели SARIMAX из `statsmodels`.

**Параметры:**

* `order`: параметры ARIMA (p,d,q)
* `seasonal_order`: сезонные параметры (P,D,Q,s)
* `exog`: внешние регрессоры
* `enforce_stationarity`: требовать ли стационарность модели
* `enforce_invertibility`: требовать ли инвертируемость
* `simple_differencing`: использовать ли простое дифференцирование

**Методы:**

* `fit(y)`: обучение модели
* `predict(steps, exog=None)`: прогноз на `steps` шагов вперёд; можно передать внешние регрессоры
* `save(path)`: сохранение модели с помощью `joblib`
* `load(path)`: загрузка модели из файла

### `BoostingWrapper`

Обёртка для модели градиентного бустинга на основе LightGBM.

**Параметры:**

* `params`: словарь параметров LightGBM (по умолчанию — регрессия с RMSE)
* `num_boost_round`: количество итераций бустинга
* `early_stopping_rounds`: количество итераций без улучшения на валидации до остановки
* `verbose_eval`: логировать ли процесс обучения

**Методы:**

* `fit(y, exog=None, val_data=None)`: обучение модели; можно передать валидационные данные `(X_val, y_val)` для ранней остановки
* `predict(steps, exog=None)`: прогноз на `steps` шагов вперёд (обязателен `exog`, если использовался при обучении)
* `save(path)`: сохранение модели и данных с помощью `joblib`
* `load(path)`: загрузка модели и восстановление состояния

**Примечание:**

* `exog` должен быть передан как при обучении, так и при прогнозировании, если он использовался
* `predict` возвращает значения прогноза в виде `pd.Series`

