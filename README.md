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
toolkit.check_residuals(plot=True)
```

---
