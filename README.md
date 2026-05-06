# House Prices — Kaggle ML Project

Проект по соревнованию **House Prices: Advanced Regression Techniques** на Kaggle.  
Цель — предсказать `SalePrice` для домов в Ames, Iowa.

## Что реализовано

В репозитории собран воспроизводимый ML-пайплайн для табличной регрессии:

- конфигурирование экспериментов через `configs.py`
- запуск пайплайна через `main.py`
- feature engineering и preprocessing
- обучение classical ML и DNN
- кросс-валидация
- логирование результатов экспериментов
- сохранение артефактов моделей
- генерация submission-файлов для Kaggle

### Реализованные модели

**Baselines**
- Linear Regression
- Ridge
- Lasso
- ElasticNet
- KNN

**Boosting / Tree-based**
- RandomForestRegressor
- GradientBoostingRegressor
- CatBoostRegressor

**Deep Learning**
- DNN
- DNN random search

**Ensemble**
- Average ensemble

---

## Структура проекта

```text
House-Prices-kaggle-/
├── checkpoints/              # сохраненные артефакты моделей
├── data/                     # train.csv / test.csv
├── notebooks/
│   ├── EDA_RESULTS.ipynb     # EDA + все CV-результаты + выводы
│   └── df_report.html        # автоотчет по данным
├── src/
│   ├── classic_runner.py
│   ├── dnn_runner.py
│   ├── data.py
│   ├── ensemble.py
│   ├── features.py
│   ├── metrics.py
│   ├── model.py
│   ├── pipeline.py
│   ├── schema.py
│   ├── search_spaces.py
│   └── utils.py
├── configs.py
├── main.py
├── requirements.txt
└── checklist.md
```

---

## Установка

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Запуск

Основной запуск пайплайна:

```bash
python main.py
```

Через `configs.py` можно менять:
- модель
- режим запуска
- гиперпараметры
- seed
- параметры DNN
- пути к данным и артефактам

---

## Данные и постановка задачи

Соревнование: **House Prices - Advanced Regression Techniques**

Целевая переменная:
- `SalePrice`

Метрика соревнования:
- **RMSE в логарифмах цены**

В проекте для classical ML и DNN используется обучение на логарифме таргета с обратным преобразованием при инференсе.

---

## Feature Engineering и preprocessing

В проекте реализованы:

- обработка пропусков
- выделение числовых, категориальных и ordinal-признаков
- ordinal encoding для упорядоченных признаков
- one-hot encoding для номинальных признаков
- scaling для числовых признаков
- `log1p` для части skewed numerical features
- отдельная логика для DNN dataloaders

Подробные аналитические выводы, EDA и все CV-эксперименты собраны в:

- `notebooks/EDA_RESULTS.ipynb`

---

## Эксперименты

Информация по всем экспериментам лежит в `EDA_RESULTS.ipynb`.

### Что означают версии экспериментов

- **baseline** — модели обучены с начальным гридом гиперпараметров
- **v1** — грид гиперпараметров расширен
- **v2** — добавлен отбор и обучение только на статистически значимых признаках
- **v3** — логарифмированы некоторые числовые признаки
- **v4** — выделены ordinal-признаки

### Лучшие CV-результаты по моделям

| Model | Best CV score |
|---|---:|
| CatBoost | **0.116480** |
| GradientBoosting | 0.119048 |
| ElasticNet | 0.123189 |
| Lasso | 0.125028 |
| Ridge | 0.126159 |
| RandomForest | 0.137007 |
| Linear | 0.139300 |
| KNN | 0.161882 |
| DNN | 0.199666 |
| DNN_RANDOM | 0.211087 |

### Сводка по экспериментам

| Model | baseline | v1 | v2 | v3 | v4 | best_score |
|---|---:|---:|---:|---:|---:|---:|
| DNN_RANDOM | NaN | 0.211087 | NaN | NaN | NaN | 0.211087 |
| DNN | 0.339993 | 0.199666 | NaN | NaN | NaN | 0.199666 |
| KNN | 0.165621 | 0.165422 | 0.161882 | 0.165657 | 0.164943 | 0.161882 |
| Linear | 0.156059 | 0.156059 | 0.156497 | 0.160954 | 0.139300 | 0.139300 |
| RandomForest | 0.142380 | 0.138055 | 0.143534 | 0.138093 | 0.137007 | 0.137007 |
| Ridge | 0.151005 | 0.138651 | 0.154802 | 0.126623 | 0.126159 | 0.126159 |
| Lasso | 0.137754 | 0.137754 | 0.155338 | 0.125028 | 0.125779 | 0.125028 |
| ElasticNet | 0.137126 | 0.136483 | 0.156289 | 0.123189 | 0.124063 | 0.123189 |
| GradientBoosting | 0.122056 | 0.120006 | 0.135520 | 0.120109 | 0.119048 | 0.119048 |
| CatBoost | 0.119265 | 0.116480 | 0.132550 | 0.116486 | 0.116596 | 0.116480 |

### Короткие выводы по CV

- Лучший CV показал **CatBoost**: `0.116480`
- Второй лучший результат у **GradientBoosting**: `0.119048`
- Среди линейных моделей лучший результат у **ElasticNet**: `0.123189`
- DNN в текущей реализации заметно уступает classical tabular models

---

## Kaggle submission results

### Public Leaderboard

| Submission | Public Score |
|---|---:|
| average_ensemble_v4_prediction.csv | **0.12138** |
| average_ensemble_v4_prediction.csv | 0.12338 |
| Ridge_v3_prediction.csv | 0.12403 |
| CatBoost_v1_prediction.csv | 0.12457 |
| Lasso_v4_prediction.csv | 0.12511 |
| CatBoost_v4_prediction.csv | 0.12527 |
| average_ensemble_v2_prediction.csv | 0.12527 |
| CatBoost_baseline_prediction.csv | 0.12740 |
| Lasso_v3_prediction.csv | 0.12812 |
| average_ensemble_v2_prediction.csv | 0.12822 |
| GradientBoosting_v4_prediction.csv | 0.12860 |
| GradientBoosting_v1_prediction.csv | 0.13290 |
| GradientBoosting_baseline_prediction.csv | 0.13290 |
| DNN_v1_prediction.csv | 0.21630 |
| DNN_RANDOM_v1_prediction.csv | 0.27566 |
| average_ensemble_v2_prediction.csv | 9.45963 |

### Итог по test / leaderboard

На тесте лучшей моделью стал **Ensemble**,  
**Public score = 0.12138**

---

## Как воспроизвести результат

1. Установить зависимости
2. Проверить пути к `train.csv` и `test.csv` в `configs.py`
3. Выбрать модель и режим запуска в `configs.py`
4. Запустить:

```bash
python main.py
```

После запуска:
- результаты экспериментов логируются
- лучшие артефакты сохраняются в `checkpoints/`
- submission-файлы сохраняются для отправки на Kaggle

---

## Основные выводы

- Лучший CV-результат показал **CatBoost**
- Лучший public leaderboard результат показал **average ensemble**
- Выделение ordinal-признаков и логарифмирование части числовых признаков улучшили качество ряда моделей
- DNN в текущей версии не смог обогнать сильные классические табличные модели

---

## Что можно улучшить дальше

- отдельная нативная ветка для CatBoost без внешнего OHE
- более сильный stacking / blending ensemble
- улучшение DNN-архитектуры и search space
- ранняя остановка и более системный tracking экспериментов

---

## Автор

Ruslan / rusiich
