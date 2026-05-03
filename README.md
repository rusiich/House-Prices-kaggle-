# Titanic Kaggle Project

Проект для соревнования **Titanic - Machine Learning from Disaster** на Kaggle.

Репозиторий объединяет:
- классические ML-модели через `Pipeline + ColumnTransformer + RandomizedSearchCV`;
- отдельную DNN-ветку на PyTorch с кросс-валидацией, финальным обучением на всём train и предсказанием на test;
- ансамбли: усреднение вероятностей и voting;
- сохранение классических моделей и DNN-артефактов;
- таблицу результатов (`leaderboard.csv`) и ноутбуки с экспериментами.

---

## Результаты

Текущие результаты и история запусков хранятся в:
- `checkpoints/leaderboard.csv`
- `notebooks/EDA_RESULTS.ipynb`

### Лучший результат
- Лучший публичный Kaggle score: **`79,18`**
- Лучшая версия DNN: **`DNN_v2`**
- После перехода от `v1` к `v2` был добавлен дополнительный feature engineering, что дало прирост примерно **на 1 пункт**.

---

## Что умеет проект

### 1. Classic ML
Поддерживается обучение и предсказание для классических моделей, заданных в `src/search_spaces.py` и выбираемых через `config.training.model_name`.

Основная логика:
1. чтение train/test;
2. feature engineering через `FeatureEngineer`;
3. preprocessing через `ColumnTransformer`;
4. подбор гиперпараметров через `RandomizedSearchCV`;
5. сохранение лучшей модели;
6. предсказание на test и запись submission.

Поддерживаемые classic-модели:
- `RFC`
- `LogR`
- `LogR_l1`
- `LogR_elasticnet`
- `KNN`
- `SVC`
- `CatBC`
- `GradBC`
- `LGBMClf`

### 2. DNN
Для нейросети реализован отдельный pipeline:
1. `StratifiedKFold` на train;
2. обучение модели на каждом фолде;
3. сбор CV-метрик;
4. финальное обучение на всём train;
5. предсказание на test и запись submission.

### 3. Ансамбли
Поддерживаются:
- `average_proba_ensemble` — усреднение вероятностей нескольких моделей;
- `voting_ensemble` — voting-ансамбль.

---

## Структура проекта

```text
Titanic/
├── checkpoints/
│   ├── Classic_models/        # сохранённые classic ML модели
│   ├── DNN_models/            # сохранённые DNN артефакты
│   ├── submission/            # submission-файлы
│   └── leaderboard.csv        # лог результатов экспериментов
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── EDA_RESULTS.ipynb
│   ├── result.ipynb
│   ├── project_all_in_one.ipynb
│   ├── df_report.html
│   └── df_fe_report.html
├── src/
│   ├── classic_runner.py      # обучение classic ML и submission
│   ├── data.py                # чтение данных и DataLoader'ы
│   ├── dnn_runner.py          # CV + final fit + inference для DNN
│   ├── ensemble.py            # averaging / voting ансамбли
│   ├── features.py            # FeatureEngineer
│   ├── metrics.py             # метрики
│   ├── model.py               # архитектура DNN
│   ├── pipeline.py            # preprocessing pipeline
│   ├── schema.py              # схемы признаков
│   ├── search_spaces.py       # search spaces для classic ML
│   └── utils.py               # сохранение, загрузка, seed, dirs, логирование
├── cheklist.md
├── configs.py
├── main.py
├── README.md
└── requirements.txt
```

---

## Feature Engineering

Feature engineering реализован через кастомный sklearn-трансформер `FeatureEngineer`.

В проекте используются признаки, основанные на исходных полях Titanic, например:
- `Family`
- `Is_alone`
- `Big_family`
- `Title`
- `Fare_per_person`
- `Age_class`
- `Fare_class`
- `Name_length`
- `Fare_log`
- и другие признаки, создаваемые внутри пайплайна

Это позволяет использовать одну и ту же логику преобразований как в classic ML, так и в DNN-ветке.

---

## Конфиг

Все основные настройки находятся в `configs.py`.

На что смотреть в первую очередь:
- `config.general.experiment_name` — имя эксперимента;
- `config.general.seed` — seed;
- `config.training.model_name` — какую модель запускать;
- `config.training.scoring` — метрика для classic ML;
- `config.training.search_n_iter` — число итераций поиска;
- `config.training.batch_size`, `num_epochs`, `lr`, `weight_decay` — параметры DNN;
- `config.paths.*` — пути до данных, чекпоинтов, submission и leaderboard.

Примеры `model_name`:
- `DNN`
- `RFC`
- `LogR`
- `LogR_l1`
- `LogR_elasticnet`
- `KNN`
- `SVC`
- `CatBC`
- `GradBC`
- `LGBMClf`
- `Ensemble_AVG`
- `Ensemble_voting`
- `All`

---

## Установка

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Как запускать

### 1. Обучить одну classic-модель
В `configs.py`:
```python
config.training.model_name = 'LogR'
```

Запуск:
```bash
python3 main.py
```

### 2. Обучить DNN
В `configs.py`:
```python
config.training.model_name = 'DNN'
```

Запуск:
```bash
python3 main.py
```

### 3. Обучить все classic-модели подряд
В `configs.py`:
```python
config.training.model_name = 'All'
```

### 4. Построить ансамбль усреднением
В `configs.py`:
```python
config.training.model_name = 'Ensemble_AVG'
```

### 5. Построить voting-ансамбль
В `configs.py`:
```python
config.training.model_name = 'Ensemble_voting'
```

---

## Что сохраняется

### Classic ML
Сохраняются:
- обученная sklearn-модель;
- submission-файл;
- запись в `leaderboard.csv`.

### DNN
Сохраняются:
- `model_state_dict`;
- `FeatureEngineer`;
- `preprocessor`;
- metadata модели (`input_size`, `output_size`, `dropout` и др.);
- submission-файл.

---

## Логирование результатов

Проект ведёт таблицу результатов в:
- `checkpoints/leaderboard.csv`

Туда пишутся, как минимум:
- дата и время запуска;
- имя эксперимента;
- имя модели;
- итоговый score;
- параметры запуска.

Дополнительно проект может логировать эксперименты через `wandb`.

---

## Ноутбуки

В `notebooks/` лежат:
- `EDA_RESULTS.ipynb` — разведочный анализ;
- `project_all_in_one.ipynb` — ранняя all-in-one версия проекта;
- `result.ipynb` — ноутбук с результатами экспериментов;
- `df_report.html` и `df_fe_report.html` — отчёты по данным и признакам.

---

## Ближайшие идеи для развития

- добавить stacking;
- аккуратно проверить DNN с embedding для категориальных признаков;
- сделать честную OOF-оценку для ансамблей;
- усилить сравнение моделей через единый leaderboard;
- добавить базовые тесты на `FeatureEngineer`, preprocessing и inference.


