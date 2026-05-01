# Titanic Kaggle Project

Проект для соревнования **Titanic - Machine Learning from Disaster** на Kaggle.

Репозиторий объединяет:
- классические ML-модели через `Pipeline + ColumnTransformer + RandomizedSearchCV`;
- отдельную DNN-ветку с кросс-валидацией, финальным дообучением на всём train и предсказанием на test;
- простой ансамбль усреднением вероятностей;
- сохранение моделей, submission-файлов и таблицы результатов.

## Что умеет проект

### Classic ML
Поддерживается обучение и предсказание для моделей из `search_spaces.py`, которые выбираются через `config.training.model_name`.

Основная логика:
1. чтение данных;
2. feature engineering;
3. preprocessing через `ColumnTransformer`;
4. подбор гиперпараметров через `RandomizedSearchCV`;
5. сохранение лучшей модели;
6. предсказание на test и запись submission.

### DNN
Для нейросети реализован отдельный pipeline:
1. `StratifiedKFold` на train;
2. обучение модели на каждом фолде;
3. сбор средней CV-метрики;
4. финальное обучение на всём train;
5. предсказание на test и запись submission.

### Ensemble
Поддерживается ансамбль усреднением вероятностей нескольких уже обученных classic ML-моделей.

## Структура проекта

```text
Titanic/
├── checkpoints/              # сохранённые модели, submission, leaderboard
├── data/                     # train.csv и test.csv
├── notebooks/                # черновые эксперименты
├── src/
│   ├── classic_runner.py     # обучение classic ML
│   ├── dnn_runner.py         # CV + финальное обучение DNN + inference
│   ├── data.py               # чтение данных и DataLoader'ы
│   ├── ensemble.py           # усреднение вероятностей
│   ├── features.py           # feature engineering
│   ├── metrics.py            # метрики
│   ├── model.py              # архитектура DNN
│   ├── pipeline.py           # sklearn preprocessing pipeline
│   ├── search_spaces.py      # search space для classic ML
│   └── utils.py              # утилиты, seed, сохранение, логирование
├── cheklist.md
├── configs.py
├── main.py
└── requirements.txt
```

## Установка

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Конфиг

Все основные настройки находятся в `configs.py`.

На что смотреть в первую очередь:
- `config.general.experiment_name` — имя эксперимента;
- `config.general.seed` — seed;
- `config.training.model_name` — какую модель запускать;
- `config.training.training_all_models` — обучать все модели подряд или только одну;
- `config.paths.*` — пути до данных, чекпоинтов и submission.

### Примеры `model_name`
- `LogR`
- `LogR_l1`
- `LogR_elasticnet`
- `KNN`
- `RFC`
- `SVC`
- `CatBC`
- `GradBC`
- `LGBMClf`
- `DNN`
- `Ensemble_AVG`

## Как запускать

### 1. Обучить одну модель
В `configs.py`:
```python
config.training.model_name = 'LGBMClf'
config.training.training_all_models = False
```

Запуск:
```bash
python main.py
```

### 2. Обучить все classic-модели подряд
В `configs.py`:
```python
config.training.training_all_models = True
```

### 3. Запустить DNN
В `configs.py`:
```python
config.training.model_name = 'DNN'
```

### 4. Запустить ансамбль усреднением
В `configs.py`:
```python
config.training.model_name = 'Ensemble_AVG'
```

## Что сохраняется

Проект сохраняет артефакты в `checkpoints/`:
- classic ML-модели;
- DNN-артефакты;
- submission-файлы;
- таблицу результатов / leaderboard.

Типичный pipeline после запуска:
1. обучение модели;
2. сохранение лучшего артефакта;
3. предсказание на `test.csv`;
4. запись submission в папку `checkpoints/submission/`.

## Логирование результатов

Для сравнения экспериментов используется простая таблица результатов.

Рекомендуемые поля:
- `created_at`
- `experiment_name`
- `model_name`
- `score`
- `params`

Идея такая:
- для classic ML в `score` писать `best_score_` из `RandomizedSearchCV`;
- для DNN — среднюю CV-метрику;
- в `params` хранить строку со словарём гиперпараметров.

## Как устроен preprocessing

Feature engineering вынесен отдельно в `src/features.py`.

Дальше признаки проходят через sklearn pipeline:
- числовые — imputing + scaling;
- категориальные — encoding;
- затем данные идут в модель.

Для DNN используется отдельная ветка подготовки данных через `get_loaders(...)`.

## Текущий workflow

### Classic ML
`main.py` → `src/classic_runner.py` → обучение → сохранение модели → предсказание на test.

### DNN
`main.py` → `src/dnn_runner.py`:
- `run_NN()` — CV;
- `fit_final_dnn()` — финальное обучение на всём train;
- `predict_test_dnn()` — submission.

### Ensemble
`main.py` → `src/ensemble.py` → загрузка обученных моделей → усреднение вероятностей → submission.

## Что можно улучшать дальше

- добавить OOF-оценку для ансамблей;
- сделать stacking;
- добавить embeddings для категориальных признаков в DNN;
- улучшить логирование экспериментов;
- добавить более аккуратное README с примерами метрик и leaderboard.

## Быстрый старт

Если нужен самый быстрый запуск:
1. положить `train.csv` и `test.csv` в папку `data/`;
2. выбрать `model_name` в `configs.py`;
3. запустить `python main.py`;
4. взять submission из `checkpoints/submission/`.
