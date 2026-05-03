"""Схема признаков и базовые константы проекта."""

TARGET_COLUMN = "Survived"
ID_COLUMN = "PassengerId"

FORCE_CATEGORICAL = [
    "Sex",
    "Embarked",
    "Cabin",
    "Pclass",
    "Title",
    "AgeGroup",
]

FORCE_NUMERICAL = [
    "Age",
    "Age_was_missing",
    "Fare",
    "Fare_log",
    "Parch",
    "SibSp",
    "Family",
    "Fare_per_person",
    "Age_class",
    "Fare_class",
    "Name_length",
    "WomanOrChild",
    "Is_alone",
    "Big_family",
]

FORCE_ORDINAL: list[str] = []


def get_feature_groups():
    """Возвращает группы признаков после FeatureEngineer.
    """
    return FORCE_CATEGORICAL, FORCE_ORDINAL, FORCE_NUMERICAL