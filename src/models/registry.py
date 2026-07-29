"""Maps the class names used in config/models.yaml to sklearn classes.

Keeping this indirection means models.yaml can name a class as a plain
string (``class: LinearSVC``) instead of every config file needing a Python
import path.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

CLASSICAL_MODEL_CLASSES = {
    "MultinomialNB": MultinomialNB,
    "LinearSVC": LinearSVC,
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
}