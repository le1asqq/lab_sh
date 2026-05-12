# -*- coding: utf-8 -*-
"""
LinearSVC: сырые данные vs StandardScaler.
Столбчатая диаграмма метрик; гистограмма P(болен); ROC; матрицы ошибок.
Кластеризация: KMeans по 7 признакам на тесте; графики — PCA (ГК1, ГК2).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import KMeans

PATH_TRAIN = r"f:\Лабы Шишкин\4\disease_train.csv"
PATH_TEST = r"f:\Лабы Шишкин\4\disease_public_test.csv"
PATH_SUB = r"f:\Лабы Шишкин\4\disease_sample_submission.csv"

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]
TARGET = "Y"
RANDOM_STATE = 42
N_SPLITS = 5


def linear_svc():
    return LinearSVC(
        max_iter=8000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def print_classification_block(y_true, y_pred, scores, title_ru):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = [0, 1]
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    print(f"\n{title_ru}")
    print(
        f"{'Класс':<8} {'Precision':>10} {'Полнота':>10} {'F1-мера':>10} {'Объектов':>10}"
    )
    for i, lab in enumerate(labels):
        print(f"{lab:<8} {pr[i]:>10.4f} {rc[i]:>10.4f} {f1[i]:>10.4f} {int(sup[i]):>10d}")
    print(f"\nТочность: {accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_true, scores):.4f}")


train_df = pd.read_csv(PATH_TRAIN)
test_df = pd.read_csv(PATH_TEST)
sub_df = pd.read_csv(PATH_SUB, sep=";", decimal=",")
for d in (train_df, test_df, sub_df):
    d.columns = [c.strip() for c in d.columns]

X = train_df[FEATURES].values
y = train_df[TARGET].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

clf_raw = linear_svc()
clf_raw.fit(X_train, y_train)
pred_raw = clf_raw.predict(X_val)
scores_raw = clf_raw.decision_function(X_val)

pipe_std = Pipeline([("scaler", StandardScaler()), ("svc", linear_svc())])
pipe_std.fit(X_train, y_train)
pred_std = pipe_std.predict(X_val)
scores_std = pipe_std.decision_function(X_val)

print_classification_block(
    y_val,
    pred_raw,
    scores_raw,
    "LinearSVC, сырые данные (проверочная выборка 20%)",
)
print_classification_block(
    y_val,
    pred_std,
    scores_std,
    "LinearSVC со стандартизацией (та же проверочная выборка)",
)

metric_names = ["Точность", "Precision", "Полнота", "F1-мера"]
raw_vals = [
    accuracy_score(y_val, pred_raw),
    precision_score(y_val, pred_raw, zero_division=0),
    recall_score(y_val, pred_raw, zero_division=0),
    f1_score(y_val, pred_raw, zero_division=0),
]
std_vals = [
    accuracy_score(y_val, pred_std),
    precision_score(y_val, pred_std, zero_division=0),
    recall_score(y_val, pred_std, zero_division=0),
    f1_score(y_val, pred_std, zero_division=0),
]

x = np.arange(len(metric_names))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w / 2, raw_vals, w, label="Сырые данные")
ax.bar(x + w / 2, std_vals, w, label="Стандартизация")
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylabel("Значение метрики")
ax.set_ylim(0, 1.05)
ax.set_title("Сравнение метрик до и после стандартизации (проверка 20%)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def print_cv(title_ru, estimator, X_all, y_all):
    acc = cross_val_score(estimator, X_all, y_all, cv=skf, scoring="accuracy")
    print(f"\n{title_ru}")
    print(
        f"Точность при {N_SPLITS}-fold кросс-валидации: {acc.mean():.4f} +/- {acc.std():.4f}"
    )


print_cv(
    "Весь disease_train, сырые данные — LinearSVC",
    linear_svc(),
    X,
    y,
)
print_cv(
    "Весь disease_train, StandardScaler + LinearSVC",
    Pipeline([("scaler", StandardScaler()), ("svc", linear_svc())]),
    X,
    y,
)

fpr_r, tpr_r, _ = roc_curve(y_val, scores_raw)
fpr_s, tpr_s, _ = roc_curve(y_val, scores_std)
plt.figure(figsize=(6, 5))
plt.plot(fpr_r, tpr_r, label=f"Сырые данные (AUC = {auc(fpr_r, tpr_r):.3f})")
plt.plot(fpr_s, tpr_s, label=f"Стандартизация (AUC = {auc(fpr_s, tpr_s):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.35)
plt.xlabel("Доля ложных тревог")
plt.ylabel("Полнота (TPR)")
plt.title("ROC-кривая, LinearSVC")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix(y_val, pred_raw, labels=[0, 1])).plot(
    ax=axes[0], colorbar=False
)
axes[0].set_title("Матрица ошибок — сырые данные")
ConfusionMatrixDisplay(confusion_matrix(y_val, pred_std, labels=[0, 1])).plot(
    ax=axes[1], colorbar=False
)
axes[1].set_title("Матрица ошибок — стандартизация")
plt.tight_layout()
plt.show()

calib = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(linear_svc(), method="sigmoid", cv=3)),
    ]
)
calib.fit(X_train, y_train)
p_sick = calib.predict_proba(X_val)[:, 1]

plt.figure(figsize=(7, 4.5))
plt.hist(p_sick[y_val == 0], bins=20, alpha=0.6, label="Y=0 (здоров)", color="tab:blue")
plt.hist(p_sick[y_val == 1], bins=20, alpha=0.6, label="Y=1 (болен)", color="tab:red")
plt.xlabel("Вероятность класса «болен» после калибровки")
plt.ylabel("Количество объектов")
plt.title("Распределение вероятностей (уверенность модели)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Кластеризация: 7 признаков, KMeans в исходном 7D; рисунок — PCA для наглядности ---
X_test_full = test_df[FEATURES].values.astype(float)
y_test_true = sub_df[TARGET].values.astype(int)

scaler7 = StandardScaler()
X_test_s = scaler7.fit_transform(X_test_full)

km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
lab_km = km.fit_predict(X_test_s)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
Z = pca.fit_transform(X_test_s)

print("\nКластеризация KMeans (k=2), тест, все признаки X1–X7 после масштабирования")
print(f"Silhouette (в 7D): {silhouette_score(X_test_s, lab_km):.4f}")
print(f"ARI:             {adjusted_rand_score(y_test_true, lab_km):.4f}")
print(f"NMI:             {normalized_mutual_info_score(y_test_true, lab_km):.4f}")
print(
    "График: проекция на ГК1 и ГК2 (PCA от масштабированных 7 признаков); "
    "кластеры считались в 7D."
)

colors_km = ["tab:blue", "tab:orange"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for k in np.unique(lab_km):
    m = lab_km == k
    axes[0].scatter(
        Z[m, 0],
        Z[m, 1],
        s=45,
        alpha=0.85,
        c=colors_km[int(k) % 2],
        edgecolors="black",
        linewidths=0.35,
        label=f"Кластер {k}",
    )
axes[0].set_title("KMeans (k=2): предсказанные кластеры")
axes[0].set_xlabel("ГК1 (PCA)")
axes[0].set_ylabel("ГК2 (PCA)")
axes[0].grid(alpha=0.3)
axes[0].legend(title="Обозначения")

for yv, name, color in [(0, "Здоров (Y=0)", "tab:blue"), (1, "Болен (Y=1)", "tab:red")]:
    m = y_test_true == yv
    axes[1].scatter(
        Z[m, 0],
        Z[m, 1],
        s=45,
        alpha=0.85,
        c=color,
        edgecolors="black",
        linewidths=0.35,
        label=name,
    )
axes[1].set_title("Истинные метки Y")
axes[1].set_xlabel("ГК1 (PCA)")
axes[1].set_ylabel("ГК2 (PCA)")
axes[1].grid(alpha=0.3)
axes[1].legend(title="Обозначения")

plt.tight_layout()
plt.show()

