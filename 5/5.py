# -*- coding: utf-8 -*-
"""
Лаба: LassoLarsCV + масштабирование, кластеризация, графики.
Предсказания test — в консоль (без файла).
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PATH_TRAIN = r"f:\Лабы Шишкин\5\train.tsv"
PATH_TEST = r"f:\Лабы Шишкин\5\test.tsv"
PATH_RAW = r"f:\Лабы Шишкин\5\ml_moscow_flats.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def print_reference_table():
    try:
        with open(PATH_RAW, "rb") as f:
            sig = f.read(4)
    except OSError as e:
        print("Справочный файл не открыт:", e)
        return

    if len(sig) >= 2 and sig[:2] == b"PK":
        try:
            raw = pd.read_excel(PATH_RAW, engine="openpyxl", nrows=0)
            print("ml_moscow_flats (Excel), столбцы:", list(raw.columns))
        except Exception as e:
            print("ml_moscow_flats как Excel:", e, "(pip install openpyxl)")
        return

    for sep in (None, ";", ",", "\t"):
        try:
            with open(PATH_RAW, "r", encoding="cp1251", errors="replace", newline="") as f:
                if sep is None:
                    pd.read_csv(f, sep=None, engine="python", nrows=0)
                else:
                    pd.read_csv(f, sep=sep, nrows=0)
            print("ml_moscow_flats (CSV) прочитан (заголовок есть)")
            return
        except Exception:
            continue


def plot_fact_vs_pred(y_true, y_pred, title, ax, use_hexbin=True):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="идеал")

    if use_hexbin and len(y_true) > 300:
        hb = ax.hexbin(y_true, y_pred, gridsize=40, mincnt=1, cmap="Blues")
        plt.colorbar(hb, ax=ax, label="число точек в ячейке")
    else:
        ax.scatter(y_true, y_pred, s=10, alpha=0.2, edgecolors="none")

    ax.set_xlabel("Факт (y)")
    ax.set_ylabel("Предсказание (ŷ)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")


print_reference_table()

train = pd.read_csv(PATH_TRAIN, sep="\t", header=None)
test = pd.read_csv(PATH_TEST, sep="\t", header=None)

print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)

X_train_full = train.iloc[:, :-1].values.astype(float)
y_train_full = train.iloc[:, -1].values.astype(float)
n_feat = X_train_full.shape[1]
X_test = test.iloc[:, :n_feat].values.astype(float)

print("Признаков:", n_feat)
print("Целевая: min =", y_train_full.min(), "max =", y_train_full.max())
print("Пропуски X train:", np.isnan(X_train_full).sum())
print("Пропуски y train:", np.isnan(y_train_full).sum())
print("Пропуски X test:", np.isnan(X_test).sum())

# Насколько какой-то признак похож на y (утечка / почти копия цены)
corrs = []
for j in range(n_feat):
    c = np.corrcoef(X_train_full[:, j], y_train_full)[0, 1]
    corrs.append(abs(c) if not np.isnan(c) else 0.0)
corrs = np.array(corrs)
j_max = int(np.argmax(corrs))
print(f"max |corr(признак, y)| = {corrs.max():.6f}  (столбец признака f{j_max})")
if corrs.max() > 0.999:
    print("  -> очень высокая корреляция: возможна утечка цены в признаки или вырожденный датасет.")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)
print("\nРазмеры: train =", X_train.shape, "val =", X_val.shape)

lasso_raw = LassoLarsCV(cv=CV_FOLDS, max_iter=50000, n_jobs=-1)
lasso_raw.fit(X_train, y_train)
y_pred_raw = lasso_raw.predict(X_val)

print("\n" + "=" * 60)
print("СЫРЫЕ ДАННЫЕ — LassoLarsCV")
print("=" * 60)
print("MSE:", mean_squared_error(y_val, y_pred_raw))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_raw)))
print("MAE:", mean_absolute_error(y_val, y_pred_raw))
print("R2:", r2_score(y_val, y_pred_raw))
print("max |y - pred|:", np.max(np.abs(y_val - y_pred_raw)))
print("alpha_:", lasso_raw.alpha_)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LassoLarsCV(cv=CV_FOLDS, max_iter=50000, n_jobs=-1)),
    ]
)
pipe.fit(X_train, y_train)
y_pred_scaled = pipe.predict(X_val)

print("\n" + "=" * 60)
print("StandardScaler + LassoLarsCV")
print("=" * 60)
print("MSE:", mean_squared_error(y_val, y_pred_scaled))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_scaled)))
print("MAE:", mean_absolute_error(y_val, y_pred_scaled))
print("R2:", r2_score(y_val, y_pred_scaled))
print("max |y - pred|:", np.max(np.abs(y_val - y_pred_scaled)))
print("alpha_:", pipe.named_steps["model"].alpha_)

# Scaler только для KMeans/PCA (на сплите train), как в твоей логике
scaler_km = StandardScaler()
scaler_km.fit(X_train)

results = pd.DataFrame(
    {
        "Модель": ["LassoLarsCV raw", "LassoLarsCV scaled"],
        "RMSE": [
            np.sqrt(mean_squared_error(y_val, y_pred_raw)),
            np.sqrt(mean_squared_error(y_val, y_pred_scaled)),
        ],
        "MAE": [
            mean_absolute_error(y_val, y_pred_raw),
            mean_absolute_error(y_val, y_pred_scaled),
        ],
        "R2": [
            r2_score(y_val, y_pred_raw),
            r2_score(y_val, y_pred_scaled),
        ],
    }
)
print("\nСРАВНЕНИЕ (валидация)")
print(results.to_string(index=False))

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
r2_cv_raw = cross_val_score(
    LassoLarsCV(cv=CV_FOLDS, max_iter=50000, n_jobs=-1),
    X_train_full,
    y_train_full,
    cv=kf,
    scoring="r2",
)
r2_cv_pipe = cross_val_score(
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LassoLarsCV(cv=CV_FOLDS, max_iter=50000, n_jobs=-1)),
        ]
    ),
    X_train_full,
    y_train_full,
    cv=kf,
    scoring="r2",
)
print(f"\nCV {CV_FOLDS}-fold R² весь train, raw:    {r2_cv_raw.mean():.4f} ± {r2_cv_raw.std():.4f}")
print(f"CV {CV_FOLDS}-fold R² весь train, scaled: {r2_cv_pipe.mean():.4f} ± {r2_cv_pipe.std():.4f}")

print("\n" + "=" * 60)
print("KMeans (k=2), валидация")
print("=" * 60)

X_train_scaled = scaler_km.transform(X_train)
X_val_scaled = scaler_km.transform(X_val)

try:
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init="auto")
except TypeError:
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)

kmeans.fit(X_train_scaled)
clusters = kmeans.predict(X_val_scaled)
print("Silhouette:", f"{silhouette_score(X_val_scaled, clusters):.3f}")

pca = PCA(n_components=2)
X_val_pca = pca.fit_transform(X_val_scaled)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=clusters, cmap="viridis", alpha=0.6)
plt.title("KMeans, 2 кластера")
plt.xlabel("PC1")
plt.ylabel("PC2")

y_val_binned = (y_val > np.median(y_val)).astype(int)
plt.subplot(1, 2, 2)
plt.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=y_val_binned, cmap="coolwarm", alpha=0.6)
plt.title("Цена (бинарно по медиане)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

print("ARI:", f"{adjusted_rand_score(y_val_binned, clusters):.3f}")
print("AMI:", f"{adjusted_mutual_info_score(y_val_binned, clusters):.3f}")

# Факт vs предсказание: hexbin, чтобы не «склеивались» в одну линию из-за наложения точек
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_fact_vs_pred(y_val, y_pred_raw, "Сырые (LassoLarsCV)", axes[0], use_hexbin=True)
plot_fact_vs_pred(y_val, y_pred_scaled, "Scaled (Pipeline)", axes[1], use_hexbin=True)
plt.tight_layout()
plt.show()

# Остатки — если тут не все нули, модель не «идеальная», даже если scatter узкий
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_val - y_pred_scaled, bins=40, edgecolor="black")
axes[0].set_title("Остатки y - ŷ (scaled)")
axes[0].set_xlabel("остаток")
axes[0].set_ylabel("частота")
axes[1].scatter(y_pred_scaled, y_val - y_pred_scaled, s=10, alpha=0.25)
axes[1].axhline(0.0, color="r", ls="--", lw=1)
axes[1].set_xlabel("ŷ (scaled)")
axes[1].set_ylabel("y - ŷ")
axes[1].set_title("Остатки vs предсказание")
plt.tight_layout()
plt.show()

# Финальная модель: test масштабируется ТЕМ ЖЕ pipeline (scaler внутри pipe)
pipe.fit(X_train_full, y_train_full)
y_test_pred = pipe.predict(X_test)

print("\n" + "=" * 60)
print("ПРЕДСКАЗАНИЯ TEST")
print("=" * 60)
print("Количество:", len(y_test_pred))
print("min / max / mean:", y_test_pred.min(), y_test_pred.max(), y_test_pred.mean())
np.set_printoptions(suppress=True, linewidth=120)
print("\nВсе предсказания:")
print(y_test_pred)

print("\nГотово.")