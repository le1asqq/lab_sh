# -*- coding: utf-8 -*-
"""
Лабораторная 5 — регрессия LassoLarsCV + KMeans
Вариант: только LassoLarsCV. Графики на русском, вывод на экран.
"""

from pathlib import Path
from typing import Optional
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoLarsCV
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 4)

BASE = Path(r"F:\Лабы Шишкин\5\5")
TRAIN_TSV = BASE / "train.tsv"
TEST_TSV = BASE / "test.tsv"
OUT_DIR = BASE / "results_lasso"
OUT_DIR.mkdir(exist_ok=True)

MOSCOW_FILE = None


def find_moscow_file() -> Optional[Path]:
    if MOSCOW_FILE is not None:
        p = Path(MOSCOW_FILE)
        return p if p.exists() else None

    script_dir = Path(__file__).resolve().parent
    folders = [BASE, script_dir, BASE.parent, script_dir.parent]

    exact_names = [
        "ml_moscow_flats.csv",
        "ml_moscow_flats.xlsx",
        "ml_moscow_flats (1).csv",
        "ml_moscow_flats (1).xlsx",
        "ml_moscow_flats (1)",
    ]

    for folder in folders:
        if not folder.exists():
            continue
        for name in exact_names:
            p = folder / name
            if p.exists():
                return p
        for p in folder.glob("ml_moscow_flats*"):
            if p.is_file():
                return p

    return None


RANDOM_STATE = 42
VAL_SIZE = 0.2


def _split_one_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] != 1:
        return df

    lines = df.iloc[:, 0].dropna().astype(str).str.strip()
    lines = lines[lines != ""]
    if len(lines) == 0:
        return df

    header = lines.iloc[0]
    names = [x.strip() for x in header.split(",")]
    data_lines = lines.iloc[1:] if "price" in header.lower() else lines

    parts = data_lines.str.split(",", expand=True)
    ncol = min(parts.shape[1], len(names))
    parts = parts.iloc[:, :ncol]
    parts.columns = names[:ncol]

    for col in parts.columns:
        parts[col] = pd.to_numeric(parts[col], errors="coerce")

    parts = parts.dropna(how="all").reset_index(drop=True)
    print(f"  Разобрано: {parts.shape[1]} столбцов, {parts.shape[0]} строк")
    return parts


def load_moscow_table(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()[:4]

    if raw[:2] != b"PK":
        for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
            for sep in (";", ",", "\t"):
                try:
                    df = pd.read_csv(path, encoding=enc, sep=sep, engine="python", on_bad_lines="warn")
                    if df.shape[1] >= 3:
                        print(f"  Текстовый CSV: {enc}, разделитель={repr(sep)}")
                        return _split_one_column(df) if df.shape[1] == 1 else df
                except Exception:
                    continue

    print("  Формат: Excel (файл .xlsx или .csv с содержимым Excel)")
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError("Выполните: pip install openpyxl")

    df = pd.read_excel(path, engine="openpyxl")
    if df.shape[1] >= 3:
        print(f"  Excel: {df.shape[0]} строк, {df.shape[1]} столбцов")
        return df
    df = pd.read_excel(path, engine="openpyxl", header=None)
    return _split_one_column(df)


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics(m, prefix="  "):
    print(
        f"{prefix}MSE={m['MSE']:.6g}, RMSE={m['RMSE']:.6g}, "
        f"MAE={m['MAE']:.6g}, R²={m['R2']:.6f}"
    )


def train_lasso(X_train, X_val, y_train, y_val, use_scaler, use_imputer=False):
    steps = []
    if use_imputer:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", LassoLarsCV(cv=5)))
    pipe = Pipeline(steps)

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    y_pred = pipe.predict(X_val)
    m = regression_metrics(y_val, y_pred)
    name = "Со стандартизацией" if use_scaler else "Без стандартизации"
    return m, y_pred, name, fit_time


def _indeks_dlya_grafika(n, max_tochki):
    if n > max_tochki:
        return np.random.default_rng(RANDOM_STATE).choice(n, size=max_tochki, replace=False)
    return np.arange(n)


def _scatter_na_osi(ax, y_true, y_pred, zagolovok_osi, max_tochki=1500, ceny=False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    idx = _indeks_dlya_grafika(len(y_true), max_tochki)
    ax.scatter(y_true[idx], y_pred[idx], s=16, alpha=0.35, c="#4C72B0", edgecolors="none")
    lo = min(y_true.min(), y_pred.min(), y_true[idx].min(), y_pred[idx].min())
    hi = max(y_true.max(), y_pred.max(), y_true[idx].max(), y_pred[idx].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    ax.set_xlabel(zagolovok_osi["x"])
    ax.set_ylabel(zagolovok_osi["y"])
    if ceny:
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))


def _klasterizaciya_na_dvuh_osyah(ax_kmeans, ax_istina, X_scaled, y_continuous, podpis_istiny):
    y_bin = (y_continuous >= np.median(y_continuous)).astype(int)
    metki = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, metki)
    ari = adjusted_rand_score(y_bin, metki)
    ami = adjusted_mutual_info_score(y_bin, metki)

    xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

    ax_kmeans.scatter(xy[:, 0], xy[:, 1], c=metki, cmap="tab10", s=12, alpha=0.55)
    ax_kmeans.set_xlabel("PC1")
    ax_kmeans.set_ylabel("PC2")
    ax_kmeans.set_title("KMeans кластеры")

    ax_istina.scatter(xy[:, 0], xy[:, 1], c=y_bin, cmap="coolwarm", s=12, alpha=0.55)
    ax_istina.set_xlabel("PC1")
    ax_istina.set_ylabel("PC2")
    ax_istina.set_title(f"Истинные метки ({podpis_istiny})")

    return {"Silhouette": sil, "ARI": ari, "AMI": ami}


def klasterizaciya_metriki(X_scaled, y_continuous):
    y_bin = (y_continuous >= np.median(y_continuous)).astype(int)
    metki = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled)
    return {
        "Silhouette": silhouette_score(X_scaled, metki),
        "ARI": adjusted_rand_score(y_bin, metki),
        "AMI": adjusted_mutual_info_score(y_bin, metki),
    }


def plot_klasterizaciya_odno_okno(X_scaled, y_continuous):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metriki = _klasterizaciya_na_dvuh_osyah(
        axes[0], axes[1], X_scaled, y_continuous, "по медиане цены"
    )

    fig.suptitle(
        "Визуализация кластеров (PCA, первые две главные компоненты):",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "Рисунок 1 - проекция объектов в пространство первых двух главных компонент.",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("\n  Кластеризация (Москва), KMeans k=2:")
    print(f"    Silhouette={metriki['Silhouette']:.4f}")
    print(f"    ARI        = {metriki['ARI']:.4f}")
    print(f"    AMI        = {metriki['AMI']:.4f}")
    return metriki


def plot_sintetika_vs_moskva(syn, msk):
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    lab_syn = {"x": "Истинное Y", "y": "Предсказанное Y"}
    lab_msk = {"x": "Истинная цена, руб", "y": "Предсказанная цена, руб"}
    _scatter_na_osi(axes1[0], syn["y_val"], syn["pred"], lab_syn, max_tochki=800, ceny=False)
    axes1[0].set_title("Синтетика (LassoLarsCV, со стандартизацией)")
    _scatter_na_osi(axes1[1], msk["y_val"], msk["pred"], lab_msk, max_tochki=1500, ceny=True)
    axes1[1].set_title("Москва (LassoLarsCV, со стандартизацией)")
    fig1.suptitle(
        "Сравнение датасетов: диаграмма рассеяния истинных и предсказанных значений",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    o_syn = syn["y_val"] - syn["pred"]
    o_msk = msk["y_val"] - msk["pred"]
    axes2[0].hist(o_syn, bins=50, edgecolor="black", alpha=0.75, color="#4C72B0")
    axes2[0].set_title("Синтетика — остатки")
    axes2[0].set_xlabel("Остаток")
    axes2[0].set_ylabel("Количество")
    axes2[1].hist(o_msk, bins=50, edgecolor="black", alpha=0.75, color="#55A868")
    axes2[1].set_title("Москва — остатки")
    axes2[1].set_xlabel("Остаток")
    axes2[1].set_ylabel("Количество")
    fig2.suptitle("Сравнение датасетов: гистограммы остатков LassoLarsCV", fontsize=12)
    plt.tight_layout()
    plt.show()

    cl_syn = klasterizaciya_metriki(syn["X_val_sc"], syn["y_val"])
    cl_msk = plot_klasterizaciya_odno_okno(msk["X_val_sc"], msk["y_val"])
    print("\n  Кластеризация (синтетика), только метрики в CSV:")
    print(f"    Silhouette={cl_syn['Silhouette']:.4f}, ARI={cl_syn['ARI']:.4f}, AMI={cl_syn['AMI']:.4f}")

    pd.DataFrame([{"dataset": "synthetic", **cl_syn}, {"dataset": "moscow", **cl_msk}]).to_csv(
        OUT_DIR / "clustering_sravnenie.csv", index=False, encoding="utf-8-sig"
    )


def run_synthetic():
    print("\n" + "=" * 60)
    print("1. СИНТЕТИЧЕСКИЙ НАБОР (train.tsv)")
    print("=" * 60)

    if not TRAIN_TSV.exists():
        print(f"Файл не найден: {TRAIN_TSV}")
        return None

    df = pd.read_csv(TRAIN_TSV, sep="\t", header=None)
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.float64)

    print(f"Объектов: {len(y)}, признаков: {X.shape[1]}")
    print(f"Y: от {y.min():.4f} до {y.max():.4f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    print(f"Обучение: {len(y_train)}, проверка (20%): {len(y_val)}")

    rows = []
    pred_scaled = None
    for scaled in (False, True):
        m, pred, name, t = train_lasso(X_train, X_val, y_train, y_val, use_scaler=scaled)
        print(f"\nLassoLarsCV — {name}, время: {t:.4f} с")
        print_metrics(m)
        rows.append({"вариант": name, "время_с": t, **m})
        if scaled:
            pred_scaled = pred

    pd.DataFrame(rows).to_csv(OUT_DIR / "synthetic_regression.csv", index=False, encoding="utf-8-sig")

    if TEST_TSV.exists():
        X_test = pd.read_csv(TEST_TSV, sep="\t", header=None).values.astype(np.float64)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LassoLarsCV(cv=5))])
        pipe.fit(X_train, y_train)
        pd.DataFrame({"предсказание": pipe.predict(X_test)}).to_csv(
            OUT_DIR / "synthetic_test_predictions.csv", index=False, encoding="utf-8-sig"
        )
        print("\nПредсказания test.tsv сохранены.")

    X_val_sc = StandardScaler().fit(X_train).transform(X_val)
    return {"y_val": y_val, "pred": pred_scaled, "X_val_sc": X_val_sc}


def load_moscow(path: Path):
    print(f"Чтение: {path}")
    df = load_moscow_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    print("Столбцы:", list(df.columns))

    target = next(
        (c for c in df.columns if "price" in c.lower() or "цена" in c.lower()),
        None,
    )
    if target is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("Не найден столбец price/цена.")
        target = num_cols[-1]
    print(f"Целевой столбец (цена): {target}")

    y = pd.to_numeric(df[target], errors="coerce").values.astype(np.float64)
    mask = ~np.isnan(y) & (y > 0)
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]

    if len(y) < 100:
        raise ValueError(f"Слишком мало строк ({len(y)}).")

    X_df = df.drop(columns=[target])
    cat_cols = X_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols)
        )
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))

    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    X = prep.fit_transform(X_df).astype(np.float64)
    print(f"Объектов: {len(y)}, признаков после кодирования: {X.shape[1]}")
    return X, y


def run_moscow():
    print("\n" + "=" * 60)
    print("2. РЕАЛЬНЫЙ НАБОР (квартиры Москвы)")
    print("=" * 60)

    path = find_moscow_file()
    if path is None:
        print("Файл ml_moscow_flats не найден.")
        print(f"Положите в: {BASE}")
        return None

    print(f"Найден файл: {path}")

    X, y = load_moscow(path)
    print(f"Цена, руб: от {y.min():.0f} до {y.max():.0f}, медиана {np.median(y):.0f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    n_nan = int(np.isnan(X_train).sum() + np.isnan(X_val).sum())
    if n_nan > 0:
        print(f"  Пропуски в признаках (NaN): {n_nan} — заполнение медианой по обучению")
        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(X_train)
        X_val = imp.transform(X_val)

    print(f"Обучение: {len(y_train)}, проверка: {len(y_val)}")

    rows = []
    pred_scaled = None
    for scaled in (False, True):
        m, pred, name, t = train_lasso(
            X_train, X_val, y_train, y_val, use_scaler=scaled, use_imputer=False
        )
        print(f"\nLassoLarsCV — {name}, время: {t:.4f} с")
        print_metrics(m)
        rows.append({"вариант": name, "время_с": t, **m})
        if scaled:
            pred_scaled = pred

    pd.DataFrame(rows).to_csv(OUT_DIR / "moscow_regression.csv", index=False, encoding="utf-8-sig")

    X_val_sc = StandardScaler().fit(X_train).transform(X_val)
    return {"y_val": y_val, "pred": pred_scaled, "X_val_sc": X_val_sc}


if __name__ == "__main__":
    print("Графики: 1) рассеяние  2) остатки  3) кластеризация (Москва)\n")
    print("BASE =", BASE.resolve())

    syn = run_synthetic()
    msk = run_moscow()

    if syn is not None and msk is not None:
        plot_sintetika_vs_moskva(syn, msk)
    else:
        print("\nГрафики не построены: нужны оба датасета.")

    print("\nГотово. Таблицы CSV:", OUT_DIR.resolve())
