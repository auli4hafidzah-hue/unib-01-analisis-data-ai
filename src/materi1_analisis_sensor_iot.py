"""Materi 1 - Make Sense of Data with Analysis and AI.

Analisis data sensor IoT: EDA, deteksi anomali tak-terawasi (IsolationForest),
dan evaluasi terhadap label ground-truth yang disintesis.

Jalankan:
    python src/materi1_analisis_sensor_iot.py
Output:
    outputs/materi1_anomali_per_tipe.csv
    outputs/materi1_ringkasan.txt
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

DATA = os.path.join("data", "materi1_sensor_iot.csv")
OUT = "outputs"
FEATURES = ["temp_c", "humidity_pct", "soil_moisture"]


def rolling_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("ts").copy()
    for c in FEATURES:
        g[f"{c}_mean6"] = g[c].rolling(6, min_periods=1).mean()
        g[f"{c}_std6"] = g[c].rolling(6, min_periods=1).std().fillna(0)
        g[f"{c}_diff"] = g[c].diff().fillna(0)
    return g


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["ts"])

    # --- 1. EDA dasar -----------------------------------------------------
    lines = []
    lines.append("=== MATERI 1: ANALISIS DATA SENSOR IoT ===\n")
    lines.append(f"Baris: {len(df):,} | node: {df['node'].nunique()} | "
                 f"rentang: {df['ts'].min()} .. {df['ts'].max()}")
    lines.append("\n-- Statistik deskriptif --")
    lines.append(df[FEATURES].describe().round(3).to_string())
    lines.append("\n-- Missing value --")
    lines.append(df.isna().sum().to_string())

    # --- 2. Deteksi anomali tak-terawasi ---------------------------------
    # fitur rolling per node (dihitung eksplisit, bukan groupby.apply, supaya
    # kolom `node` tetap ikut di semua versi pandas)
    feat = pd.concat([rolling_features(g) for _, g in df.groupby("node")],
                     ignore_index=True)
    feat_cols = [c for c in feat.columns
                 if c.endswith(("_mean6", "_std6", "_diff"))]
    # baris NaN (node mati) tidak bisa dipakai IsolationForest -> tandai manual
    nan_mask = feat[FEATURES].isna().any(axis=1)
    clean = feat[~nan_mask]
    model = IsolationForest(n_estimators=300, contamination=0.03,
                            random_state=42, n_jobs=-1)
    clean = clean.copy()
    clean["pred_anomali"] = (model.fit(clean[feat_cols]).predict(clean[feat_cols]) == -1).astype(int)

    # --- 3. Evaluasi vs ground-truth -------------------------------------
    truth = clean["anomaly"].to_numpy()
    pred = clean["pred_anomali"].to_numpy()
    lines.append("\n-- Evaluasi deteksi anomali (vs label sintesis) --")
    lines.append(classification_report(truth, pred, digits=3,
                                       target_names=["normal", "anomali"]))

    # --- 4. Karakteristik per tipe anomali ------------------------------
    # durasi per kejadian = jumlah baris anomaly / jumlah episode unik (per node)
    anom = feat[feat["anomaly"] == 1].copy()
    anom["episode"] = (anom.groupby(["node", "anomaly_type"])["ts"]
                       .transform(lambda s: (s.diff().dt.total_seconds().fillna(0) > 300).cumsum()))
    ep = anom.groupby(["node", "anomaly_type", "episode"]).size().reset_index(name="baris")
    by_type = (ep.groupby("anomaly_type")
               .agg(kejadian=("baris", "size"),
                    durasi_rata2_menit=("baris", lambda s: 5 * s.mean()),
                    durasi_max_menit=("baris", lambda s: 5 * s.max()),
                    node_terdampak=("node", "nunique"))
               .round(2))
    lines.append("\n-- Anomali per tipe --")
    lines.append(by_type.to_string())

    # anomali yang terdeteksi model vs yang dilewatkan, per tipe
    deteksi = clean[clean["anomaly"] == 1].groupby("anomaly_type")["pred_anomali"].mean().round(3)
    lines.append("\n-- Rasio terdeteksi per tipe (recall model) --")
    lines.append(deteksi.to_string())

    by_type.to_csv(os.path.join(OUT, "materi1_anomali_per_tipe.csv"))
    report = "\n".join(lines)
    with open(os.path.join(OUT, "materi1_ringkasan.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
