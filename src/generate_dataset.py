"""Generator dataset untuk repo ini.

Data sintetis dengan seed tetap (reproducible), tidak memuat data pribadi.
Jalankan:  python src/generate_dataset.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common import seed_default

SEED = seed_default()


def gen_iot_sensors(nodes: int = 12, days: int = 90, seed: int = SEED) -> pd.DataFrame:
    """Deret waktu sensor kebun (suhu, kelembapan, soil moisture) + anomali.

    Anomali disintesis sebagai 3 tipe supaya analisis EDA bisa membedakannya:
      - spike suhu (sensor rusak / heat stress)
      - pembacaan basah terus-menerus (sensor moisture terjebak)
      - jendela hilang (node mati, baris NaN)
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=days * 24 * 12, freq="5min")
    hour = np.asarray(idx.hour + idx.minute / 60, dtype=float)
    frames = []

    for k in range(nodes):
        base_t = 27.0 + rng.normal(0, 1.2)
        temp = base_t + 4.0 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.4, len(idx))
        hum = 75 - 10 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 1.5, len(idx))
        soil = 600 + rng.normal(0, 15, len(idx)) - (np.arange(len(idx)) % 2400) * 0.08
        soil = np.clip(soil, 200, 1000)

        anomaly = np.zeros(len(idx), dtype=int)
        atype = np.array(["none"] * len(idx), dtype=object)

        # tipe 1: spike suhu (heat stress / sensor rusak)
        for s in rng.choice(len(idx) - 36, size=len(idx) // 1500, replace=False):
            temp[s : s + 12] += rng.uniform(7, 12)
            hum[s : s + 12] -= rng.uniform(15, 25)
            anomaly[s : s + 12] = 1
            atype[s : s + 12] = "spike_suhu"

        # tipe 2: soil moisture macet di nilai tinggi (sensor short)
        for s in rng.choice(len(idx) - 288, size=len(idx) // 6000, replace=False):
            soil[s : s + 288] = rng.uniform(940, 990)
            anomaly[s : s + 288] = 1
            atype[s : s + 288] = "moisture_macet"

        # tipe 3: node mati -> NaN (gateway putus)
        for s in rng.choice(len(idx) - 72, size=len(idx) // 9000, replace=False):
            temp[s : s + 72] = np.nan
            hum[s : s + 72] = np.nan
            soil[s : s + 72] = np.nan
            anomaly[s : s + 72] = 1
            atype[s : s + 72] = "node_mati"

        frames.append(
            pd.DataFrame(
                {
                    "ts": idx,
                    "node": f"node-{k + 1:02d}",
                    "temp_c": temp,
                    "humidity_pct": hum,
                    "soil_moisture": soil,
                    "anomaly": anomaly,
                    "anomaly_type": atype,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- materi 3

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=seed_default())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = gen_iot_sensors(seed=a.seed)
    path = os.path.join(a.out, "materi1_sensor_iot.csv")
    df.to_csv(path, index=False)
    print(f"{path}  {len(df)} baris, {len(df.columns)} kolom")


if __name__ == "__main__":
    main()
