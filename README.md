# Studi Kasus 1 — Make Sense of Data with Analysis and AI

Materi dari [Informatika UNIB Official](https://youtube.com/@informatikaunibofficial):
**"Make Sense of Data with Analysis and AI"** — Andria Arisal.

Studi kasus analisis deret waktu sensor IoT: dari EDA, rekayasa fitur rolling window,
deteksi anomali tak-terawasi, sampai evaluasi terhadap label ground-truth.

> Data **sintetis dengan seed tetap** (reproducible), tidak memuat data pribadi,
> dan tidak meniru isi video. Tujuannya mempelajari metode analisis data.

## Menjalankan

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash run.sh          # atau:
python src/generate_dataset.py
python src/materi1_analisis_sensor_iot.py
```

Hasil masuk ke `outputs/`.

## Dataset

`data/materi1_sensor_iot.csv` — 311.040 baris, 12 node, 90 hari, interval 5 menit.

Kolom: `ts`, `node`, `temp_c`, `humidity_pct`, `soil_moisture`, `anomaly`, `anomaly_type`.

Tiga tipe anomali disintesis supaya bisa diuji pemisahannya:

| Tipe | Mekanisme | Ciri |
|---|---|---|
| `spike_suhu` | lonjakan suhu +7–12 °C selama 12 baris | Terlihat jelas di rolling mean |
| `moisture_macet` | nilai tanah macet di 940–990 selama 24 jam | **Tidak** terlihat aneh secara statistik |
| `node_mati` | baris NaN selama 72 baris (6 jam) | Harus ditangani sebelum masuk model |

## Metode

1. **EDA** — statistik deskriptif, hitung missing value.
2. **Fitur** — rolling window per node: `mean6`, `std6`, `diff` untuk tiap variabel.
3. **Deteksi tak-terawasi** — `IsolationForest(n_estimators=300, contamination=0.03)`,
   dilatih hanya pada baris non-NaN.
4. **Evaluasi** — `classification_report` terhadap label sintesis, plus rasio
   terdeteksi per tipe anomali.

## Hasil terakhir

```
              precision    recall  f1-score   support
      normal      0.972     0.994     0.983    293550
     anomali      0.806     0.475     0.598     15762
    accuracy                          0.967    309312
```

Rasio terdeteksi per tipe:

| Tipe | Recall model |
|---|---|
| `spike_suhu` | 0,486 |
| `moisture_macet` | 0,473 |
| `node_mati` | — (baris NaN, tidak masuk model) |

## Temuan penting

Deteksi anomali **tak-terawasi saja tidak cukup**:

- `node_mati` tidak terdeteksi karena baris NaN dibuang sebelum pemodelan.
  Data hilang justru harus jadi sinyal tersendiri (mis. alert kalau node hening > 1 jam).
- `moisture_macet` baru terdeteksi ~47% karena nilai yang konstan tidak "ekstrem"
  secara statistik. Perlu aturan berbasis domain: nilai berubah < 1 unit selama N jam.

Ini alasan nyata kenapa pipeline produksi menggabungkan model tak-terawasi dengan
rule-based check, bukan mengandalkan salah satunya.

## Struktur

```
src/common.py                     # helper path & direktori
src/generate_dataset.py           # generator data sintetis
src/materi1_analisis_sensor_iot.py
data/    outputs/                 # dibuat saat runtime, tidak di-commit
```

## Lisensi

MIT — lihat [LICENSE](LICENSE).
