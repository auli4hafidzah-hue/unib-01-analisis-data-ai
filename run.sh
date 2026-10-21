#!/usr/bin/env bash
# Pipeline repo ini: generate dataset -> analisis.
# Pakai:  bash run.sh
set -euo pipefail
PY="${PYTHON:-python3}"
echo "== generate dataset =="
"$PY" src/generate_dataset.py
echo "== analisis =="
"$PY" src/materi1_analisis_sensor_iot.py
echo
echo "Selesai. Hasil ada di outputs/"
