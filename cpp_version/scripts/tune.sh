#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA="data/data.csv"
TRAIN_CSV="data/data_training.csv"
VALID_CSV="data/data_validation.csv"
RESULTS_CSV="models/tuning_results.csv"

mkdir -p models

SEEDS=(42 7 123 2024 999)

CONFIGS=(
  "--layers 30 24 24 2 --activation sigmoid --lr 0.1 --batch_size 8 --epochs 50"
  "--layers 30 24 24 2 --activation sigmoid --lr 0.05 --batch_size 8 --epochs 80"
  "--layers 30 24 24 2 --activation sigmoid --lr 0.02 --batch_size 16 --epochs 120"
  "--layers 30 32 16 2 --activation sigmoid --lr 0.05 --batch_size 8 --epochs 80"
  "--layers 30 24 16 8 2 --activation sigmoid --lr 0.05 --batch_size 8 --epochs 100"
  "--layers 30 16 8 2 --activation tanh --lr 0.01 --batch_size 16 --epochs 80"
  "--layers 30 24 12 2 --activation relu --lr 0.01 --batch_size 16 --epochs 80"
)

echo "seed,config,bce,accuracy,precision,recall,f1" > "$RESULTS_CSV"

for seed in "${SEEDS[@]}"; do
  ./split "$DATA" "$TRAIN_CSV" "$VALID_CSV" "$seed" 0.2 > /tmp/mlp_split.log
  for cfg in "${CONFIGS[@]}"; do
    ./train "$TRAIN_CSV" "$VALID_CSV" $cfg > /tmp/mlp_train.log
    ./predict models/model.txt models/scaler.txt "$VALID_CSV" > /tmp/mlp_predict.log

    bce="$(awk -F': ' '/Binary cross-entropy/ {print $2}' /tmp/mlp_predict.log)"
    acc="$(awk -F': ' '/Accuracy/ {print $2}' /tmp/mlp_predict.log)"
    prec="$(awk -F': ' '/Precision/ {print $2}' /tmp/mlp_predict.log)"
    rec="$(awk -F': ' '/Recall/ {print $2}' /tmp/mlp_predict.log)"
    f1="$(awk -F': ' '/F1/ {print $2}' /tmp/mlp_predict.log)"

    printf '%s,"%s",%s,%s,%s,%s,%s\n' \
      "$seed" "$cfg" "$bce" "$acc" "$prec" "$rec" "$f1" >> "$RESULTS_CSV"
  done
done

echo
echo "Saved: $RESULTS_CSV"
echo "Top-10 by F1:"
{
  head -n 1 "$RESULTS_CSV"
  tail -n +2 "$RESULTS_CSV" | sort -t',' -k7,7gr | head -n 10
} | column -s, -t
