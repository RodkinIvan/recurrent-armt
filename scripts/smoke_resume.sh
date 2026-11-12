#!/bin/bash
# End-to-end smoke of train.py on one GPU: run to step 6 (checkpoints at 3 and 6), then resume to step 10.
# Exercises freeze/unfreeze, state dropout (CUDA RNG in the checkpoint), the atomic checkpoint swap, log truncation
# on resume, the memory diagnostics and wandb. Usage: bash scripts/smoke_resume.sh <base model dir> <out dir>
set -euo pipefail
BASE=$1; OUT=$2
rm -rf "$OUT"
common="--base_model $BASE --out_dir $OUT --batch_size 2 --grad_accum 1 --warmup_steps 2 --eval_every 3 --save_every 3
        --eval_windows 4 --eval_long_windows 2 --log_every 1 --freeze_backbone_steps 4 --rnn_state_dropout 0.5
        --eval_state_reset --num_proc 2 --wandb_project recurrent-armt-smoke"
python train.py $common --max_steps 6
echo "---- first run done; log steps: $(python -c "import json; print([json.loads(l)['step'] for l in open('$OUT/log.jsonl')])")"
ls "$OUT"
# simulate a crash: a log line written after the checkpoint, and a swap interrupted after "last -> last.old"
echo '{"step": 7, "after_the_checkpoint": true}' >> "$OUT/log.jsonl"
mv "$OUT/last" "$OUT/last.old"
python train.py $common --max_steps 10
echo "---- resumed run done; log steps: $(python -c "import json; print([json.loads(l)['step'] for l in open('$OUT/log.jsonl')])")"
ls "$OUT"
