#!/bin/bash
# Shard sparge_decode_eval.py over every idle GPU (under 1 GB used), one process each.
# Usage: scripts/launch_gpu_sweep.sh <tag> [extra sparge_decode_eval.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
tag=$1; shift
mkdir -p results/"$tag"
# GPUS="1,2,3" pins the set; default is every GPU under 1 GB used.
if [ -n "${GPUS:-}" ]; then IFS=, read -ra gpus <<< "$GPUS"; else
gpus=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1}')); fi
n=${#gpus[@]}
[ "$n" -gt 0 ] || { echo "no idle GPU"; exit 1; }
echo "tag=$tag on GPUs ${gpus[*]}"
for i in "${!gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpus[$i]} nohup .venv/bin/python scripts/sparge_decode_eval.py \
    --device cuda --shard "$i" --num-shards "$n" --out results/"$tag"/shard"$i".json "$@" \
    > results/"$tag"/shard"$i".log 2>&1 &
done
wait
.venv/bin/python - "$tag" <<'EOF'
import glob, json, sys
merged = {}
for p in sorted(glob.glob(f"results/{sys.argv[1]}/shard*.json")):
    d = json.load(open(p))
    merged.setdefault("args", d["args"])
    merged.setdefault("results", {}).update(d["results"])
json.dump(merged, open(f"results/{sys.argv[1]}.json", "w"), indent=2)
print("MERGED", len(merged["results"]), "configs")
EOF
