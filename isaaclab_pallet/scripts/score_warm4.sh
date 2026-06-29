#!/usr/bin/env bash
# Batch PHYSICS scoring of run_warm4 checkpoints on competition seq 0 & 1.
# For each checkpoint: (1) generate algorithm_results (CPU, no Isaac),
# (2) score with the official simulator.py (Isaac physics) + evaluator,
# (3) print avg_score + per-sequence score/fill/oob/drop.
# Results kept under /tmp/w4score/ (baseline sim_results/ untouched).
#
# Run from YOUR terminal or foreground; each checkpoint boots Isaac (~20-30s).
cd "$(dirname "$0")/../.."   # -> project root

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab
set -u

RUN=isaaclab_pallet/runs/run_warm4
OUT=/tmp/w4score
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
: > "$SUMMARY"

# checkpoints to score (override by passing names as args, e.g. PCT-update-002500.pt)
CKPTS=("$@")
if [ "${#CKPTS[@]}" -eq 0 ]; then
  CKPTS=(PCT-update-001750.pt PCT-update-002000.pt PCT-update-002250.pt \
         PCT-update-002500.pt PCT-update-002750.pt PCT-update-003000.pt \
         PCT-update-003250.pt PCT-update-003500.pt)
fi

for ck in "${CKPTS[@]}"; do
  tag="${ck%.pt}"
  gendir="$OUT/$tag"; simdir="${gendir}_sim"
  mkdir -p "$gendir"
  echo "================ $ck ================"
  python3 isaaclab_pallet/scripts/eval_competition_generate.py \
    --checkpoint "$RUN/$ck" --out-dir "$gendir" \
    --sequences box_sequence_0 box_sequence_1 --device cpu 2>&1 | grep -E "^\[gen\]" || true

  python3 palletizing_simulator/simulator.py \
    --config palletizing_simulator/config/sim_config.yaml \
    --input-dir "$gendir" -o "$simdir" > "$simdir.log" 2>&1

  python3 - "$ck" "$simdir/result.json" "$SUMMARY" <<'PY'
import json, sys
ck, path, summ = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    r = json.load(open(path))
except Exception as e:
    print(f"  !! no result.json ({e})");
    open(summ,"a").write(f"{ck}\tERROR\n"); sys.exit()
avg = r["results"]["avg_score"]
cells=[ck, f"avg={avg}"]
line=[ck, str(avg)]
for f in r["files"]:
    s=f"{f['source'].replace('box_sequence_','seq').replace('.json','')}:score={f['total_score']},fill={f['stacking_rate_pct']},oob={f['out_of_bounds_count']},drop={f['drop_count']},maxH={f['max_height_m']}"
    cells.append(s); line.append(str(f['total_score']))
print("  " + " | ".join(cells))
open(summ,"a").write("\t".join(line)+"\n")
PY
done

echo; echo "################ SUMMARY (avg vs baseline cjspec_v2=92.30) ################"
sort -t$'\t' -k2 -nr "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
