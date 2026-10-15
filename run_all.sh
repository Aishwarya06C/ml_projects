#!/usr/bin/env bash
# Runs every project, saves metrics.txt + results.png in each folder, builds RESULTS.md
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
echo "Using: $PY"
$PY -m pip install -q -r requirements.txt
HAS_TF=0; $PY -c "import tensorflow" 2>/dev/null && HAS_TF=1
[ $HAS_TF -eq 0 ] && echo ">> TensorFlow not installed: project 15 will be skipped"
failed=()
for d in [0-9][0-9]_*/; do
  d=${d%/}
  echo "=== $d"
  if [[ $d == 15_* ]] && [ $HAS_TF -eq 0 ]; then echo "skipped"; continue; fi
  (cd "$d" && $PY main.py 2>&1 | grep -v -i warning | tee metrics.txt) || true
  [ -f "$d/results.png" ] || failed+=("$d")
done
{
  echo "# Results summary"; echo
  for d in [0-9][0-9]_*/; do
    [ -f "$d/metrics.txt" ] || continue
    echo "## ${d%/}"; echo '```'; tail -n 25 "$d/metrics.txt"; echo '```'; echo
  done
} > RESULTS.md
echo; echo "Done. Projects that failed or were skipped: ${failed[*]:-none}"
