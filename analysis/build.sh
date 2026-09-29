#!/usr/bin/env bash
# Compute every table, check and draft figure of the paper from ../results/ (CPU only); outputs go to out/.
# usage: ./build.sh [python]      (default: python3, with requirements-standard.txt installed)
set -euo pipefail
cd -- "$(dirname -- "$0")"
PY=${1:-python3}
mkdir -p out/logs
export MPLCONFIGDIR="$PWD/out/.matplotlib"
for s in prereg_score routing_window std_analyze std_musique_paired loop_musique_paired std_facts_paired fig_teaser std_figures fig_default_loop fig_reach_loop fig_band fig_mechanism_loop fig_downstream_loop fig_window fig_toy tab_panel tab_reach ext_dsv4 tab_masks_ci tab_interventions; do
  echo "== $s"
  "$PY" "$s.py" > "out/logs/$s.log" 2>&1 || { echo "failed: $s (see out/logs/$s.log)"; exit 1; }
done
echo "done: tables, checks and figures in out/"
