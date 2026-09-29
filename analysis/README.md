# Analysis

These scripts compute the paper's tables, checks and figures from the experiments' outputs in `../results/`, without a GPU.
`./build.sh [python]` runs them in order and writes everything to `out/`:
- `out/tables/`: LaTeX tables and macros;
- `out/checks/`: machine-readable summaries;
- `out/figs/`: draft versions of the figures;
- `out/logs/`: each script's printed output, which includes the numbers quoted in the text.

| Scripts | What they cover |
|---|---|
| `std_analyze.py`, `tab_panel.py` | The 13-model panel, placement sweeps, the last useful layer and the prospective test |
| `std_figures.py`, `fig_teaser.py`, `fig_band.py` | Standard-model figures, the first-page figure and the relay band |
| `std_musique_paired.py`, `std_facts_paired.py` | MuSiQue and fact chains, with paired bootstrap intervals |
| `tab_interventions.py` | Other edits at the same layer (projection LoRA, FLAS-style flows) |
| `fig_default_loop.py`, `fig_reach_loop.py`, `fig_mechanism_loop.py`, `fig_downstream_loop.py`, `tab_reach.py`, `tab_masks_ci.py`, `loop_musique_paired.py` | Looped models (Ouro, Huginn) |
| `prereg_score.py`, `routing_window.py`, `fig_window.py` | The routing window of the frozen models |
| `fig_toy.py` | Toy models trained from scratch |
| `ext_dsv4.py` | DeepSeek-V4-Flash |

A script stops with a missing-file error if an experiment it reads has not been run. `REPRODUCE.md` lists which runs each result
needs. The MuSiQue scripts also need the exported selection in `../data/` (see `../data/README.md`).
