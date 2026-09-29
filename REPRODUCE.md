# Reproducing the paper

This guide maps each result of the paper to the script that produces it and gives the commands we ran. The experiment scripts in
`scripts/` write their outputs to `results/`: JSON result files (each records the arguments of its run under `args`) and the
trained LoRA weights (`*.pt`). The analysis scripts in `analysis/` compute the paper's tables, checks and figures from
`results/` on a CPU.

## Setup

    python -m venv .venv-std  && .venv-std/bin/pip install -r requirements-standard.txt     # Python 3.12
    python -m venv .venv-loop && .venv-loop/bin/pip install -r requirements-looped.txt      # Python 3.11, for Ouro and Huginn
    export HF_HOME=/path/to/a/large/cache
    cd scripts && ../.venv-std/bin/python export_qa_data.py                                  # MuSiQue selection -> ../data/

- All commands run from `scripts/` and write to `../results/`.
- `$PY` is the interpreter of the standard-model environment (`../.venv-std/bin/python`). `$LPY` is the interpreter of the
  looped-model environment (`../.venv-loop/bin/python`); it runs the Ouro and Huginn scripts (`e6`, `e8`, `e39`, `e71`-`e88`).
- **MuSiQue runs need `PYTHONHASHSEED=0`.** The prompt builder shuffles each question's paragraphs with a generator seeded by
  Python's string hash. With the hash seed fixed, every run sees the same prompts, and results can be paired by question.
- Accuracies in the JSON files are fractions. Reach is computed from them by the analysis scripts.
- `jobq.sh GPU QUEUEFILE` runs the lines of a queue file one after another on one GPU. `waitmem.sh GPU MIB` waits until a GPU
  has enough free memory.

## Names in the code

The code uses older names than the paper:

| Paper | Code |
|---|---|
| The rank-8 LoRA at one layer (h ← s h + B A h at the input of layer *a*) | a *map*: `--kind map`, `--map FILE`; files `e19_adapter_*.pt`, `e52_adapter_*.pt`, `e71_map_*.pt`, `e80_map_*.pt`, `e81_map_*.pt` |
| The layer whose input the LoRA edits | `--a` (in `e19_reentry.py`: `--band A:A --no_loop`) |
| Projection LoRA | `--kind lora` (with `--lora_layers FIRST:LAST` for a range of layers) |
| FLAS-style low-rank flow; flow block | `--kind flow`; `--kind flowblock` (E91) or `--kind flowmlp` (E52) |
| Steering vector; soft prompt | `--kind steer` or `--steer`; `--kind prompt` |
| Frozen model | `frozen`; `--kind none` |
| Variable names | `--var_pool upper` (capital letters), `upperlower` (capital and lower-case letters), `both` (letters and two-letter names) |

## Where each result comes from

### Section 3: the default computation stops early (Appendix B)

| Result | Scripts | Outputs in `results/` |
|---|---|---|
| Reach of the 13 base models, commit depth and value-copy depth (panel table) | `e10_panel.py`, `commit_depth.py`; `analysis/tab_panel.py` | `e10_panel_<model>.json` |
| DeepSeek-V4-Flash: reach and the layer at which the root becomes the top candidate | a separate one-GPU inference runner for the 292B model (not part of this release); `analysis/ext_dsv4.py` reads its per-program output | `external/dsv4/vb_dsv4_items.jsonl` |
| The root value is copied late, for one- to four-line chains | `e4_trace.py` | `e4_trace_q8base_val_d{1,2,3,4}.json` |
| By default the program passes chain identity on for only a few lines (13 models) | `e32b_wave2.py` without a LoRA | `e32b_wave_relay_<model>.json` |
| The query's pointer walk | `e21_adapted_trace.py` (field `off`) | `e21_adapted_q8_noloop_d5.json` |
| Blocking the query after the routing window | `e76_window_block.py` | `e76_block_{q8,o3,l31}.json` |
| Routing-window width across models | `analysis/prereg_score.py`, then `analysis/routing_window.py` | `analysis/out/notes/routing_window_table.json` |
| Ouro and Huginn by loops; fact chains by hop; few-shot prompts | `e6_ouro_vb.py`, `e74_ouro_hops.py`, `e81_huginn_map.py` (frozen rows), `e88_fewshot.py` | `e6_ouro_*.json`, `e74_hops_*.json`, `e81_huginn_hug_r8.json`, `e88_fewshot_*.json` |
| Looped models: the query reads in the late loops, and the program does not relay | `e8_ouro_trace.py`, `e78_ouro_block.py`, `e39_ouro_relay.py`, `e83_huginn_relay.py`, `e82_ouro_strides.py`, `e84_huginn_strides.py` | `e8_ouro_*.json`, `e78_ouro_block_*.json`, `e39_ouro_relay_*.json`, `e83_*`, `e82_*`, `e84_*` |

### Section 4: a small LoRA makes the layers count (Appendix C)

| Result | Scripts | Outputs in `results/` |
|---|---|---|
| Qwen3-8B, 24-line chains: 15.5% → 99%; order and chain-count controls | `e19_reentry.py` (trains the LoRA), `e33_order_control.py` | `e19_reentry_vdeep_q8_r8_kl1.json`, `e19_adapter_vdeep_q8_r8_kl1.pt`, `e33_order_r8kl_matrix.json` |
| Three training seeds; WikiText-103 loss | `e19_reentry.py --seed`, `e19b_textloss.py` | `e19_reentry_vdeep_q8_r8_kl1_s{1,2}.json` |
| A longer-trained LoRA reaches 50 lines | `e19_reentry.py` | `e19_reentry_long_q8_r8_kl1.json` |
| Transfer to JavaScript and English | `e20_adapter_transfer.py` | `e20_transfer_q8_kl1_fixed.json` |
| The LoRA in the other panel models | `e19_reentry.py` (placement runs) | `e19_reentry_place_<model>_a<layer>.json` |
| Ouro: reach by loops, 160 lines, seeds, steering vector, three and four chains, Ouro-2.6B, first loop only, loops used in training | `e71_ouro_map.py`, `e71b_ouro_scaling.py`, `e71c_ouro_controls.py` | `e71_ouro_*.json`, `e71b_scaling_*.json`, `e71c_controls_o14_a6.json` |
| Huginn: reach by recurrences, longer training, perplexity | `e81_huginn_map.py`, `e81b_huginn_long.py`, `e81c_huginn_ppl.py`, `e86_huginn_stride_block.py` | `e81_huginn_*.json`, `e81c_*.json`, `e86_*.json` |
| Other edits at the same layer: projection LoRA, FLAS-style flow and flow block, the LoRA at three layers; the same edits after the placement limit | `e91_intervention_compare.py`; `analysis/tab_interventions.py` | `e91_compare_*.json` |
| Looping Qwen3-8B's middle layers | `e19_reentry.py --k_train_set 1,2 --kmax 2` | `e19_reentry_loop_q8_r8_kl1_long.json` |

### Section 5: the LoRA starts a relay in the middle layers (Appendices D and E)

| Result | Scripts | Outputs in `results/` |
|---|---|---|
| The relay front per layer (Qwen3-8B, Llama-3.1-8B) | `e32b_wave2.py`; `analysis/fig_band.py` prints the numbers | `e32b_wave_q8r8kl_d16*.json`, `e32b_wave_l31r8_a11_d16.json` |
| Restoring the query | `e34_deep_trace.py` | `e34_deep_trace_q8r8kl_d16.json` |
| Attention reads further up the chain as the relay advances; first two-line reads in five models | `e37b_strides.py` | `e37b_strides_*.json` |
| Head profiles and head ablations (Qwen3-8B) | `pick_relay_heads.py`, `e68_head_profile.py`, `e66_head_ablation.py` | `e68_heads_q8r8kl_inter.json`, `e66_ablate_q8r8kl_inter_after.json` |
| Cutting parent-line attention | `e38_relay_cut.py` | `e38_relaycut_q8r8kl_relaylayers{,_zero}.json` |
| Ancestor names in the representations | `e62_ancestor_code.py`, `e87_ouro_ancestor_probe.py`, `e87_split_prefix.py` | `e62_anc_q8r8kl_d16.json`, `e87_ancestor_*.json` |
| Ouro: the relay band, read distances, masks and head ablations | `e77_ouro_relay_map.py`, `e82_ouro_strides.py`, `e82_head_profiles.py`, `e85_ouro_stride_block.py`; `analysis/tab_masks_ci.py` | `e77_ouro_relay_*.json`, `e82_strides_*.json`, `e85_stride_block_*.json` |
| A LoRA on program tokens only; removing the LoRA's subspace | `e44_position_maps.py`, `e65_map_subspace.py` | `e44_posmap_q8_*_v2.json`, `e65_subspace_q8r8kl.json` |

### Section 6: where to put the fix (Appendix F)

| Result | Scripts | Outputs in `results/` |
|---|---|---|
| Placement sweeps in nine models, the last useful layer, the prospective test on held-out models | `e19_reentry.py`; `analysis/std_analyze.py` | `e19_reentry_place_*_a*.json` |
| Past the limit the relay does not start (Qwen3-8B, layer 20 vs 21) | `e37b_strides.py` | `e37b_strides_q8a{20,21}_d16_inter.json` |
| Ouro: the LoRA at layers 6, 10 and 20; in the last loop only | `e71_ouro_map.py [--apply last]` | `e71_ouro_o14_a{6,10,20}*.json`, `e71_ouro_o26_a{12,24}_last.json` |

### Section 7: multi-hop question answering (Appendix G)

| Result | Scripts | Outputs in `results/` |
|---|---|---|
| Fictional fact chains (Qwen3-8B; Ouro) | `e52_mqa_train.py --evals synth` (data from `synth_mhqa.py`); `e79_ouro_nl_map.py`; `analysis/std_facts_paired.py` | `e52_mqa_rv_q8_syn_map_a14.json`, `e52_mqa_rv_q8_frozen_synth.json`, `e79_ouro_nl_o14_a6.json` |
| MuSiQue, standard models: the LoRA by layer, projection LoRA on early, late or all layers, a LoRA trained on SQuAD, a steering vector, FLAS-style flows | `e52_mqa_train.py`; `analysis/std_musique_paired.py` | `e52_mqa_<tag>.json` (training), `e52_mqa_rv_<tag>.json` (fixed-prompt evaluation with per-question predictions) |
| MuSiQue, Ouro | `e80_ouro_musique.py`; `analysis/loop_musique_paired.py` | `e80_ouro_musique_o14_*_h0.json` |

### Appendices H-J

| Result | Scripts | Outputs in `results/` |
|---|---|---|
| OLMo-3-7B pretraining checkpoints; retrieval on checkpoints | `e19_reentry.py` with `MODEL@revision`, `e67_ckpt_basic.py` | `e19_reentry_ckpt_olmo_*.json`, `e67_basic_olmo_step5000.json` |
| A base-trained LoRA in post-trained descendants | `e75_zoo.py`, `zoo_parents.py` | `e75_zoo_*.json`, `zoo_parents.json` |
| A relay learned from scratch | `e60_arch_reach.py`, `p3_grid_summary.py`; `analysis/fig_toy.py` | `e60_*.json` |
| Looped reach with intervals | `analysis/tab_reach.py` | (reads the E71, E71b, E81 and E86 outputs) |

## Commands

### Standard models: the default computation

    for m in Qwen/Qwen3-0.6B-Base:q06base Qwen/Qwen3-1.7B-Base:q17base Qwen/Qwen3-4B-Base:q4base Qwen/Qwen3-8B-Base:q8base \
             Qwen/Qwen3-14B-Base:q14base NousResearch/Llama-3.2-1B:llama32_1b unsloth/Llama-3.2-3B:llama32_3b \
             NousResearch/Meta-Llama-3.1-8B:llama31_8b allenai/Olmo-3-1025-7B:olmo3_7b allenai/Olmo-3-1125-32B:olmo3_32b \
             unsloth/gemma-3-4b-pt:gemma3_4b unsloth/gemma-3-12b-pt:gemma3_12b unsloth/gemma-3-27b-pt:gemma3_27b; do
      $PY e10_panel.py ${m%%:*} --tag ${m##*:}                                  # reach, commit depth, value copy
    done
    $PY e32b_wave2.py Qwen/Qwen3-8B-Base --tag relay_q8 --depth 8 --n 600       # default relay front; likewise for the 12 other models
    for d in 1 2 3 4; do $PY e4_trace.py Qwen/Qwen3-8B-Base --tag q8base_val_d$d --depth $d --n 40; done
    $PY e76_window_block.py Qwen/Qwen3-8B-Base --tag q8 --a 17 --b 24 --H 32
    $PY e76_window_block.py allenai/Olmo-3-1025-7B --tag o3 --a 13 --b 20 --H 25
    $PY e76_window_block.py NousResearch/Meta-Llama-3.1-8B --tag l31 --a 10 --b 17 --H 25

The query walk (`e21_adapted_trace.py`) reads only the frozen model (field `off`), but it loads an adapter to run. We used a
rank-64 three-chain adapter at layer 14:

    $PY e19_reentry.py Qwen/Qwen3-8B-Base --tag q8_band14_23_noloop --band 14:23 --no_loop
    $PY e21_adapted_trace.py Qwen/Qwen3-8B-Base --adapter ../results/e19_adapter_q8_band14_23_noloop.pt --a 14 --tag q8_noloop_d5 --depth 5 --n 24

### Standard models: training and evaluating the LoRA

The headline LoRA (Qwen3-8B, layer 14, rank 8, two chains of up to 20 lines, 1,200 steps, a KL penalty on WikiText-103):

    $PY e19_reentry.py Qwen/Qwen3-8B-Base --tag vdeep_q8_r8_kl1 --band 14:23 --no_loop --rank 8 --chains 2 --dmax 20 --var_pool upperlower --depths_eval 4,8,12,16,20,24 --steps 1200 --kmax 1 --text_kl 1.0
    $PY e19_reentry.py Qwen/Qwen3-8B-Base --tag vdeep_q8_r8_kl1_s1 --seed 1 --band 14:23 --no_loop --rank 8 --chains 2 --dmax 20 --var_pool upperlower --depths_eval 4,8,12,16,20,24 --steps 1200 --kmax 1 --text_kl 1.0 --neval 150   # and --seed 2
    $PY e19b_textloss.py Qwen/Qwen3-8B-Base ../results/e19_adapter_vdeep_q8_r8_kl1.pt 14
    $PY e33_order_control.py Qwen/Qwen3-8B-Base --map ../results/e19_adapter_vdeep_q8_r8_kl1.pt --a 14 --tag r8kl_matrix --var_pool letters --conds 2:forward,2:interleave,3:forward,3:interleave,2:reverse,3:reverse --depths 2,4,6,8,12,16,20,24 --n 200

The longer-trained LoRA, and the LoRA with the middle layers looped:

    $PY e19_reentry.py Qwen/Qwen3-8B-Base --tag long_q8_r8_kl1 --band 14:23 --no_loop --rank 8 --chains 2 --dmax 40 --var_pool both --depths_eval 8,16,24,32,40,48,64 --steps 2000 --kmax 1 --text_kl 1.0 --neval 100
    $PY e19_reentry.py Qwen/Qwen3-8B-Base --tag loop_q8_r8_kl1_long --band 14:23 --rank 8 --chains 2 --dmax 40 --var_pool both --depths_eval 16,24,32,40,48,64 --steps 2000 --k_train_set 1,2 --kmax 2 --text_kl 1.0 --neval 100

Transfer to other formats (the three-chain LoRA of `e19_reentry.py --tag q8_noloop_kl1 --band 14:23 --no_loop --text_kl 1.0`):

    $PY e20_adapter_transfer.py Qwen/Qwen3-8B-Base --adapter ../results/e19_adapter_q8_noloop_kl1.pt --band 14:23 --tag q8_kl1_fixed --Ks 0,1 --no_loop

Other edits at layer 14 of Qwen3-8B (E91; trained like the LoRA and evaluated on 200 programs per length):

    $PY e91_intervention_compare.py Qwen/Qwen3-8B-Base --kind map --a 14 --tag q8_map_a14
    $PY e91_intervention_compare.py Qwen/Qwen3-8B-Base --kind lora --a 14 --tag q8_lora_a14                 # projection LoRA
    $PY e91_intervention_compare.py Qwen/Qwen3-8B-Base --kind multimap --layers 6,10,14 --tag q8_multimap_6_10_14
    $PY e91_intervention_compare.py Qwen/Qwen3-8B-Base --kind flow --a 14 --N 3 --tag q8_flow_N3_a14        # and --N 1
    $PY e91_intervention_compare.py Qwen/Qwen3-8B-Base --kind flowblock --a 14 --attn 0 --time 1 --N 3 --tag q8_fb_time_N3
    # flow-block variants: --attn 1 (q8_fb_attn_time_N3), --time 0 (q8_fb_notime_N3), --N 1 (q8_fb_time_N1)
    # after the placement limit: --a 26 with --kind map, lora, flow --N 3 and flowblock --N 1

### Placement sweeps

One rank-8 LoRA at layer `A` of `MODEL`, the same training as above, evaluated on 150 programs per length:

    $PY e19_reentry.py MODEL --tag place_TAG_aA --band A:A --no_loop --rank 8 --chains 2 --dmax 20 --var_pool upperlower --depths_eval 2,3,4,6,8,12,16,20,24 --steps 1200 --kmax 1 --text_kl 1.0 --neval 150 [--seed 1] [--ckpt]

`--ckpt` turns on gradient checkpointing; we used it for Gemma-3-12B and OLMo-3-32B. The tested layers:

| Model | `MODEL` | Tag | Layers `A` |
|---|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B-Base` | `q8` | 6, 10, 14, 17-23, 25, 27, 30 (seed 1 at 20, 21) |
| OLMo-3-7B | `allenai/Olmo-3-1025-7B` | `olmo` | 5, 9, 12, 15, 17, 19, 21, 23, 26, 29 (seed 1 at 12, 15) |
| Llama-3.1-8B | `NousResearch/Meta-Llama-3.1-8B` | `l31` | 8, 11, 13, 15, 17, 20 (seed 1 at 13, 15) |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B-Base` | `q17` | 3, 6, 8, 10, 12, 14, 16, 18, 20, 23 |
| Qwen3-0.6B | `Qwen/Qwen3-0.6B-Base` | `q06` | 3, 6, 8, 10, 12, 14, 16, 18, 20, 23 |
| Llama-3.2-1B (held out) | `NousResearch/Llama-3.2-1B` | `l1b` | 1-8, 10, 12 (seed 1 at 7, 8) |
| Qwen3-4B (held out) | `Qwen/Qwen3-4B-Base` | `q4` | 12, 14, ..., 26 (seed 1 at 20, 22) |
| Gemma-3-12B (held out) | `unsloth/gemma-3-12b-pt` | `g12` | 14, 18, 21, 24, 27, 30, 34 (seed 1 at 24, 27) |
| OLMo-3-32B (held out) | `allenai/Olmo-3-1125-32B` | `o32` | 14, 18, 22, 26, 30, 34 (seed 1 at 22, 26) |
| Llama-3.2-3B | `unsloth/Llama-3.2-3B` | `l32_3b` | 6, 8, 10 |
| Qwen3-14B | `Qwen/Qwen3-14B-Base` | `q14` | 10, 14 |

The rules for the prospective test were fixed before the held-out models were run. `analysis/std_analyze.py` scores them.

Read distances at the limit and in the other models (`--frozen_only` traces the frozen model):

    $PY e37b_strides.py Qwen/Qwen3-8B-Base --map ../results/e19_adapter_place_q8_a20.pt --a 20 --tag q8a20_d16_inter --depth 16 --n 60 --order interleave --jmax 12   # and a21
    $PY e37b_strides.py allenai/Olmo-3-1025-7B --map ../results/e19_adapter_place_olmo_a5.pt --a 5 --tag o3a5_d16_inter --depth 16 --n 60 --order interleave --jmax 12
    $PY e37b_strides.py Qwen/Qwen3-4B-Base --frozen_only --tag frozen_q4_d16_inter --depth 16 --n 60 --order interleave --jmax 12

### Qwen3-8B: the relay

Every trace loads the headline LoRA:

    M="--map ../results/e19_adapter_vdeep_q8_r8_kl1.pt --a 14"
    $PY e32b_wave2.py Qwen/Qwen3-8B-Base $M --tag q8r8kl_d16 --depth 16 --n 600
    $PY e32b_wave2.py Qwen/Qwen3-8B-Base $M --tag q8r8kl_d16_inter --depth 16 --n 600 --order interleave
    $PY e37b_strides.py Qwen/Qwen3-8B-Base $M --tag q8r8kl_d16_inter --depth 16 --n 60 --order interleave --jmax 12
    $PY e37b_strides.py Qwen/Qwen3-8B-Base $M --tag q8r8kl_d16_level --depth 16 --n 60 --order forward --jmax 12
    $PY e66_head_ablation.py Qwen/Qwen3-8B-Base $M --heads_from ../results/e37b_strides_q8r8kl_d16_inter.json --tag q8r8kl_inter_after --min_layer 14 --order interleave --depths 2,4,8,12,16 --n 150
    H=$($PY pick_relay_heads.py ../results/e37b_strides_q8r8kl_d16_inter.json)
    $PY e68_head_profile.py Qwen/Qwen3-8B-Base $M --heads $H --tag q8r8kl_inter --depth 16 --n 60
    $PY e34_deep_trace.py Qwen/Qwen3-8B-Base $M --rank 8 --tag q8r8kl_d16 --depth 16 --Ks 2,4,6,8,10,12,14,16 --n 16 --modes map
    $PY e38_relay_cut.py Qwen/Qwen3-8B-Base $M --rank 8 --tag q8r8kl_relaylayers --cut 14:23 --late 23:30 --n 100
    $PY e38_relay_cut.py Qwen/Qwen3-8B-Base $M --rank 8 --tag q8r8kl_relaylayers_zero --cut 14:23 --late 23:30 --renorm 0 --n 100
    $PY e62_ancestor_code.py Qwen/Qwen3-8B-Base $M --tag q8r8kl_d16 --depth 16 --n 400
    $PY e65_map_subspace.py Qwen/Qwen3-8B-Base $M --tag q8r8kl --n 400
    for w in program query all; do $PY e44_position_maps.py Qwen/Qwen3-8B-Base --tag q8_${w}_v2 --where $w; done
    # Llama-3.1-8B, with its placement LoRA at layer 11:
    $PY e32b_wave2.py NousResearch/Meta-Llama-3.1-8B --map ../results/e19_adapter_place_l31_a11.pt --a 11 --tag l31r8_a11_d16 --depth 16 --n 600

### Pretraining checkpoints and post-trained models

    for rev in stage1-step5000 stage1-step20000 stage1-step80000; do
      $PY e19_reentry.py allenai/Olmo-3-1025-7B@$rev --tag ckpt_olmo_${rev//-/}_a6_r8 --band 6:6 --no_loop --rank 8 --chains 2 --dmax 20 --var_pool upperlower --depths_eval 2,3,4,6,8,12,16,20,24 --steps 1200 --kmax 1 --text_kl 1.0 --neval 150
    done
    $PY e67_ckpt_basic.py allenai/Olmo-3-1025-7B@stage1-step5000 --tag olmo_step5000 --maps ../results/e19_adapter_ckpt_olmo_stage1step5000_a6_r8.pt:6

A base-trained LoRA (the placement LoRA named in `--base_map`) evaluated unchanged in post-trained descendants, then their weight
distances (`zoo_parents.py` reads the weights from the local model cache):

    $PY e75_zoo.py --base_map ../results/e19_adapter_place_q17_a10.pt:10 --tag q17 --models Qwen/Qwen3-1.7B-Base,Qwen/Qwen3-1.7B,logan7000/grpo-qwen3-1p7b-math345-best,logan7000/grpo-qwen3-1p7b-math345-end,Thinking-Space/Qwen3-1.7B-Base-OPD,Thinking-Space/Qwen3-1.7B-SFT,boyanbiji/Qwen3-1.7B-OPD-step120,Klingspor/Qwen3-1.7B-SFT,ayushshah/Qwen3-1.7B-UltraChat-SFT,reaperdoesntknow/Qwen3-1.7B-Coder-Distilled-SFT,AI-MO/Kimina-Prover-Distill-1.7B,FRPO/qwen3-1.7b-a1_base-k1-cNone-clip0.2-mb4-eta100-bs256x5-n2
    $PY e75_zoo.py --base_map ../results/e19_adapter_place_q4_a16.pt:16 --tag q4 --models Qwen/Qwen3-4B-Base,Qwen/Qwen3-4B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-4B-Thinking-2507,Qwen/Qwen3-4B-SafeRL,Thinking-Space/Qwen3-4B-Base-GRPO,NotoriousH2/Qwen3-4B-Countdown-RLVR,caiyuchen/Qwen3-4B-Non-Thinking-Math-RL,FRPO/qwen3-4b-a1_base-k1-cNone-clip0.2-mb4-eta100-bs256x5-n4,Tuwhy/Qwen3-4B-OPSA,graf/Qwen3-4B-SFT-science-1e-5,AmberYifan/capsd-qwen3-numina-Qwen3-4B-Base-math_random_b8000_s0,davidnichols-ops/qwen3-4b-devin-sft
    $PY e75_zoo.py --base_map ../results/e19_adapter_place_q8_a14.pt:14 --tag q8 --models Qwen/Qwen3-8B-Base,Qwen/Qwen3-8B,bobbycxy/Qwen3-8B-Base-SFT-step3000-OpenThoughts3,IIGroup/X-Coder-SFT-Qwen3-8B,espressovi/SUTRA-qwen3-8b-distil
    $PY e75_zoo.py --base_map ../results/e19_adapter_place_olmo_a5.pt:5 --tag o3 --models allenai/Olmo-3-1025-7B,allenai/Olmo-3-7B-Think-SFT,allenai/Olmo-3-7B-Think-DPO,allenai/Olmo-3-7B-Think,allenai/Olmo-3-7B-Instruct,allenai/Olmo-3-7B-RL-Zero-Math
    $PY e75_zoo.py --base_map ../results/e19_adapter_place_l31_a8.pt:8 --tag l31 --models NousResearch/Meta-Llama-3.1-8B,NousResearch/Meta-Llama-3.1-8B-Instruct
    $PY zoo_parents.py

### Looped models (Ouro, Huginn)

    $LPY e6_ouro_vb.py ByteDance/Ouro-1.4B --tag ouro14_nl_n300 --Tmax 10 --n 300          # default reach by loops
    $LPY e74_ouro_hops.py --model ByteDance/Ouro-1.4B --tag o14                            # fact chains by hop
    $LPY e88_fewshot.py --arch ouro --tag ouro14 --shots 0,4 --steps 1,2,3,4,6,8 --depths 1,2,3,4,6,8 --n 100
    $LPY e39_ouro_relay.py ByteDance/Ouro-1.4B --tag ouro14                                 # the frozen loops do not relay
    $LPY e8_ouro_trace.py ByteDance/Ouro-1.4B --tag o14_ptr_d3_T4_l3 --depth 3 --T 4 --kind ptr --level 3 --n 30
    $LPY e78_ouro_block.py --tag o14_frozen --T 4 --chains 3 --depths 2,3

The Ouro LoRA (layer 6 of Ouro-1.4B unless stated) and its variants:

    $LPY e71_ouro_map.py --a 6 --tag o14_a6
    $LPY e71_ouro_map.py --a 6 --seed 1 --tag o14_a6_s1 --Ts_eval 1,2,3,4,6,8                   # and seed 2
    $LPY e71_ouro_map.py --a 6 --steer --tag o14_a6_steer --Ts_eval 1,2,3,4,6,8                 # steering vector
    $LPY e71_ouro_map.py --a 6 --apply first --tag o14_a6_first                                 # first loop only
    $LPY e71_ouro_map.py --a 6 --T_train 2 --tag o14_a6_T2 --Ts_eval 1,2,3,4,6,8                # and --T_train 1
    $LPY e71_ouro_map.py --a 6 --pool both --dmax 40 --tag o14_a6_long --Ts_eval 2,3,4,6,8,12 --depths_eval 4,8,16,24,32,40,48,64,96 --neval 100
    $LPY e71_ouro_map.py --a 6 --pool both --dmax 40 --chains 2,3,4 --eval_chains 2,3,4 --tag o14_a6_long_c234 --Ts_eval 2,3,4,6,8 --depths_eval 8,16,24,32,48,64 --neval 100
    $LPY e71_ouro_map.py --a 10 --tag o14_a10                                                   # and --a 20
    $LPY e71_ouro_map.py --a 6 --apply last --tag o14_a6_last --Ts_eval 1,2,3,4,6,8             # and --a 20
    $LPY e71_ouro_map.py --model ByteDance/Ouro-2.6B --a 12 --tag o26_a12 --Ts_eval 1,2,3,4,6,8
    $LPY e71b_ouro_scaling.py --map ../results/e71_map_o14_a6.pt --a 6 --tag o14_a6 --Ts 1,2,3,4,6,8,12,16 --depths 8,16,24,32,48,64 --n 100 --frozen 1
    $LPY e71b_ouro_scaling.py --map ../results/e71_map_o14_a6_long.pt --a 6 --pool both --tag o14_a6long_128 --Ts 6,8,12 --depths 96,128,160 --n 60 --frozen 0
    $LPY e71c_ouro_controls.py --map ../results/e71_map_o14_a6.pt --a 6 --tag o14_a6

The relay in Ouro:

    $LPY e77_ouro_relay_map.py ByteDance/Ouro-1.4B --map ../results/e71_map_o14_a6.pt --a 6 --tag o14_a6 --T 4 --depth 16 --n 600 --mode map   # and --mode frozen
    $LPY e82_ouro_strides.py --map ../results/e71_map_o14_a6.pt --a 6 --tag o14_a6_d16 --T 6 --depth 16 --n 60 --jmax 15
    $LPY e85_ouro_stride_block.py --tag o14_a6_nr --map ../results/e71_map_o14_a6.pt --a 6 --Ts 3,4 --depths 4,8,12,16,20,24 --n 100 --conds none,long,same_far,other_far,rand_far,near_same,near_other --renorm 0
    $LPY e85_ouro_stride_block.py --tag o14_a6_nr_band --map ../results/e71_map_o14_a6.pt --a 6 --Ts 3,4 --depths 4,8,12,16,20,24 --n 100 --conds none,near_same,long --renorm 0 --mask_layers 7-15
    $LPY e85_ouro_stride_block.py --tag o14_a6_abl_far --map ../results/e71_map_o14_a6.pt --a 6 --Ts 3,4 --depths 4,8,12,16,20,24 --n 100 --conds none --ablate 12:6,14:12,7:7,15:4,6:3,17:7,1:1
    $LPY e85_ouro_stride_block.py --tag o14_a6_abl_far_rand20 --map ../results/e71_map_o14_a6.pt --a 6 --Ts 3,4 --depths 4,8,12,16,20,24 --n 100 --ablate_sets HEADSETS.json
    $LPY e87_ouro_ancestor_probe.py --tag o14_a6 --map ../results/e71_map_o14_a6.pt --a 6 --T 4 --depth 16 --n 400 --jmax 10

`--ablate_sets` takes a JSON list of random head sets: 20 sets, each with one head per layer of the target heads.

Huginn:

    $LPY e81_huginn_map.py --tag hug_r8 --r_train 8 --steps 600 --accum 8
    $LPY e81b_huginn_long.py --tag hug_long_r8 --r_train 8 --dmin 8 --dmax 24 --steps 1000
    $LPY e81c_huginn_ppl.py --map ../results/e81_map_hug_long_r8.pt --tag hug_long_r8 --rs 8,16 --n 200
    $LPY e83_huginn_relay.py --map ../results/e81_map_hug_r8.pt --tag hug_r8_d16 --r 16 --depth 16 --n 500
    $LPY e84_huginn_strides.py --map ../results/e81_map_hug_r8.pt --tag hug_r8_d16 --r 12 --depth 16 --n 60 --jmax 15
    $LPY e86_huginn_stride_block.py --map ../results/e81_map_hug_r8.pt --tag hug_r8 --rs 4,8,16,32 --depths 4,8,12,16,20,24 --n 60

### Fact chains and MuSiQue

Standard models. Training (the WikiText penalty is on by default):

    export PYTHONHASHSEED=0
    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag q8_mus_map_a6 --kind map --a 6 --rank 8 --steps 1500 --micro 4               # the LoRA at layer 6; likewise 10, 14, 20, 26, 30
    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag q8_mus_lora8_lr3e4 --kind lora --rank 8 --lr 3e-4 --steps 1500 --micro 2     # projection LoRA, all layers
    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag q8_mus_lora8_pre21_lr3e4 --kind lora --rank 8 --lr 3e-4 --steps 1500 --micro 2 --lora_layers 0:21
    # likewise --lora_layers 21:36 (post21) and the quarters 0:9, 9:18, 18:27, 27:36
    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag q8_squad_map_a14 --kind map --a 14 --rank 8 --steps 1500 --train squad
    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag q8_mus_flow_a6 --kind flow --a 6 --rank 8 --N 3 --steps 1500 --evals musique --settings gold
    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag q8_mus_flowmlp_a6 --kind flowmlp --a 6 --N 3 --time 1 --lr 1e-4 --steps 1500 --evals musique --settings gold
    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag q8_syn_map_a14 --kind map --a 14 --rank 8 --steps 1500 --train synth         # fact chains
    # OLMo-3-7B and Llama-3.1-8B: the same commands with allenai/Olmo-3-1025-7B (tags o3_*) and NousResearch/Meta-Llama-3.1-8B
    # (tags l31_*), --lr 1e-4 for projection LoRA, and early/late ranges 0:15 / 15:32

Evaluation on the 900 development questions with gold paragraphs, saving per-question predictions (the numbers in the paper):

    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag rv_q8_frozen --kind none --evals musique --settings gold --save_preds 1 --text_kl 0 --eval_n 900
    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag rv_q8_mus_map_a6 --kind map --a 6 --rank 8 --load ../results/e52_adapter_q8_mus_map_a6.pt --evals musique --settings gold --save_preds 1 --text_kl 0 --eval_n 900
    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag rv_q8_frozen_synth --kind none --evals synth --settings gold --save_preds 1 --text_kl 0
    $PY e52_mqa_train.py Qwen/Qwen3-8B-Base --tag rv_q8_syn_map_a14 --kind map --a 14 --rank 8 --load ../results/e52_adapter_q8_syn_map_a14.pt --evals synth --settings gold --save_preds 1 --text_kl 0
    # every other adapter: --tag rv_<tag> --load ../results/e52_adapter_<tag>.pt with the options it was trained with

Ouro:

    $LPY e79_ouro_nl_map.py --a 6 --tag o14_a6                                                 # fact chains
    $LPY e80_ouro_musique.py --a 6 --tag o14_a6                                                # trains the LoRA
    PYTHONHASHSEED=0 $LPY e80_ouro_musique.py --a 6 --load ../results/e80_map_o14_a6.pt --tag o14_a6_h0 --Ts_eval 2,3,4,6,8
    PYTHONHASHSEED=0 $LPY e80_ouro_musique.py --a 6 --kind lora --rank 4 --tag o14_a6_lora_h0 --Ts_eval 2,3,4,6,8
    PYTHONHASHSEED=0 $LPY e80_ouro_musique.py --a 6 --kind steer --tag o14_a6_steer_h0 --Ts_eval 2,3,4,6,8
    PYTHONHASHSEED=0 $LPY e80_ouro_musique.py --a 6 --kind prompt --tag o14_prompt_h0 --Ts_eval 2,3,4,6,8
    # layer 20 and the last loop: --a 20, --apply last (train without --load first, then evaluate with it)

### A relay learned from scratch

A looped toy transformer (a two-layer body applied `L/2` times), trained up to `D`-line chains (`D` = 8 or 16), then analyzed:

    $PY e60_arch_reach.py --arch loop --loop_block 2 --layers L --dmax D --eval_dmax D+4 --chains 2,3 --nvars 64 --curriculum 1 --steps 30000 --seed S --mech_d 8 --tag p3g_loop_LL_dD_sS
    $PY e60_arch_reach.py --arch loop --loop_block 2 --layers L --dmax D --eval_dmax D+4 --chains 2,3 --nvars 64 --tag p3g_loop_LL_dD_sS --analyze 1 --mech_d 8
    $PY e60_arch_reach.py --arch loop --loop_block 2 --layers L --dmax D --eval_dmax D+4 --chains 2,3 --nvars 64 --tag p3g_loop_LL_dD_sS --strides 8
    $PY e60_arch_reach.py --arch loop --loop_block 2 --layers L --dmax D --eval_dmax D+4 --chains 2,3 --nvars 64 --tag p3g_loop_LL_dD_sS --names 1 --mech_d D
    $PY e60_arch_reach.py --arch loop --loop_block 2 --layers L --dmax D --eval_dmax D+4 --chains 2,3 --nvars 64 --tag p3g_loop_LL_dD_sS --masks 1 --mask_depths 2,4,6,8,12,16,20

The grid uses `L` in {4, 6, 8, 12, 16}, `D` in {8, 16} and seeds 1 and 2. Write `D+4` as a number (12 or 20). The mask depths
are 2,4,6,8,10,12 for `D` = 8 and 2,4,6,8,12,16,20 for `D` = 16.

### Analysis

    cd analysis && ./build.sh ../.venv-std/bin/python

`build.sh` runs every analysis script in order. It writes tables, checks and draft figures to `analysis/out/`.

## Notes

- Several runs were launched by hand or on a cluster. The settings of every run are in `args` of its result file.
- The MuSiQue adapters were trained with the per-process hash as well. Set `PYTHONHASHSEED=0` for a fully deterministic re-run.
