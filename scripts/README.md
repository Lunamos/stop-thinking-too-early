# Experiment code

One script per experiment; each has a docstring with its question and usage. Scripts run from this directory and write to
`../results/` (maps and adapters as `*.pt`). Standard-model scripts need `requirements-standard.txt`; the Ouro and Huginn scripts
need `requirements-looped.txt` (transformers 4.56). `REPRODUCE.md` at the top of the repository gives the commands behind the
results of the paper.

`jobq.sh GPU QUEUEFILE` runs the lines of a queue file one after another on one GPU (finished lines are appended to
`QUEUEFILE.done`); `waitmem.sh GPU MIB` waits until a GPU has that much free memory. Set `PY` to the interpreter and `HF_HOME`
to the model cache before starting `jobq.sh`.

## Standard models: the default computation (standard-model environment)

| Script | What it does |
|---|---|
| `e10_panel.py` | E10: cross-model panel for the depth clock. For one standard HF decoder: 1. VB accuracy (acc, racc) vs depth d=1..6 (n per depth). 2. Value trace (root-value counterfactual) for d in --val_depths: sha... |
| `e4_trace.py` | E4: causal tracing of value propagation in variable binding. For each prompt we build a counterfactual (cf) prompt that differs only in the root value of the QUERY chain (v -> v'). Keep pairs where th... |
| `e21_adapted_trace.py` | E21: how does a small adapter change the clock? Loads a no-loop re-entry adapter (applied once to all positions at the input of block a) and runs (i) the pointer causal trace (level K pointer counterf... |
| `e32b_wave2.py` | E32b: where does a mapped model resolve long chains? For each line of the queried chain, fit per-layer linear read-outs at two positions of that line (the right-hand-side token and the line-ending new... |
| `e76_window_block.py` | E76: a causal test of the routing window. Block the attention from the query tokens (print(X)\nOutput:) to every program line except the root lines (so the value can still be copied) in a band of laye... |
| `commit_depth.py` | Commit depth from pointer traces (e10 panel JSONs): for pointer counterfactuals on three-line chains, the first layer at which the pointer's own token holds less than half of the effect, for the point... |

## Standard models: the map and what it does

| Script | What it does |
|---|---|
| `e19_reentry.py` | E19: retrofit a loop into a frozen pretrained LM by learning ONLY a small re-entry adapter at the loop boundary. Schedule with K re-entries of band [a, b]: blocks 0..b ; repeat K times { h <- M(h) ; b... |
| `e19b_textloss.py` | text loss (WikiText) of the frozen model with / without a no-loop adapter at block a. |
| `e33_order_control.py` | E33: do deep maps chase pointers, or exploit the level structure of forward-ordered prompts? With two chains in forward order, every level holds one line of each chain, so the root could in principle ... |
| `e20_adapter_transfer.py` | E20: does the re-entry adapter (trained on code-format forward-order VB) transfer to other surface forms of pointer chasing? Formats: code (training), code_shuffled (random line order), js, natural. A... |
| `e34_deep_trace.py` | E34: where does pointer information flow when a mapped model follows a long chain? Pointer counterfactuals on long two-chain programs (line K of the queried chain redirected to the other chain), with ... |
| `e37b_strides.py` | E37b: chain-selective relay strides with fixed heads. At the token naming the previous variable on line k (level k of the queried chain), attention to the same token on the line j levels earlier in th... |
| `e68_head_profile.py` | E68: per-head attention profiles of the relay heads. For chosen heads, at the token naming the previous variable on each line k of the queried programs, the attention to the same token on every ancest... |
| `pick_relay_heads.py` | Picks the heads for E68 from an E37b stride file by rule: the stride-1 head of the first measured layer, then the distinct heads whose same-chain minus other-chain attention is at least 0.1 two or more lines up in the given layers. |
| `e66_head_ablation.py` | E66: are the long-stride relay heads necessary? With a map at layer a, zero the outputs of the heads that carry strides of two or more lines (chosen by E37b on held-out programs: for each layer and st... |
| `e38_relay_cut.py` | E38: is the relay necessary? Cut the attention from the right-hand-side token of every program line to the line that defines its variable (the four tokens of that line), in a band of layers, for all h... |
| `e91_intervention_compare.py` | E91: which small change starts the relay? Trains one change at one layer of a frozen standard model exactly like the map (map, rank-8 LoRA of all projections, FLAS-style low-rank flow, FLAS flow block with or without self-attention and time embedding, the map at several layers) and evaluates exact accuracy by chain length. |
| `e44_position_maps.py` | E44: where must the map act? Train the same single-layer map (rank r, at the input of block a) applied only at the program's tokens, only at the query tokens (the 'print(X)\nOutput:' suffix), or at al... |
| `e65_map_subspace.py` | E65: does the relay run inside the map's own subspace? A rank-r map writes rms(h) B A h/rms(h), i.e. only into span(B) (r dimensions). With the map on, read out each line's chain (which chain's root l... |
| `e62_ancestor_code.py` | E62: which ancestors does each program line know, at which layer, and in what code? At the right-hand-side token of line k (chain c), the target anc_j is the name of the variable defined j levels earl... |
| `e67_ckpt_basic.py` | E67: basic retrieval in early checkpoints. For a model (optionally model@revision) and saved rank-r maps, exact accuracy and choice among the roots on one- to four-line chains (two chains, upper- and ... |
| `e75_zoo.py` | E75: post-training zoo. For post-trained descendants of a base model: default reach on setup A (three chains, choice among roots, chains of 1-6 lines) and whether a map trained on the BASE model, appl... |
| `zoo_parents.py` | Weight-verified parents for the post-training zoo. For every descendant, the relative Frobenius distance of q_proj and down_proj at three depths (1/4, 1/2, 3/4) to each candidate parent (the base and ... |

## Question answering (standard models)

| Script | What it does |
|---|---|
| `e52_mqa_train.py` | E52: post-training for direct-answer multi-hop QA with a tiny single-layer map vs LoRA. Trains on MuSiQue (or HotpotQA) train with teacher-forced cross-entropy on the answer, plus KL(frozen // adapted... |
| `export_qa_data.py` | Export the question-answering selections to loopdyn/data/ as JSON (for the looped-model environment, whose older `datasets` cannot load the benchmarks, and for the analysis scripts, which need no GPU)... |

## Looped models: the default (looped-model environment, `.venv-loop`)

| Script | What it does |
|---|---|
| `e6_ouro_vb.py` | E6: variable-binding accuracy of a looped LM (Ouro) as a function of the number of recurrent steps T and the chain depth d. Runs in the transformers-4.56 venv (loopdyn/.venv-loop). Uses the model's ow... |
| `e74_ouro_hops.py` | E74: does a looped model add about one hop per loop on natural-language multi-hop questions too? Synthetic questions about fictional entities (synth_mhqa: a chain of typed facts shuffled among two dis... |
| `e8_ouro_trace.py` | E8: causal tracing inside a looped LM (Ouro), value and pointer counterfactuals. Same design as e4_trace.py / e4b_pointer.py but steps run over (loop, layer). Scored with the restricted metric LD = lo... |
| `e78_ouro_block.py` | E78: where in a looped model does the query read the program? Ouro-1.4B (T loops of 24 layers), frozen or with an E71 map. Block the attention from the query tokens (print(X)\nOutput:) to every progra... |
| `e39_ouro_relay.py` | E39: does a looped model relay chain identity through the program one step per loop? Ouro, two chains, level-ordered programs; binary read-out of which chain a line belongs to (named by the order of t... |
| `e88_fewshot.py` | E88: few-shot baseline for the default. Frozen Ouro (T loops) or Huginn (r recurrences) on two-chain programs, with k solved demonstration programs (lengths 1-6, answers given) before the test program... |

## Looped models: the map, the relay band, masks and names

| Script | What it does |
|---|---|
| `e71_ouro_map.py` | E71: a map in a looped model. Ouro-1.4B applies its 24 layers T times (trained with T=4). We insert the rank-r map of Eq. 1 at the input of layer a of the loop body, applied in every loop (or only in ... |
| `e71b_ouro_scaling.py` | E71b: with a map switched on, does a looped model's reach keep growing with the number of loops at inference? Ouro-1.4B with a map trained by e71_ouro_map.py (T=4, chains of up to 20 lines), evaluated... |
| `e71c_ouro_controls.py` | E71c: controls for the map in Ouro. (1) WikiText perplexity at T loops, frozen and with the map. (2) Specificity: programs in reversed order (every line names a variable defined later) and interleaved... |
| `e77_ouro_relay_map.py` | E77: the relay in a looped model with a map switched on. As E39 (binary read-out of which chain each line of the queried chain belongs to, at its right-hand-side token, at every recorded step (loop, l... |
| `e82_ouro_strides.py` | E82: chain-selective attention strides per loop in Ouro. As E37b (at the token naming the previous variable on line k of a chain, attention to the same token on the line j levels earlier in the same c... |
| `e82_head_profiles.py` | Per-head stride profiles from E82 (Ouro) or E84 (Huginn) runs that saved same_full / other_full ([step, head, stride], held-out programs). For the heads with the largest chain selectivity at strides >... |
| `e85_ouro_stride_block.py` | E85: are the long strides needed? Ouro-1.4B with an E71 map (or frozen). In every step of every loop, the tokens of each program line at level k may attend only to the header, to lines of their own le... |
| `e87_ouro_ancestor_probe.py` | E87: does each line accumulate its chain's ancestors across loops? Ouro-1.4B, frozen or with an E71 map. At the token naming the previous variable on line k (the pointer token, which is itself the nam... |
| `e87_split_prefix.py` | Split the E87 ancestor-name probe by the labelled prefix: for each loop t, lines k inside the prefix that the chain read-out (E77) has labelled by the end of loop t, and lines beyond it. If lines gath... |
| `e81_huginn_map.py` | E81: a map in a second looped architecture. Huginn-0125 (2 prelude layers, a 4-layer recurrent core applied r times, 2 coda layers). A rank-r map is applied to the output of the core's input adapter, ... |
| `e81b_huginn_long.py` | E81b: the Huginn map on long chains. Same map and hook as E81 (rank-r map on the output of the core's input adapter, so it acts at the start of the 4-layer core in every recurrence), with three change... |
| `e81c_huginn_ppl.py` | E81c: does the Huginn map keep the model's text predictions? WikiText-103 perplexity of Huginn-0125, frozen and with an E81/E81b map on the core's input adapter, at several recurrence counts (same ran... |
| `e83_huginn_relay.py` | E83: the relay per recurrence in Huginn. As E77 for Ouro: a binary ridge read-out of which chain each line of the queried chain belongs to, at the line's right-hand-side token, from the state after ev... |
| `e84_huginn_strides.py` | E84: chain-selective attention strides per recurrence in Huginn. As E82 for Ouro: at the token naming the previous variable on line k of a chain, attention to the same token on the line j levels earli... |
| `e86_huginn_stride_block.py` | E86: are the long strides needed in Huginn? As E85 for Ouro: in every block (prelude, every recurrence of the core, coda), the tokens of each program line at level k may attend only to the header, to ... |

## Looped models: fact chains and MuSiQue

| Script | What it does |
|---|---|
| `e79_ouro_nl_map.py` | E79: a map in a looped model, trained on natural-language multi-hop questions (synthetic chains of facts about fictional entities, 1-4 hops, two distractor chains; target: first token of the answer en... |
| `e80_ouro_musique.py` | E80: does a map make extra loops pay on a real multi-hop benchmark? Ouro-1.4B, rank-8 map at the input of layer a of the loop body (every loop), trained with T=4 loops on MuSiQue training questions (s... |

## Looped transformers trained from scratch (standard-model environment)

| Script | What it does |
|---|---|
| `e60_arch_reach.py` | E60: does the residual design set how far a transformer can follow a chain in one pass? Small transformers trained from scratch on in-context pointer chasing, identical except for how layers communica... |
| `p3_grid_summary.py` | Paper 3: summary of the toy grid (looped 2-layer blocks trained from scratch with an adaptive curriculum, e60_arch_reach.py). For every run: the step at which the curriculum reached its maximal chain ... |

## Shared modules

| Script | What it does |
|---|---|
| `ld_common.py` | Shared utilities for the loop / depth-dynamics study (loopdyn). Core pieces: * load_model(): HF causal LM in bf16 on one GPU, eval mode, no grads. * Runner: executes an arbitrary *layer schedule* (a l... |
| `ld_loop.py` | Runner for Ouro-style looped LMs (transformers 4.56 environment). Ouro forward: h = embed(x); for t in range(T): for l in range(L): h = layer_l(h); h = norm(h) (at every loop end); logits ... |
| `mqa_common.py` | Multi-hop QA utilities: HotpotQA loading, prompts, answer normalization (official HotpotQA EM/F1), greedy generation with an optional single-layer map (rank-r map or steering vector) applied at the in... |
| `synth_mhqa.py` | Synthetic natural-language multi-hop QA with fictional entities: chains of facts over typed relations, written as short titled passages (one fact each, shuffled with facts from distractor chains), and... |
| `interventions.py` | FLAS-style hidden-state edits shared by E52 and E91: `FlowLowRank` (a low-rank velocity field integrated over N Euler steps) and `FlowMLP` (the FLAS flow block without concept encoder or self-attention), with FLAS's sinusoidal time embedding. |
| `figstyle.py` | Shared matplotlib style for loopdyn paper figures (palette from the dataviz reference). |

