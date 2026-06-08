# LeWM — Hierarchical Recurrent Planner

A fork of [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm) that replaces the
single-vector world state with a **hierarchical, recurrent state** maintained at three
abstraction levels and reconciled before every prediction. The base model (a stable,
single-GPU JEPA trained end-to-end from pixels) is kept as-is and credited below; this
README focuses on the planner this fork adds.

Code: [`jepa_planning.py`](jepa_planning.py) (world-model wiring) and
[`module_planning.py`](module_planning.py) (building blocks). Everything is opt-in
(`--config-name=lewm_planning`) and collapses back to the baseline when the recurrence is
switched off (`num_cycles=0`).

---

## 1. Motivation

Vanilla LeWM encodes an image into one vector `emb`, predicts the next `emb` from a short
history of `emb`s + actions, and plans by rolling that scalar embedding forward. A single
vector has to carry fine pixel detail *and* the global gist of the scene at once, and the
predictor changes all of it at the same rate. For control tasks like PushT this is
wasteful: the global goal ("the T is in the target pose") changes slowly, while local
detail ("where exactly is the pusher this frame") changes fast.

The hierarchical planner makes that structure explicit. The world state is split into
three sets of latent tokens, each with its own meaning, time scale, and loss weight:

| Level | Tokens | Intuition | Predictor history | Loss weight |
|:-----:|:------:|:----------|:-----------------:|:-----------:|
| **Z15** (`z_detail`) | 15 | fine, local detail — *"what I see"* | 2 frames (short) | 1.0 |
| **Z5**  (`z_rel`)    | 5  | relations / structure — *"what it means"* | 4 frames (mid) | 0.5 |
| **Z1**  (`z_glob`)   | 1  | global / abstract — *"what follows"* | 8 frames (long) | 0.25 |

A single `emb` is still produced as a **compatibility readout**, so the existing LeWM
losses, evaluation, and the `stable_worldmodel` cost API keep working unchanged — but the
*actual* state the planner reasons over is the structured `(Z15, Z5, Z1)` triple.

## 2. From pixels to a hierarchical state — `RecursiveStateEncoder`

The encoder is a ViT-tiny (patch 14, image 224, `embed_dim=192`, trained from scratch).
The new part is what happens to its patch tokens:

```
ViT-tiny patch tokens  (B, 256, 192)          # CLS token dropped
        │
        │  Perceiver-style pooling: learned query tokens cross-attend the patches
        ├──────────────► Z15  (B, 15, 192)     # 15 learned "detail" queries
        ├──────────────► Z5   (B,  5, 192)     # 5  learned "relation" queries
        └──────────────► Z1   (B,  1, 192)     # 1  learned "global" query
        │
        │  RecursiveStateBlock: reconcile the levels for K cycles  (default K=2)
        ▼
   reconciled (Z15, Z5, Z1)
        │
        │  fuse_levels: Z5/Z1 gently correct Z15 (gated, non-destructive) → fused_z15
        ▼
   emb = readout(mean(fused_z15))              # (B, 192) compatibility vector
```

**Pooling.** Each level has its own learned query tokens (`q_detail`, `q_rel`, `q_glob`)
that cross-attend over the patch tokens (`CrossAttentionBlock`). The query is kept as a
residual so the 15 detail tokens stay distinct instead of collapsing to one pooled vector.

**Reconciliation (`RecursiveStateBlock`)** — the *recurrent* part. For `K` cycles the
levels exchange information, bottom-up then top-down:

```
for _ in range(K):
    Z5  <- update(Z5,  from=Z15)   # bottom-up: detail informs relations
    Z1  <- update(Z1,  from=Z5 )   # bottom-up: relations inform the global gist
    Z5  <- update(Z5,  from=Z1 )   # top-down:  global corrects relations
    Z15 <- update(Z15, from=Z5 )   # top-down:  relations correct detail
```

Each `update` is a **gated residual cross-attention** step (`GatedCrossUpdate`):

```
delta = CrossAttn(q=z, kv=source)
gate  = sigmoid(MLP([mean(z), mean(source)]))     # per-channel, in (0,1)
z     = LayerNorm(z + gate * delta)
z     = z + FeedForward(z)
```

The **gate** decides how strong each correction is: small when a level already agrees
with the others, large when it must be pulled into agreement. Top-down passes are
*corrections* (a residual `delta`), not generation.

`K` is effectively **adaptive compute**: `num_cycles=0` returns the pooled levels
untouched (no recurrence — closest to the baseline), `K=1/2/3` are the ablations. Default
is `K=2`.

**Fusion (`fuse_levels`).** To build the `emb` readout without discarding the hierarchy,
`Z5` and `Z1` cross-attend into `Z15` through their own gates, yielding `fused_z15`;
`emb = readout(mean(fused_z15))`. The three levels survive fusion intact and remain the
model's state.

> A lighter sibling, `RecursiveStateRefiner`, instead folds the three levels into a
> **zero-initialised residual correction** around a vanilla `emb` (exact identity at init),
> for the "is the hierarchy even helping?" ablation. The shipped config uses the full
> `RecursiveStateEncoder`, where the state *is* hierarchical.

## 3. Predicting the future — `HierarchicalStatePredictor`

Each level is predicted by its **own planner** (a transformer with AdaLN-zero action
conditioning), and the coarse levels condition the fine ones top-down:

```
Z1 history (8 frames) ─► global_planner   ──► pred_Z1        # slow, long horizon
                                              │
Z5 history (4 frames) ─► relation_planner ◄──┘ (top-down)
        + action       ──────────────────► pred_Z5           # medium horizon
                                              │
Z15 history (2 frames)─► detail_planner   ◄──┘ (top-down)
        + action       ──────────────────► pred_Z15          # fast, short horizon
```

- **Different time scales.** Z1 attends over the longest history (`z1_history=8`), Z5
  medium (`z5_history=4`), Z15 the shortest (`z15_history=2`): slow/abstract state from
  long context, fast/detailed state from recent context.
- **Top-down conditioning.** `pred_Z1` conditions the relation planner; `pred_Z5`
  conditions the detail planner — coarse decisions are made first and steer the finer ones.
- **Residual prediction.** Each level predicts a residual added to its last state
  (`pred = LayerNorm(last_state + plan)`), with a `GatedCrossUpdate` seed carrying the
  coarse prediction into the finer level. Output is one next state per level.

## 4. Putting it together — `PlanningJEPA`

[`jepa_planning.py`](jepa_planning.py) subclasses the base `JEPA` and overrides the
state-aware paths while staying API-compatible:

- **`encode`** → ViT patch tokens → `RecursiveStateEncoder` → `{z15, z5, z1, fused_z15, emb}`.
- **`predict_state`** → `HierarchicalStatePredictor` → next `{z15, z5, z1}` + an `emb` readout.
- **`rollout`** → autoregresses **the structured state** (not just `emb`): keeps a
  per-level history, truncates to `history_size`, predicts the next state, appends, and
  returns `predicted_z15/z5/z1` alongside `predicted_emb`.
- **`criterion` / `get_cost`** → the planning (MPC) cost is computed **per level** and
  summed with the `1.0 / 0.5 / 0.25` weights. The goal image is encoded into
  `goal_z15/z5/z1`, candidate action sequences are rolled out, and the last-step MSE to
  the goal state at each level is the cost the planner minimises. If the structured
  tensors are absent it falls back transparently to the vanilla `emb`-only cost.

## 5. Training objective

The recipe stays minimal — **one prediction loss + one regularizer** — but the prediction
loss is now **multi-scale**:

```
pred_loss = pred_z15_loss + 0.5 · pred_z5_loss + 0.25 · pred_z1_loss     # per-level MSE
loss      = pred_loss + λ · sigreg_loss                                  # λ = 0.09
```

`sigreg_loss` is **SIGReg** (Sketch Isotropic Gaussian Regularizer), a single-GPU
Epps–Pulley statistic over random projections that keeps the latent space Gaussian and
prevents collapse. It is the *only* regularizer the model relies on — no EMA, no
stop-grad, no pretrained encoder.

## 6. Configuration & ablations

Assembled by Hydra from
[`config/train/model/lewm_planning.yaml`](config/train/model/lewm_planning.yaml); run
config [`config/train/lewm_planning.yaml`](config/train/lewm_planning.yaml). Key knobs:

| Where | Field | Default | Meaning |
|:------|:------|:-------:|:--------|
| `state_encoder` | `level_tokens` | `[15, 5, 1]` | tokens per level (Z15/Z5/Z1) |
| `state_encoder` | `num_cycles` | `2` | reconciliation cycles **K** (0 = no recurrence ≈ baseline) |
| `predictor` | `z1/z5/z15_history` | `8/4/2` | per-level prediction context length |
| `predictor` | `depth`, `heads`, `mlp_dim` | `6`, `16`, `2048` | per-level planner transformer size |
| run | `history_size` | `8` | rollout / context window |
| run | `embed_dim` | `192` | ViT-tiny hidden size (shared by all levels) |
| `loss.sigreg` | `weight` | `0.09` | λ for the Gaussian regularizer |

**Suggested ablations** (all single-GPU, a few hours each):
- `state_encoder.num_cycles=0` — pure pooling, no recurrence (recurrence on/off).
- `state_encoder.num_cycles=1` / `3` — adaptive-compute sweep.
- Swap `state_encoder` to `module_planning.RecursiveStateRefiner` — hierarchy as a
  residual correction vs. hierarchy as the state.

**Diagnostics.** Set `collect_diagnostics=True` on the encoder to record detached metrics
each step — per-level token diversity (collapse check) and the mean correction magnitude /
gate value of the top-down passes — via `encoder.diagnostics`.

---

## Setup & running

This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
(environments, planning, evaluation) and
[stable-pretraining](https://github.com/galilai-group/stable-pretraining) (training).

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

**Data.** Download the PushT data from
[HuggingFace](https://huggingface.co/collections/quentinll/lewm), decompress
(`tar --zstd -xvf archive.tar.zst`), and place the `.h5` files under `$STABLEWM_HOME`
(defaults to `~/.stable-wm/`). Dataset names drop the `.h5` extension — e.g.
[`config/train/data/pusht.yaml`](config/train/data/pusht.yaml) references
`pusht_expert_train` → `$STABLEWM_HOME/pusht_expert_train.h5`.

**Train.**
```bash
# hierarchical recurrent planner
python train.py --config-name=lewm_planning data=pusht

# baseline-ish: same model, recurrence disabled
python train.py --config-name=lewm_planning data=pusht state_encoder.num_cycles=0

# original vanilla LeWM
python train.py --config-name=lewm data=pusht
```
Checkpoints are saved to `$STABLEWM_HOME` each epoch.

**Plan / evaluate.** Eval configs live under `config/eval/`. Set `policy` to the
checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:
```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm_planning
```

## Credit

The base world model, training objective (SIGReg), and the
`stable-worldmodel`/`stable-pretraining` integration are the work of the LeWM authors —
Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero
([paper](https://arxiv.org/pdf/2603.19312v1) ·
[code](https://github.com/lucas-maes/le-wm) ·
[checkpoints & data](https://huggingface.co/collections/quentinll/lewm)).

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

The hierarchical recurrent planner (`jepa_planning.py`, `module_planning.py`, the
`*_planning` configs) is the contribution of this fork.
