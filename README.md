
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Hierarchical Recurrent Planner

This fork extends vanilla LeWM with a **hierarchical, recurrent world state**. Instead
of representing each observation as a single latent embedding, the model maintains the
state at **three abstraction levels** that are iteratively reconciled before every
prediction. The relevant code lives in [`jepa_planning.py`](jepa_planning.py) (the
world-model wiring) and [`module_planning.py`](module_planning.py) (the building
blocks). Everything below is opt-in — selected with `--config-name=lewm_planning` — and
collapses back to the vanilla baseline when the recurrence is switched off.

### 1. Motivation

Vanilla LeWM encodes an image into one vector `emb`, predicts the next `emb` from a
short history of `emb`s + actions, and plans by rolling that scalar embedding forward
(see [`jepa.py`](jepa.py)). A single vector has to simultaneously carry fine pixel
detail *and* the global gist of the scene, and the predictor changes all of it at the
same rate. For control tasks like PushT this is wasteful: the global goal ("the T is in
the target pose") changes slowly, while local detail ("where exactly is the pusher this
frame") changes fast.

The hierarchical planner makes that structure explicit. The world state is split into
three sets of latent tokens, each with its own meaning, time scale, and loss weight:

| Level | Tokens | Intuition (from the design notes) | Predictor history | Loss weight |
|:-----:|:------:|:----------------------------------|:-----------------:|:-----------:|
| **Z15** (`z_detail`) | 15 | fine, local detail — *"what I see"* | 2 frames (short) | 1.0 |
| **Z5**  (`z_rel`)    | 5  | relations / structure — *"what it means"* | 4 frames (mid) | 0.5 |
| **Z1**  (`z_glob`)   | 1  | global / abstract — *"what follows"* | 8 frames (long) | 0.25 |

`emb` is still produced as a **compatibility readout** so all existing LeWM losses,
evaluation, and the `stable_worldmodel` cost API keep working unchanged — but the
*actual* world state the planner reasons over is the structured `(Z15, Z5, Z1)` triple.

### 2. From pixels to a hierarchical state — `RecursiveStateEncoder`

The encoder is the same ViT-tiny as vanilla LeWM (patch 14, image 224, `embed_dim=192`,
trained from scratch). The new part is what happens to its patch tokens.

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

**Pooling.** Each level has its own set of learned query tokens (`q_detail`, `q_rel`,
`q_glob`) that cross-attend over the patch tokens (`CrossAttentionBlock`). The query is
kept as a residual so the 15 detail tokens stay distinct from one another instead of
collapsing to a single pooled vector.

**Reconciliation (`RecursiveStateBlock`).** This is the *recurrent* part. For `K`
cycles the levels exchange information, bottom-up then top-down:

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

The **gate** lets the model decide how strong each correction should be: small when a
level is already consistent with the others, large when it must be pulled into
agreement. Top-down passes are *corrections*, not generation — a residual `delta` added
to the existing level, never a replacement.

`K` is effectively **adaptive compute**: `num_cycles=0` returns the pooled levels
untouched (no recurrence — the closest thing to the vanilla baseline), while `K=1/2/3`
are the ablations. Default is `K=2`.

**Fusion (`fuse_levels`).** To produce the `emb` readout without throwing the hierarchy
away, `Z5` and `Z1` cross-attend into `Z15` through their own gates, yielding
`fused_z15`; `emb = readout(mean(fused_z15))`. The three levels survive fusion intact
and remain the model's state.

> There is also a lighter sibling, `RecursiveStateRefiner`, which instead folds the
> three levels into a **zero-initialised residual correction** around a vanilla `emb`.
> It is an exact identity at init, so it can be bolted onto the baseline as an ablatable
> add-on. The config in this fork uses the full `RecursiveStateEncoder` (state *is*
> hierarchical); the refiner is kept for the "is the hierarchy even helping?" ablation.

### 3. Predicting the future — `HierarchicalStatePredictor`

Vanilla LeWM predicts the next `emb` with one autoregressive transformer. Here each
level is predicted by its **own planner** (a transformer with AdaLN-zero action
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
  medium (`z5_history=4`), Z15 the shortest (`z15_history=2`) — slow/abstract state is
  predicted from long context, fast/detailed state from recent context.
- **Top-down conditioning.** `pred_Z1` conditions the relation planner; `pred_Z5`
  conditions the detail planner. Coarse decisions are made first and steer the finer
  predictions, mirroring the top-down half of the reconciliation loop.
- **Residual prediction.** Each level predicts a residual added to its last state
  (`pred = LayerNorm(last_state + plan)`), with a `GatedCrossUpdate` seed carrying the
  coarse prediction into the finer level. Output is one next state per level.

### 4. Putting it together — `PlanningJEPA`

[`jepa_planning.py`](jepa_planning.py) subclasses the original `JEPA` and overrides the
state-aware paths while staying API-compatible:

- **`encode`** → ViT patch tokens → `RecursiveStateEncoder` → `{z15, z5, z1, fused_z15, emb}`.
- **`predict_state`** → `HierarchicalStatePredictor` → next `{z15, z5, z1}` + an `emb` readout.
- **`rollout`** → autoregresses **the structured state** (not just `emb`): it keeps a
  per-level history, truncates to `history_size`, predicts the next state, appends, and
  returns `predicted_z15/z5/z1` alongside `predicted_emb`.
- **`criterion` / `get_cost`** → the planning (MPC) cost is computed **per level** and
  summed with the same `1.0 / 0.5 / 0.25` weights. The goal image is encoded into
  `goal_z15/z5/z1`, candidate action sequences are rolled out, and the last-step MSE to
  the goal state at each level becomes the cost the planner minimises. If the structured
  goal/prediction tensors are absent it transparently falls back to the vanilla
  `emb`-only cost.

### 5. Training objective

Training reuses the LeWM recipe (`lejepa_forward` in [`train.py`](train.py)): encode the
sequence, predict next states from the first `history_size` frames, and combine **one
prediction loss with one regularizer**. The only change is that the prediction loss is
now **multi-scale**:

```
pred_loss = pred_z15_loss + 0.5 · pred_z5_loss + 0.25 · pred_z1_loss     # per-level MSE
loss      = pred_loss + λ · sigreg_loss                                  # λ = 0.09
```

`sigreg_loss` is the **SIGReg** (Sketch Isotropic Gaussian Regularizer) from LeWM — a
single-GPU Epps–Pulley statistic over random projections that keeps the latent space
Gaussian and prevents collapse. It is the *only* regularizer the model relies on; no EMA,
no stop-grad tricks, no pretrained encoder. (`pred_emb_loss` is also logged for parity
with the baseline but is not part of the optimised objective.)

### 6. Configuration & ablations

The model is assembled by Hydra from
[`config/train/model/lewm_planning.yaml`](config/train/model/lewm_planning.yaml); the run
config is [`config/train/lewm_planning.yaml`](config/train/lewm_planning.yaml). Key knobs:

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

**Diagnostics.** Setting `collect_diagnostics=True` on the encoder records detached
metrics each step — per-level token diversity (collapse check) and the mean correction
magnitude / gate value of the top-down passes — exposed via `encoder.diagnostics`.

### Run it

```bash
# hierarchical recurrent planner
python train.py --config-name=lewm_planning data=pusht

# baseline-ish: same model, recurrence disabled
python train.py --config-name=lewm_planning data=pusht state_encoder.num_cycles=0
```

See [Training](#training) and [Planning](#planning) below for the shared data, logging,
and evaluation setup.

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch the vanilla LeWM training run:
```bash
python train.py --config-name=lewm data=pusht
```

To launch the hierarchical recurrent planner variant:
```bash
python train.py --config-name=lewm_planning data=pusht
```

You can also switch only the model group from the default config:
```bash
python train.py model=lewm_planning output_model_name=lewm_planning data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
