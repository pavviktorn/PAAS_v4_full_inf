# PAAS_v4_full_inf — inference build

Standalone, inference-only build of `PAAS_v4_full`. Every weight it needs is in-tree; the only
external dependency is the shared venv (`PAAS_qwen3vl/venv` — a toolchain, not a weight).

**Weights promoted 2026-08-31** from `PAAS_v4_full/runs/finetune_20260825_104051`: all 8 detector
branches retrained on the full 1,366,146-image `no_delete_mids_train`, plus a Qwen3.5-4B LoRA
(48h) + conditioning distillation + merge, and a from-scratch MIDS 4-class head.

## Serving

```bash
bash run_server.sh                    # :3000, GPU 0
APP_PORT=8080 bash run_server.sh
bash stop_server.sh                   # kills by process pattern, verifies the GPU came back
```

Default fusion: `mean{ffaa, A1_9c, A2_9c, gsdA, selop, pespc}`, threshold **0.14085**
(99% real-recall floor on 30,218 held-out frames: real 99.00%, fake 100.00%).

Change the combination without editing code:

```bash
PAAS_COMPONENTS="ffaa,A2_9c,gsdA,selop" bash run_server.sh
```

All eight detectors ship, so any combination is a config edit rather than a retrain. Re-fit the
threshold with the source project's `train/fit_threshold.py` whenever you change the member set —
a threshold belongs to the fusion it was fitted for.

## What was optimized out (86 GB → 23 GB)

| removed | size | why |
|---|---|---|
| `runs/` | 37 GB | training artifacts, per-run logs, feature caches |
| `weights/_prev_20260831` | 12 GB | rollback copies — they stay in the training project |
| `base_models/Qwen3.5-4B` | 8.8 GB | LoRA **training** base; inference serves the merged model |
| `t5-base` duplicates | 3.4 GB | shipped 5 copies of one model (bin/flax/rust/tf); kept safetensors |
| `clip` `tf_model.h5` | 1.7 GB | TensorFlow copy of `pytorch_model.bin` |
| `weights/_legacy_v4` | 1.2 GB | superseded weights |
| pre-retrain 9c heads | 360 MB | `A*_svd_9c.pt` etc., replaced by `A1/A2/A3_9c.pt` |
| `train/`, `qwen/`, `docs/`, `run_finetuning.sh`, `mine_hard.py`, `smoke_*` | — | training-only code |
| `pespc/` minus `head.py` | — | feature extraction, dev-split and verification tooling |

Kept deliberately: **all 8 detector checkpoints** (A3 + gsd add 1.3 GB of 23 GB and preserve
config-selectable fusion), and `gsd/faithful.py` (`gsd_model.py` dispatches on the checkpoint's
`gsd_variant`).

## Relocatable

`config/ensemble9.json` uses paths **relative** to itself, so the tree can be moved or copied
without editing configs. (The older `PAAS_ensemble_v4_inf` used absolute paths into
`PAAS_ensemble_v4`, which quietly made it non-standalone — that is not repeated here.)

Verify at any time:

```bash
python scripts/verify_standalone.py
```
