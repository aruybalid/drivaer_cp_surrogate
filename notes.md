# Training Script Arguments (`train.py`)

## Overview
This document summarizes the command-line arguments for `train.py` and key training concepts.

---

## Command-Line Arguments

| Argument            | Default     | Description                                                                 | Effect on Training |
|---------------------|-------------|-----------------------------------------------------------------------------|--------------------|
| `--processed_dir`   | `./processed` | Path to folder containing the preprocessed `.pt` files                     | — |
| `--epochs`          | `150`       | Number of full passes through the training data                             | More epochs → more training, risk of overfitting |
| `--batch_size`      | `2`         | Number of samples processed together before one weight update               | Affects gradient noise, memory, and generalization |
| `--lr`              | `1e-3`      | Learning rate — size of the weight update step                              | Too high = unstable; too low = slow convergence |
| `--device`          | `cuda` if available, else `cpu` | Device used for training | — |

---

## Key Concepts

### Batch Size (`--batch_size`)
- Controls how many meshes are processed together in one forward/backward pass.
- Smaller batches (e.g. 2) produce noisier gradients → often better generalization.
- Larger batches (e.g. 8) produce smoother gradients but can hurt generalization.
- Also affects memory usage and padding behavior in `collate_fn`.

### Learning Rate (`--lr`)
- Determines the magnitude of weight updates.
- Default `1e-3` (0.001) is a reasonable starting point for Adam.

### Linear Scaling Rule
When increasing the batch size, the learning rate should be scaled proportionally to maintain similar effective step sizes:

```
new_lr = old_lr × (new_batch_size / old_batch_size)
```

**Examples** (starting from `batch_size=2`, `lr=0.001`):

| New Batch Size | Scaling Factor | Recommended New LR |
|----------------|----------------|--------------------|
| 2              | ×1             | 0.001              |
| 4              | ×2             | 0.002              |
| 8              | ×4             | 0.004              |

**Caveat**: The rule works only up to a certain batch size. Beyond that, simply scaling the learning rate linearly may stop helping or even hurt performance.

---

## Recommended Starting Points (for this small dataset)

- `batch_size=2`, `lr=0.001` (current default) — good starting point
- `batch_size=4`, `lr=0.002` — slightly larger, smoother gradients
- Avoid very large learning rates (`> 0.01`) unless carefully tuned

---

## Validation (`val_runs`)

- The last 2 runs (`run_445`, `run_474`) are used as a validation set.
- Validation loss is computed **every epoch**.
- The model with the best `val_loss` is saved automatically (`models/best_model.pt`).
- `val_loss` eventually plateaus or rises when the model starts overfitting.

---

## Training, Validation, and Final Testing Workflow

### 1. Data Split
- **8 training runs** (`run_80`, `run_98`, `run_102`, `run_208`, `run_213`, `run_281`, `run_397`, `run_425`).
- **2 validation runs** (`run_445`, `run_474`) — held back during training.

### 2. Decimation (applied to all processed data)
- All data used for training and per-epoch validation is decimated to ~15,000 faces (`target_faces=15000`).
- This keeps memory usage manageable and ensures consistent input size for the model.

### 3. Per-Epoch Validation (Intermediate Testing)
- After every training epoch, the model is evaluated on the 2 held-out validation runs.
- The validation loss (`val_loss`) is used to:
  - Monitor training convergence.
  - Detect overfitting (when `val_loss` starts rising while `train_loss` keeps decreasing).
  - Automatically save the best model (`models/best_model.pt`).
- **Important**: These validation runs are **not fully blind**. They are used to guide training decisions and model selection.

### 4. Final Blind Test on `run_1`
- After training is complete, a completely separate test is performed on `run_1`.
- This run was **never used** during training or validation.
- It can be run at **full resolution** (no decimation) by passing a very large `target_faces` value (e.g. `1000000000`).
- This constitutes the true, unbiased evaluation of the surrogate’s ability to generalize to entirely new geometries.

### Summary of the Two Testing Stages

| Stage                    | Runs Used          | Decimation       | Purpose                                      | Fully Blind? |
|--------------------------|--------------------|------------------|----------------------------------------------|--------------|
| Per-epoch validation     | `run_445`, `run_474` | Yes (~5k faces) | Monitor convergence, save best model         | No           |
| Final test (after training) | `run_1`          | Optional (can be full resolution) | True generalization test on unseen geometry | Yes          |

---

## Notes

- With only 8 training samples, the possible batch sizes are limited (2, 4, or 8).
- The linear scaling rule is less critical here but still useful as a guideline.
- Monitor both `train_loss` and `val_loss` to detect overfitting.
- The final test on `run_1` (especially at full resolution) is the most meaningful measure of the surrogate’s practical value.