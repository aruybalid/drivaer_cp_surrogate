## 2.2 Surrogate Testing

## What the Stage Does
1. runs inference on the trained model, using unseen surface geometries.
2. Plot the results in an interactive plot.

## How to Run
Make sure the model and the test data are in `./references`.
```bash
python ../../utils/inference.py --vtp_path ./references/data/run_2/boundary_2.vtp --out ./output/predicted_decimated-5000_hl-128_run_2.vtp --target_faces=10000000000
python ../../utils/plot_prediction.py --vtp ./output/predicted_decimated-5000_hl-128_run_2.vtp
```

Make sure to use a unique name for the output (vtp)

## Human Review