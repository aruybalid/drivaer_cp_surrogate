## 2.1 Surrogate Training

## What the Stage Does
1. Trains a PointNet Regressor on the processed (decimated) data. Uses the model coded in `../utils/model.py`.

## How to Run
Move the `processed` data from the previous `output` folder to this stages `references` folder. Pick either one of these models:

```bash
python ../../utils/train.py --model pointnet
python train.py --model mlp_no_global --epochs 100
python train.py --model mlp_with_params --hidden 256
python train.py --model pointnet_with_params --batch_size 4
```

Batch size, step size and number of epochs are provided as arguments.