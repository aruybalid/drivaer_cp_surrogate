## 2.1 Surrogate Training

## What the Stage Does
1. Trains a PointNet Regressor on the processed (decimated) data. Uses the model coded in `../utils/model.py`.

## How to Run
Move the `processed` data from the previous `output` folder to this stages `references` folder.

```bash
python ../../utils/train.py --epochs 50 --batch_size 4 --lr 2e-3
```

Batch size, step size and number of epochs are provided as arguments.