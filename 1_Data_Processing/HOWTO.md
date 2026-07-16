## 1 Data Processing
Load, inspect and decimate (downsample) a surface geometry dataset with surface variables. Then store the processed data to disk.

## What the Stage Does
1. Plot the raw data from the `./references` folder.
2. preprocess raw data: deciminate and select variables (`CpMeanTrim`). Store to disk in `./output/processed`
3. Plot the processed data

## How to Run
First make sure the data is in this stage's `./reference` folder.
```bash
python ../utils/plot_data.py
python ../utils/preprocess.py --target_faces 5000
python ../utils/plot_processed.py
```

## Human Review
Human reviews processed data in `./output/processed` before continuing to the next step. Do not continue without human instruction.