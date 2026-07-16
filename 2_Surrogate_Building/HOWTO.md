## What the Stage Does
1. Plot the raw data from the `./data` folder
2. preprocess raw data: deciminate and select variables (`CpMeanTrim`)
3. Plot the processed data

## How to Run
```bash
python ../utils/plot_data.py
python ../utils/preprocess.py --target_faces 5000
python ../utils/plot_processed.py
```