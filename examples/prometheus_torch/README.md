# Prometheus Torch

Example harness for Prometheus model reproduced in Torch.
Prodives script to train TemoralPrediction model in Keras and Torch,
as well as evaluation and comparison between two models.

The example bundle contains a reproduction script:
the script trains a random intialized Prometheus model over each datetime labeled CSV data file.
It's important to note that the final operation before prediction is un-normalizing the 
model outputs using statistics gathered over training data. 
The script saves checkpoints after 1 epoch on each dataset file, saving both torch 
model weights and a json with the current un-normalization terms.

Using the `npm1_pwr_model.keras` checkpoint of Prometheus as a baseline,
the reproduced torch model works comparably with the keras model. 

<img width="1500" height="450" alt="test03_2025-09-18" src="https://github.com/user-attachments/assets/71f762d0-0119-4cda-9af7-ef8cfaf73a64" />

Select an early checkpoint of the reproduced model for tests with APEIRON. 
The harness checkpoints both the weights and the stats for comparisons against the base Prometheus model.

## Getting Started

Place the data and model weights under `examples/prometheus_torch/data/` 
```
data/
├── test
│   ├── 2025-03-20.csv
│   ├── 2025-05-12.csv
│   ├── 2025-07-23.csv
│   └── 2025-09-18.csv
├── train
│   ├── 2025-02-27.csv
│   ├── 2025-03-12.csv
│   ├── 2025-03-19.csv
│   ├── 2025-03-27.csv
│   ├── 2025-04-23.csv
│   ├── 2025-04-28.csv
│   ├── 2025-04-30.csv
│   ├── 2025-05-20.csv
│   ├── 2025-06-02.csv
│   ├── 2025-06-04.csv
│   ├── 2025-06-10.csv
│   ├── 2025-06-12.csv
│   ├── 2025-06-26.csv
│   ├── 2025-07-21.csv
│   ├── 2025-07-22.csv
│   ├── 2025-07-30.csv
│   ├── 2025-07-31.csv
│   ├── 2025-08-25.csv
│   ├── 2025-08-26.csv
│   ├── 2025-09-02.csv
│   ├── 2025-09-16.csv
│   ├── 2025-09-17.csv
│   └── 2025-09-25.csv
├── npm1_pwr_config.pkl
├── npm1_pwr_model.h5
└── npm1_pwr_model.keras

3 directories, 30 files
```

### Reproduce Prometheus model and save as .pt
```
cd ./examples/prometheus_torch/
python reproduce_prometheus.py train --save ./output/prometheus_torch/reproduced_prometheus.pt
```

### Compare model checkpoint against base model on test set
```
python reproduce_prometheus.py compare --model ./data/npm1_pwr_model.keras --torch-model ./output/apeiron/drift_adaptation_5.pt
```
