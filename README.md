# dynamic_imputation
Source code for the paper: Dynamic imputation for improved training of neural network with missing values

This repository provides dynamic imputation implementations using two UCI datasets:
- Avila (https://archive.ics.uci.edu/ml/datasets/Avila)
- Letter (https://archive.ics.uci.edu/ml/datasets/letter+recognition)

## Components
- **preprocessing.py** - data preprocessing functions
- **model.py** - model architecture with dynamic imputation
- **main.py** - script for model training & evaluation

## Dependencies
- **Python**
- **TensorFlow**
- **scikit-learn**
- **NumPy**
- **SciPy**

## Run Code Example
```shell
$ python main.py --seed 0 --dataset avila --missing_rate 30 --num_mi 5 --m 10 --tau 0.05
```

## Citation
TBU
