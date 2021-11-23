# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import Dynamic_imputation_nn
from preprocessing import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse



def main(args):

    seed = args.seed
    dataset = args.dataset
    missing_rate = args.missing_rate
    
    hyperparameters = {'num_mi': args.num_mi, 'm': args.m, 'tau': args.tau}

    data = pd.read_csv('./datasets/{}.csv'.format(dataset), delimiter=',', header=None).values
    if len(data)>10000:
        np.random.seed(seed)
        random_sampled_idx = np.random.choice(len(data), 10000, replace=False)
        data = data[random_sampled_idx]
    
    x = data[:,:-1]
    y = data[:,-1]
    
    x_trnval_o, x_tst_o, y_trnval_o, y_tst_o = train_test_split(x, y, random_state = seed, stratify = y, test_size = 0.5)
    x_trnval, x_tst, x_tst_missing, y_trnval, y_tst = preprocessing(x_trnval_o, x_tst_o, y_trnval_o, y_tst_o, missing_rate, seed)
    
    dim_x = x_trnval.shape[1]
    
    if y_trnval.shape[1] > 2:
        dim_y = y_trnval.shape[1]
    else:
        dim_y = 1
    
    save_path = ('./{0}_{1}_{2}_model'.format(seed, dataset, missing_rate))
    
    print('start:::::::','seed:', seed, 'dataset:', dataset, 'missing_rate:',missing_rate)
    
    model = Dynamic_imputation_nn(dim_x, dim_y, seed)
    model.train_with_dynamic_imputation(x_trnval, y_trnval, save_path, **hyperparameters)
    
    acc = model.get_accuracy(x_tst, y_tst)
    auroc = model.get_auroc(x_tst, y_tst)

    print('seed:', seed, 'dataset:', dataset, 'missing_rate:',missing_rate, 'accuracy:', acc, 'auroc:', auroc)


if __name__ == '__main__':


    arg_parser = argparse.ArgumentParser(description='Dynamic imputation')
    
    arg_parser.add_argument('--seed', help='Random seed', default=27407, type= int)
    arg_parser.add_argument('--dataset', help='Dataset name', choices=['avila', 'letter'], default=256, type=str)
    arg_parser.add_argument('--missing_rate', help='Missing rate of dataset', default=30, type=float)
    arg_parser.add_argument('--num_mi', help='Number of multiple imputation for validation set', default=5, type=int)
    arg_parser.add_argument('--m', help='Number of imputations to calculate imputation uncertainty', default=10, type=int)
    arg_parser.add_argument('--tau', help='Threshold of imputation uncertainty', default=0.05, type=float)
    
    args = arg_parser.parse_args()
    
    main(args)